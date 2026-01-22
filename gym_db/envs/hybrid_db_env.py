# gym_db/envs/hybrid_db_env.py
"""
混合数据库环境
支持索引和物化视图的联合强化学习环境
"""

import collections
import copy
import logging
import random
from typing import Dict, List, Set, Tuple, Any, Optional

import gym
import numpy as np

from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.materialized_view import MaterializedView
from index_selection_evaluation.selection.utils import b_to_mb

from .db_env_v1 import DBEnvV1


class HybridDBEnv(DBEnvV1):
    """
    混合数据库环境
    扩展DBEnvV1以支持索引和物化视图的联合操作
    """
    
    def __init__(self, environment_type: EnvironmentType = EnvironmentType.TRAINING, 
                 config: Dict[str, Any] = None):
        """
        初始化混合数据库环境
        
        Args:
            environment_type: 环境类型（训练/验证/测试）
            config: 配置字典，包含:
                - mv_candidates: 物化视图候选列表
                - index_candidates: 索引候选列表（可选，用于替代globally_indexable_columns）
                - mv_storage_costs: 物化视图存储成本
                - action_manager: 混合动作管理器
                - observation_manager: 混合观察管理器
                - reward_calculator: 奖励计算器
        """
        # 保存物化视图相关配置
        self.mv_candidates = config.get("mv_candidates", [])
        self.index_candidates = config.get("index_candidates", [])
        self.mv_storage_costs = config.get("mv_storage_costs", 
                                           [mv.estimated_size for mv in self.mv_candidates])
        
        # 当前选中的物化视图
        self.current_mvs: Set[MaterializedView] = set()
        
        # 物化视图存储消耗
        self.mv_storage_consumption = 0
        self.index_storage_consumption = 0
        
        # 查询重写器（用于物化视图）
        self.query_rewriter = config.get("query_rewriter", None)
        
        # 调用父类初始化
        super().__init__(environment_type, config)
        
        # 重新设置成本评估器以支持物化视图
        if self.query_rewriter:
            self.cost_evaluation = CostEvaluation(
                self.connector, 
                query_rewriter=self.query_rewriter
            )
        
        logging.info(f"HybridDBEnv initialized with {len(self.mv_candidates)} MV candidates")
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始观察
        """
        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken
        
        initial_observation = self._init_modifiable_state()
        
        return initial_observation
    
    def _init_modifiable_state(self) -> np.ndarray:
        """
        初始化可修改状态
        
        Returns:
            初始观察
        """
        # 重置索引和物化视图
        self.current_indexes = set()
        self.current_mvs = set()
        self.steps_taken = 0
        self.current_storage_consumption = 0
        self.index_storage_consumption = 0
        self.mv_storage_consumption = 0
        self.reward_calculator.reset()
        
        # 清理数据库中的索引和物化视图
        self.connector.drop_indexes()
        self._drop_all_materialized_views()
        
        # 选择工作负载
        if len(self.workloads) == 0:
            self.workloads = copy.copy(self.config["workloads"])
        
        if self.environment_type == EnvironmentType.TRAINING:
            if self.similar_workloads:
                self.current_workload = self.workloads.pop(0 + self.env_id * 200)
            else:
                self.current_workload = self.rnd.choice(self.workloads)
        else:
            self.current_workload = self.workloads[self.current_workload_idx % len(self.workloads)]
        
        self.current_budget = self.current_workload.budget
        self.previous_cost = None
        
        # 获取初始有效动作
        self.valid_actions = self.action_manager.get_initial_valid_actions(
            self.current_workload, self.current_budget
        )
        
        # 更新环境状态
        environment_state = self._update_return_env_state(init=True)
        
        # 初始化观察管理器
        state_fix_for_episode = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs,
        }
        self.observation_manager.init_episode(state_fix_for_episode)
        
        # 获取初始观察
        initial_observation = self.observation_manager.get_observation(environment_state)
        
        return initial_observation
    
    def step(self, action: int, start: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作ID
            start: 是否为起始步骤
            
        Returns:
            (观察, 奖励, 是否结束, 信息)
        """
        # 验证动作
        self._step_asserts_hybrid(action)
        
        self.steps_taken += 1
        
        # 判断动作类型并执行
        action_type, type_specific_id = self.action_manager.get_action_type(action)
        
        if action_type == "INDEX":
            return self._execute_index_action(action, type_specific_id)
        else:
            return self._execute_mv_action(action, type_specific_id)
    
    def _step_asserts_hybrid(self, action: int):
        """混合环境的动作验证"""
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert (
            self.valid_actions[action] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {action}"
    
    def _execute_index_action(self, action: int, index_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行索引创建动作
        
        Args:
            action: 动作ID
            index_id: 索引候选中的ID
            
        Returns:
            (观察, 奖励, 是否结束, 信息)
        """
        old_index_size = 0
        
        # 获取索引对象
        if self.index_candidates:
            new_index = self.index_candidates[index_id]
        else:
            new_index = Index(self.globally_indexable_columns[index_id])
        
        # 处理多列索引的父索引
        if not new_index.is_single_column():
            parent_index = Index(new_index.columns[:-1])
            
            for index in self.current_indexes:
                if index == parent_index:
                    old_index_size = index.estimated_size
                    break
            
            if parent_index in self.current_indexes:
                self.current_indexes.remove(parent_index)
        
        # 添加新索引
        self.current_indexes.add(new_index)
        
        # 更新存储消耗
        self._update_index_storage(new_index, old_index_size)
        
        # 更新环境状态
        environment_state = self._update_return_env_state(
            init=False, 
            new_index=new_index, 
            old_index_size=old_index_size
        )
        
        # 获取观察
        current_observation = self.observation_manager.get_observation(environment_state)
        
        # 更新有效动作
        self.valid_actions, is_valid_action_left = self.action_manager.update_valid_actions(
            action, self.current_budget, self.current_storage_consumption
        )
        
        # 检查是否结束
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left
        
        # 计算奖励
        reward = self.reward_calculator.calculate_reward(environment_state)
        
        # 报告性能（如果结束）
        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            self._report_episode_performance(environment_state)
            self.current_workload_idx += 1
        
        return current_observation, reward, episode_done, {"action_mask": self.valid_actions}
    
    def _execute_mv_action(self, action: int, mv_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行物化视图创建动作
        
        Args:
            action: 动作ID
            mv_id: 物化视图候选中的ID
            
        Returns:
            (观察, 奖励, 是否结束, 信息)
        """
        mv = self.mv_candidates[mv_id]
        
        # 检查物化视图是否已存在
        if mv in self.current_mvs:
            logging.warning(f"MV {mv.name} already exists, skipping")
        else:
            # 添加物化视图
            self.current_mvs.add(mv)
            
            # 创建物化视图（在数据库中）
            try:
                self._create_materialized_view(mv)
            except Exception as e:
                logging.warning(f"Failed to create MV {mv.name}: {e}")
            
            # 更新存储消耗
            self._update_mv_storage(mv)
        
        # 更新环境状态
        environment_state = self._update_return_env_state(
            init=False,
            new_mv=mv
        )
        
        # 获取观察
        current_observation = self.observation_manager.get_observation(environment_state)
        
        # 更新有效动作
        self.valid_actions, is_valid_action_left = self.action_manager.update_valid_actions(
            action, self.current_budget, self.current_storage_consumption
        )
        
        # 检查是否结束
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left
        
        # 计算奖励
        reward = self.reward_calculator.calculate_reward(environment_state)
        
        # 报告性能（如果结束）
        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            self._report_episode_performance(environment_state)
            self.current_workload_idx += 1
        
        return current_observation, reward, episode_done, {"action_mask": self.valid_actions}
    
    def _update_index_storage(self, new_index: Index, old_index_size: int):
        """更新索引存储消耗"""
        if new_index.estimated_size:
            self.index_storage_consumption += new_index.estimated_size - old_index_size
        self.current_storage_consumption = self.index_storage_consumption + self.mv_storage_consumption
    
    def _update_mv_storage(self, mv: MaterializedView):
        """更新物化视图存储消耗"""
        mv_size = mv.estimated_size if mv.estimated_size else 1024 * 1024  # 默认1MB
        self.mv_storage_consumption += mv_size
        self.current_storage_consumption = self.index_storage_consumption + self.mv_storage_consumption
    
    def _create_materialized_view(self, mv: MaterializedView):
        """
        在数据库中创建物化视图
        
        Args:
            mv: 物化视图对象
        """
        try:
            create_sql = f"CREATE MATERIALIZED VIEW IF NOT EXISTS {mv.name} AS {mv.definition_sql}"
            self.connector.exec_only(create_sql)
            logging.debug(f"Created materialized view: {mv.name}")
        except Exception as e:
            logging.warning(f"Failed to create MV {mv.name}: {e}")
    
    def _drop_materialized_view(self, mv: MaterializedView):
        """
        删除物化视图
        
        Args:
            mv: 物化视图对象
        """
        try:
            drop_sql = f"DROP MATERIALIZED VIEW IF EXISTS {mv.name}"
            self.connector.exec_only(drop_sql)
            logging.debug(f"Dropped materialized view: {mv.name}")
        except Exception as e:
            logging.warning(f"Failed to drop MV {mv.name}: {e}")
    
    def _drop_all_materialized_views(self):
        """删除所有物化视图"""
        for mv in self.mv_candidates:
            self._drop_materialized_view(mv)
        self.current_mvs.clear()
        self.mv_storage_consumption = 0
    
    def _update_return_env_state(self, 
                                 init: bool, 
                                 new_index: Index = None, 
                                 old_index_size: int = None,
                                 new_mv: MaterializedView = None,
                                 print_flag: bool = False) -> Dict[str, Any]:
        """
        更新并返回环境状态
        
        Args:
            init: 是否为初始化
            new_index: 新索引（如果有）
            old_index_size: 旧索引大小
            new_mv: 新物化视图（如果有）
            print_flag: 是否打印
            
        Returns:
            环境状态字典
        """
        # 计算成本（考虑物化视图）
        total_costs, plans_per_query, costs_per_query = self._calculate_cost_with_mvs()
        
        if not init:
            self.previous_cost = self.current_costs
            self.previous_storage_consumption = self.current_storage_consumption
        
        self.current_costs = total_costs
        
        if init:
            self.initial_costs = total_costs
        
        # 处理新索引
        new_index_size = None
        if new_index is not None:
            if new_index.estimated_size and old_index_size is not None:
                new_index_size = new_index.estimated_size - old_index_size
                if new_index_size == 0:
                    new_index_size = 1
        
        # 处理新物化视图
        new_mv_size = None
        if new_mv is not None:
            new_mv_size = new_mv.estimated_size if new_mv.estimated_size else 1024 * 1024
        
        # 获取索引和MV状态
        index_status = self._get_index_status()
        mv_status = self._get_mv_status()
        
        # 构建环境状态
        environment_state = {
            # 基础状态
            "action_status": self.action_manager.current_action_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            
            # 索引相关
            "index_status": index_status,
            "index_storage_consumption": self.index_storage_consumption,
            "new_index_size": new_index_size,
            "current_indexes": self.current_indexes,
            
            # 物化视图相关
            "mv_status": mv_status,
            "mv_storage_consumption": self.mv_storage_consumption,
            "new_mv_size": new_mv_size,
            "current_mvs": self.current_mvs,
            
            # 查询计划和成本
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query,
            
            # 工作负载
            "workload": self.current_workload,
        }
        
        return environment_state
    
    def _calculate_cost_with_mvs(self) -> Tuple[float, List, List]:
        """
        计算考虑物化视图的成本
        
        Returns:
            (总成本, 每个查询的计划, 每个查询的成本)
        """
        try:
            total_costs, plans_per_query, costs_per_query = self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload, 
                self.current_indexes, 
                list(self.current_mvs),
                store_size=True
            )
        except Exception as e:
            logging.warning(f"Cost calculation with MVs failed: {e}, falling back to index-only")
            total_costs, plans_per_query, costs_per_query = self.cost_evaluation.calculate_cost_and_plans(
                self.current_workload,
                self.current_indexes,
                store_size=True
            )
        
        return total_costs, plans_per_query, costs_per_query
    
    def _get_index_status(self) -> np.ndarray:
        """获取索引状态向量"""
        num_index_actions = self.action_manager.number_of_index_actions
        status = np.zeros(num_index_actions)
        
        for i, index_candidate in enumerate(self.index_candidates[:num_index_actions]):
            if index_candidate in self.current_indexes:
                status[i] = 1.0
        
        return status
    
    def _get_mv_status(self) -> np.ndarray:
        """获取物化视图状态向量"""
        num_mv_actions = self.action_manager.number_of_mv_actions
        status = np.zeros(num_mv_actions)
        
        for i, mv in enumerate(self.mv_candidates[:num_mv_actions]):
            if mv in self.current_mvs:
                status[i] = 1.0
        
        return status
    
    def _report_episode_performance(self, environment_state: Dict[str, Any]):
        """报告episode性能"""
        episode_performance = {
            "achieved_cost": self.current_costs / self.initial_costs * 100,
            "memory_consumption": self.current_storage_consumption,
            "index_memory": self.index_storage_consumption,
            "mv_memory": self.mv_storage_consumption,
            "available_budget": self.current_budget,
            "evaluated_workload": self.current_workload,
            "indexes": self.current_indexes,
            "materialized_views": self.current_mvs,
            "num_indexes": len(self.current_indexes),
            "num_mvs": len(self.current_mvs),
        }
        
        output = (
            f"Evaluated Workload ({self.environment_type}): {self.current_workload}\n    "
            f"Initial cost: {self.initial_costs:,.2f}, now: {self.current_costs:,.2f} "
            f"({episode_performance['achieved_cost']:.2f}%). "
            f"Reward: {self.reward_calculator.accumulated_reward}.\n    "
            f"Storage: {b_to_mb(self.current_storage_consumption):.2f} MB "
            f"(Index: {b_to_mb(self.index_storage_consumption):.2f} MB, "
            f"MV: {b_to_mb(self.mv_storage_consumption):.2f} MB)\n    "
            f"Configuration: {len(self.current_indexes)} indexes, {len(self.current_mvs)} MVs\n"
        )
        logging.warning(output)
        
        self.episode_performances.append(episode_performance)
    
    def _calculate_total_storage(self) -> int:
        """计算总存储消耗"""
        return self.index_storage_consumption + self.mv_storage_consumption
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        Returns:
            当前配置字典
        """
        return {
            "indexes": list(self.current_indexes),
            "materialized_views": list(self.current_mvs),
            "index_storage": self.index_storage_consumption,
            "mv_storage": self.mv_storage_consumption,
            "total_storage": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "initial_cost": self.initial_costs,
            "cost_reduction": (self.initial_costs - self.current_costs) / self.initial_costs * 100 
                             if self.initial_costs > 0 else 0
        }
    
    def close(self):
        """关闭环境"""
        # 清理物化视图
        self._drop_all_materialized_views()
        
        # 清理索引
        self.connector.drop_indexes()
        
        logging.info("HybridDBEnv closed")


class HybridDBEnvWithReplay(HybridDBEnv):
    """
    带经验回放的混合数据库环境
    支持保存和加载经验
    """
    
    def __init__(self, environment_type: EnvironmentType = EnvironmentType.TRAINING,
                 config: Dict[str, Any] = None):
        super().__init__(environment_type, config)
        
        # 经验缓冲区
        self.experience_buffer = []
        self.max_buffer_size = config.get("max_buffer_size", 10000)
    
    def step(self, action: int, start: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作并保存经验"""
        # 保存当前状态
        current_state = self._get_current_state_for_replay()
        
        # 执行动作
        observation, reward, done, info = super().step(action, start)
        
        # 保存经验
        experience = {
            "state": current_state,
            "action": action,
            "reward": reward,
            "next_state": observation,
            "done": done
        }
        self._add_experience(experience)
        
        return observation, reward, done, info
    
    def _get_current_state_for_replay(self) -> Dict[str, Any]:
        """获取用于回放的当前状态"""
        return {
            "indexes": list(self.current_indexes),
            "mvs": list(self.current_mvs),
            "cost": self.current_costs,
            "storage": self.current_storage_consumption
        }
    
    def _add_experience(self, experience: Dict[str, Any]):
        """添加经验到缓冲区"""
        if len(self.experience_buffer) >= self.max_buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(experience)
    
    def sample_experiences(self, batch_size: int) -> List[Dict[str, Any]]:
        """采样经验"""
        if len(self.experience_buffer) < batch_size:
            return self.experience_buffer
        return random.sample(self.experience_buffer, batch_size)


def create_hybrid_db_env(environment_type: EnvironmentType,
                        config: Dict[str, Any],
                        use_replay: bool = False) -> HybridDBEnv:
    """
    创建混合数据库环境的工厂函数
    
    Args:
        environment_type: 环境类型
        config: 配置字典
        use_replay: 是否使用经验回放
        
    Returns:
        HybridDBEnv实例
    """
    if use_replay:
        return HybridDBEnvWithReplay(environment_type, config)
    else:
        return HybridDBEnv(environment_type, config)

