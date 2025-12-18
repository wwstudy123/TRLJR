# balance/hybrid_observation_manager.py
"""
混合观察管理器模块
扩展状态管理以包含索引和物化视图状态
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
from gym import spaces

from .observation_manager import (
    ObservationManager, 
    EmbeddingObservationManager,
    VERY_HIGH_BUDGET
)


class HybridObservationManager(EmbeddingObservationManager):
    """
    混合观察管理器
    扩展EmbeddingObservationManager以支持索引和物化视图的联合状态
    """
    
    def __init__(self, number_of_actions: int, config: Dict[str, Any]):
        """
        初始化混合观察管理器
        
        Args:
            number_of_actions: 总动作数（索引 + 物化视图）
            config: 配置字典，包含:
                - workload_embedder: 工作负载嵌入器
                - workload_size: 工作负载大小
                - number_of_index_actions: 索引动作数量
                - number_of_mv_actions: 物化视图动作数量
        """
        # 先保存索引和MV动作数量
        self.number_of_index_actions = config.get("number_of_index_actions", number_of_actions)
        self.number_of_mv_actions = config.get("number_of_mv_actions", 0)
        
        # 调用父类初始化
        super().__init__(number_of_actions, config)
        
        # 重新计算特征数量，包含物化视图状态
        self.number_of_features = (
            self.number_of_actions +                              # 索引+物化视图动作状态
            (self.representation_size * self.workload_size) +     # 工作负载嵌入
            self.workload_size +                                  # 查询频率
            self.number_of_index_actions +                        # 当前索引状态（详细）
            self.number_of_mv_actions +                           # 当前物化视图状态（详细）
            1 +                                                   # 预算
            2 +                                                   # 索引存储消耗 + MV存储消耗
            1 +                                                   # 初始成本
            1                                                     # 当前成本
        )
        
        # 是否每次观察都更新嵌入
        self.UPDATE_EMBEDDING_PER_OBSERVATION = config.get("update_embedding_per_observation", False)
        
        # 物化视图相关状态
        self.mv_status = np.zeros(self.number_of_mv_actions)
        self.index_status = np.zeros(self.number_of_index_actions)
        
        logging.info(f"HybridObservationManager initialized with {self.number_of_features} features "
                    f"({self.number_of_index_actions} index actions, {self.number_of_mv_actions} MV actions)")
    
    def init_episode(self, state_fix_for_episode: Dict[str, Any]):
        """
        初始化episode
        
        Args:
            state_fix_for_episode: episode固定状态
        """
        # 调用父类初始化
        self._init_episode(state_fix_for_episode)
        
        # 初始化工作负载嵌入
        if not self.UPDATE_EMBEDDING_PER_OBSERVATION:
            if hasattr(self.workload_embedder, 'get_embeddings'):
                self.workload_embedding = np.array(
                    self.workload_embedder.get_embeddings(state_fix_for_episode["workload"])
                )
            else:
                self.workload_embedding = None
        else:
            self.workload_embedding = None
        
        # 重置索引和MV状态
        self.index_status = np.zeros(self.number_of_index_actions)
        self.mv_status = np.zeros(self.number_of_mv_actions)
    
    def _init_episode(self, state_fix_for_episode: Dict[str, Any]):
        """初始化episode的内部方法"""
        episode_workload = state_fix_for_episode["workload"]
        self.frequencies = np.array(self._get_frequencies_from_workload(episode_workload))
        
        self.episode_budget = state_fix_for_episode.get("budget")
        if self.episode_budget is None:
            self.episode_budget = VERY_HIGH_BUDGET
        
        self.initial_cost = state_fix_for_episode.get("initial_cost", 0)
    
    def get_observation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        获取观察
        
        Args:
            environment_state: 环境状态字典
            
        Returns:
            观察向量
        """
        # 获取工作负载嵌入
        if self.UPDATE_EMBEDDING_PER_OBSERVATION:
            plans = environment_state.get("plans_per_query", environment_state.get("workload"))
            workload_embedding = np.array(self.workload_embedder.get_embeddings(plans))
        else:
            if self.workload_embedding is None:
                plans = environment_state.get("plans_per_query", environment_state.get("workload"))
                self.workload_embedding = np.array(self.workload_embedder.get_embeddings(plans))
            workload_embedding = self.workload_embedding
        
        # 基础动作状态
        action_status = np.array(environment_state.get("action_status", 
                                                       [0] * self.number_of_actions))
        
        # 索引详细状态
        index_status = np.array(environment_state.get("index_status", 
                                                      self.index_status))
        
        # 物化视图详细状态
        mv_status = np.array(environment_state.get("mv_status", 
                                                   self.mv_status))
        
        # 存储消耗
        index_storage = environment_state.get("index_storage_consumption", 0)
        mv_storage = environment_state.get("mv_storage_consumption", 0)
        total_storage = environment_state.get("current_storage_consumption", 
                                              index_storage + mv_storage)
        
        # 成本信息
        current_cost = environment_state.get("current_cost", self.initial_cost)
        
        # 组合完整观察
        observation = np.concatenate([
            action_status,                          # 动作状态
            workload_embedding.flatten(),           # 工作负载嵌入
            self.frequencies,                       # 查询频率
            index_status,                           # 索引详细状态
            mv_status,                              # 物化视图详细状态
            [self.episode_budget],                  # 预算
            [index_storage, mv_storage],            # 分别的存储消耗
            [self.initial_cost],                    # 初始成本
            [current_cost]                          # 当前成本
        ])
        
        return observation.astype(np.float32)
    
    @staticmethod
    def _get_frequencies_from_workload(workload) -> List[float]:
        """从工作负载获取查询频率"""
        frequencies = []
        for query in workload.queries:
            frequencies.append(query.frequency)
        return frequencies


class HybridPlanEmbeddingObservationManager(HybridObservationManager):
    """
    基于查询计划嵌入的混合观察管理器
    每次观察都更新嵌入
    """
    
    def __init__(self, number_of_actions: int, config: Dict[str, Any]):
        config["update_embedding_per_observation"] = True
        super().__init__(number_of_actions, config)


class HybridWorkloadEmbeddingObservationManager(HybridObservationManager):
    """
    基于工作负载嵌入的混合观察管理器
    嵌入在episode开始时计算，不随步骤更新
    """
    
    def __init__(self, number_of_actions: int, config: Dict[str, Any]):
        config["update_embedding_per_observation"] = False
        super().__init__(number_of_actions, config)


class HybridObservationManagerWithCost(HybridObservationManager):
    """
    带成本信息的混合观察管理器
    包含每个查询的成本信息
    """
    
    def __init__(self, number_of_actions: int, config: Dict[str, Any]):
        super().__init__(number_of_actions, config)
        
        # 添加每个查询的成本特征
        self.number_of_features += self.workload_size  # 每个查询的成本
        
        logging.info(f"HybridObservationManagerWithCost: total features = {self.number_of_features}")
    
    def get_observation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """获取带成本信息的观察"""
        # 获取基础观察
        base_observation = super().get_observation(environment_state)
        
        # 添加每个查询的成本
        costs_per_query = np.array(environment_state.get("costs_per_query", 
                                                         [0] * self.workload_size))
        
        # 组合观察
        observation = np.concatenate([base_observation, costs_per_query])
        
        return observation.astype(np.float32)


class HybridObservationManagerWithMVBenefit(HybridObservationManager):
    """
    带物化视图收益信息的混合观察管理器
    包含每个物化视图的预估收益
    """
    
    def __init__(self, number_of_actions: int, config: Dict[str, Any]):
        super().__init__(number_of_actions, config)
        
        # 添加MV收益特征
        self.number_of_features += self.number_of_mv_actions  # 每个MV的预估收益
        
        # MV收益估算器
        self.mv_benefit_estimator = config.get("mv_benefit_estimator", None)
        
        logging.info(f"HybridObservationManagerWithMVBenefit: total features = {self.number_of_features}")
    
    def get_observation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """获取带MV收益信息的观察"""
        # 获取基础观察
        base_observation = super().get_observation(environment_state)
        
        # 获取MV收益估算
        if self.mv_benefit_estimator is not None:
            mv_benefits = self.mv_benefit_estimator.estimate_benefits(
                environment_state.get("workload"),
                environment_state.get("mv_candidates", [])
            )
        else:
            mv_benefits = environment_state.get("mv_benefits", 
                                                [0] * self.number_of_mv_actions)
        
        mv_benefits = np.array(mv_benefits)
        
        # 组合观察
        observation = np.concatenate([base_observation, mv_benefits])
        
        return observation.astype(np.float32)


class HybridObservationManagerFull(HybridObservationManager):
    """
    完整的混合观察管理器
    包含所有可能的特征：成本、MV收益、索引收益等
    """
    
    def __init__(self, number_of_actions: int, config: Dict[str, Any]):
        super().__init__(number_of_actions, config)
        
        # 添加额外特征
        self.number_of_features += (
            self.workload_size +                    # 每个查询的成本
            self.number_of_mv_actions +             # MV预估收益
            self.number_of_index_actions +          # 索引预估收益
            1 +                                     # 成本改善率
            1                                       # 配置复杂度
        )
        
        logging.info(f"HybridObservationManagerFull: total features = {self.number_of_features}")
    
    def get_observation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """获取完整观察"""
        # 获取基础观察
        base_observation = super().get_observation(environment_state)
        
        # 每个查询的成本
        costs_per_query = np.array(environment_state.get("costs_per_query", 
                                                         [0] * self.workload_size))
        
        # MV收益
        mv_benefits = np.array(environment_state.get("mv_benefits", 
                                                     [0] * self.number_of_mv_actions))
        
        # 索引收益
        index_benefits = np.array(environment_state.get("index_benefits", 
                                                        [0] * self.number_of_index_actions))
        
        # 成本改善率
        current_cost = environment_state.get("current_cost", self.initial_cost)
        if self.initial_cost > 0:
            cost_improvement = (self.initial_cost - current_cost) / self.initial_cost
        else:
            cost_improvement = 0.0
        
        # 配置复杂度（已选择的索引和MV数量）
        index_status = environment_state.get("index_status", self.index_status)
        mv_status = environment_state.get("mv_status", self.mv_status)
        config_complexity = (np.sum(index_status > 0) + np.sum(mv_status > 0)) / self.number_of_actions
        
        # 组合观察
        observation = np.concatenate([
            base_observation,
            costs_per_query,
            mv_benefits,
            index_benefits,
            [cost_improvement],
            [config_complexity]
        ])
        
        return observation.astype(np.float32)


class MVBenefitEstimator:
    """物化视图收益估算器"""
    
    def __init__(self, cost_evaluation=None):
        """
        初始化收益估算器
        
        Args:
            cost_evaluation: 成本评估器
        """
        self.cost_evaluation = cost_evaluation
        self.benefit_cache = {}
    
    def estimate_benefits(self, workload, mv_candidates: List) -> List[float]:
        """
        估算每个物化视图的收益
        
        Args:
            workload: 工作负载
            mv_candidates: 物化视图候选列表
            
        Returns:
            收益列表
        """
        benefits = []
        
        for mv in mv_candidates:
            benefit = self._estimate_single_mv_benefit(workload, mv)
            benefits.append(benefit)
        
        return benefits
    
    def _estimate_single_mv_benefit(self, workload, mv) -> float:
        """
        估算单个物化视图的收益
        
        Args:
            workload: 工作负载
            mv: 物化视图
            
        Returns:
            收益值
        """
        cache_key = f"{id(workload)}_{mv.name}"
        
        if cache_key in self.benefit_cache:
            return self.benefit_cache[cache_key]
        
        benefit = 0.0
        
        if self.cost_evaluation is not None:
            try:
                # 计算无MV的成本
                cost_without_mv = self.cost_evaluation.calculate_cost(workload, [], [])
                
                # 计算有MV的成本
                cost_with_mv = self.cost_evaluation.calculate_cost(workload, [], [mv])
                
                # 收益 = 成本减少量
                benefit = max(0, cost_without_mv - cost_with_mv)
                
            except Exception as e:
                logging.warning(f"Failed to estimate MV benefit: {e}")
                benefit = 0.0
        else:
            # 简单的启发式估算
            benefit = self._heuristic_benefit_estimate(workload, mv)
        
        self.benefit_cache[cache_key] = benefit
        return benefit
    
    def _heuristic_benefit_estimate(self, workload, mv) -> float:
        """
        启发式收益估算
        
        Args:
            workload: 工作负载
            mv: 物化视图
            
        Returns:
            估算收益
        """
        benefit = 0.0
        mv_sql = mv.definition_sql.upper()
        
        for query in workload.queries:
            query_text = query.text.upper()
            
            # 检查表重叠
            if self._has_table_overlap(query_text, mv_sql):
                benefit += query.frequency * 0.1
            
            # 检查聚合函数重叠
            if self._has_aggregation_overlap(query_text, mv_sql):
                benefit += query.frequency * 0.2
            
            # 检查GROUP BY重叠
            if 'GROUP BY' in query_text and 'GROUP BY' in mv_sql:
                benefit += query.frequency * 0.15
        
        return benefit
    
    def _has_table_overlap(self, query_sql: str, mv_sql: str) -> bool:
        """检查表重叠"""
        import re
        
        def extract_tables(sql):
            pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)'
            matches = re.findall(pattern, sql)
            tables = set()
            for m in matches:
                tables.update(t.lower() for t in m if t)
            return tables
        
        query_tables = extract_tables(query_sql)
        mv_tables = extract_tables(mv_sql)
        
        return len(query_tables & mv_tables) > 0
    
    def _has_aggregation_overlap(self, query_sql: str, mv_sql: str) -> bool:
        """检查聚合函数重叠"""
        agg_funcs = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']
        
        query_aggs = set(agg for agg in agg_funcs if agg in query_sql)
        mv_aggs = set(agg for agg in agg_funcs if agg in mv_sql)
        
        return len(query_aggs & mv_aggs) > 0
    
    def clear_cache(self):
        """清除缓存"""
        self.benefit_cache.clear()


def create_hybrid_observation_manager(number_of_actions: int,
                                     config: Dict[str, Any],
                                     manager_type: str = "basic") -> HybridObservationManager:
    """
    创建混合观察管理器的工厂函数
    
    Args:
        number_of_actions: 总动作数
        config: 配置字典
        manager_type: 管理器类型 ("basic", "plan", "workload", "cost", "mv_benefit", "full")
        
    Returns:
        HybridObservationManager实例
    """
    manager_classes = {
        "basic": HybridObservationManager,
        "plan": HybridPlanEmbeddingObservationManager,
        "workload": HybridWorkloadEmbeddingObservationManager,
        "cost": HybridObservationManagerWithCost,
        "mv_benefit": HybridObservationManagerWithMVBenefit,
        "full": HybridObservationManagerFull
    }
    
    manager_class = manager_classes.get(manager_type, HybridObservationManager)
    
    return manager_class(number_of_actions, config)

