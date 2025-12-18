# balance/hybrid_action_manager.py
"""
混合动作管理器模块
支持索引和物化视图的联合动作空间管理
"""

import copy
import logging
from typing import List, Set, Tuple, Dict, Any, Optional

import numpy as np
from gym import spaces

from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.materialized_view import MaterializedView
from index_selection_evaluation.selection.utils import b_to_mb

from .action_manager import ActionManager

# 动作掩码常量
FORBIDDEN_ACTION_SB3 = -np.inf
ALLOWED_ACTION_SB3 = 0
FORBIDDEN_ACTION_SB2 = 0
ALLOWED_ACTION_SB2 = 1


class HybridActionManager(ActionManager):
    """
    混合动作管理器：支持索引和物化视图的联合动作空间
    
    动作空间结构：
    [0, ..., N_idx-1] : 索引动作
    [N_idx, ..., N_idx+N_mv-1] : 物化视图动作
    """
    
    def __init__(self, 
                 index_candidates: List,
                 mv_candidates: List[MaterializedView],
                 index_storage_costs: List[int],
                 mv_storage_costs: List[int],
                 sb_version: int,
                 max_index_width: int,
                 reenable_indexes: bool = False):
        """
        初始化混合动作管理器
        
        Args:
            index_candidates: 索引候选列表
            mv_candidates: 物化视图候选列表
            index_storage_costs: 索引存储成本列表
            mv_storage_costs: 物化视图存储成本列表
            sb_version: Stable Baselines版本
            max_index_width: 最大索引宽度
            reenable_indexes: 是否允许重新启用索引
        """
        super().__init__(sb_version, max_index_width)
        
        self.index_candidates = index_candidates
        self.mv_candidates = mv_candidates
        self.index_storage_costs = index_storage_costs
        self.mv_storage_costs = mv_storage_costs
        self.reenable_indexes = reenable_indexes
        
        # 动作空间大小
        self.number_of_index_actions = len(index_candidates)
        self.number_of_mv_actions = len(mv_candidates)
        self.number_of_actions = self.number_of_index_actions + self.number_of_mv_actions
        
        # 合并存储成本
        self.action_storage_consumptions = index_storage_costs + mv_storage_costs
        
        # 当前选中的索引和物化视图
        self.current_indexes: Set[Index] = set()
        self.current_mvs: Set[MaterializedView] = set()
        
        # 索引依赖关系映射（用于多列索引）
        self.index_dependency_map = self._build_index_dependency_map()
        
        # 物化视图依赖关系映射（MV可能依赖于某些索引）
        self.mv_dependency_map = self._build_mv_dependency_map()
        
        # 索引到动作ID的映射
        self.index_to_action = {idx: i for i, idx in enumerate(index_candidates)}
        self.mv_to_action = {mv: i + self.number_of_index_actions 
                            for i, mv in enumerate(mv_candidates)}
        
        # 列信息（用于索引相关操作）
        self._setup_column_info()
        
        logging.info(f"HybridActionManager initialized with {self.number_of_index_actions} index actions "
                    f"and {self.number_of_mv_actions} MV actions")
    
    def _setup_column_info(self):
        """设置列信息"""
        self.indexable_columns = set()
        self.column_to_idx = {}
        
        for idx, index_candidate in enumerate(self.index_candidates):
            if hasattr(index_candidate, 'columns'):
                for col in index_candidate.columns:
                    self.indexable_columns.add(col)
                    if col not in self.column_to_idx:
                        self.column_to_idx[col] = len(self.column_to_idx)
        
        self.number_of_columns = len(self.column_to_idx)
    
    def _build_index_dependency_map(self) -> Dict[int, List[int]]:
        """
        构建索引依赖关系映射
        多列索引依赖于其前缀索引
        
        Returns:
            依赖关系映射 {parent_action_id: [dependent_action_ids]}
        """
        dependency_map = {}
        
        for i, index in enumerate(self.index_candidates):
            if not hasattr(index, 'columns') or len(index.columns) <= 1:
                continue
            
            # 查找前缀索引
            prefix_columns = index.columns[:-1]
            for j, other_index in enumerate(self.index_candidates):
                if hasattr(other_index, 'columns') and tuple(other_index.columns) == tuple(prefix_columns):
                    if j not in dependency_map:
                        dependency_map[j] = []
                    dependency_map[j].append(i)
                    break
        
        return dependency_map
    
    def _build_mv_dependency_map(self) -> Dict[int, List[int]]:
        """
        构建物化视图依赖关系映射
        某些物化视图可能在特定索引存在时效果更好
        
        Returns:
            依赖关系映射 {index_action_id: [mv_action_ids]}
        """
        # 简化实现：物化视图暂不依赖索引
        return {}
    
    def get_action_space(self) -> spaces.Discrete:
        """获取动作空间"""
        return spaces.Discrete(self.number_of_actions)
    
    def get_action_type(self, action_id: int) -> Tuple[str, int]:
        """
        判断动作类型
        
        Args:
            action_id: 动作ID
            
        Returns:
            (动作类型, 类型内部ID)
        """
        if action_id < self.number_of_index_actions:
            return "INDEX", action_id
        else:
            return "MATERIALIZED_VIEW", action_id - self.number_of_index_actions
    
    def get_action_object(self, action_id: int):
        """
        获取动作对应的对象（索引或物化视图）
        
        Args:
            action_id: 动作ID
            
        Returns:
            Index或MaterializedView对象
        """
        action_type, type_specific_id = self.get_action_type(action_id)
        
        if action_type == "INDEX":
            return self.index_candidates[type_specific_id]
        else:
            return self.mv_candidates[type_specific_id]
    
    def get_initial_valid_actions(self, workload, budget: Optional[float]) -> np.ndarray:
        """
        获取初始有效动作
        
        Args:
            workload: 工作负载
            budget: 存储预算
            
        Returns:
            有效动作掩码数组
        """
        # 初始化动作状态
        self.current_action_status = [0 for _ in range(self.number_of_columns)]
        
        # 初始化有效动作
        self.valid_actions = [self.FORBIDDEN_ACTION for _ in range(self.number_of_actions)]
        self._remaining_valid_actions = []
        
        # 清空当前选择
        self.current_indexes = set()
        self.current_mvs = set()
        self.current_combinations = set()
        
        # 基于工作负载设置有效动作
        self._valid_actions_based_on_workload(workload)
        
        # 基于预算设置有效动作
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)
        
        return np.array(self.valid_actions)
    
    def update_valid_actions(self, last_action: int, budget: Optional[float], 
                            current_storage_consumption: float) -> Tuple[np.ndarray, bool]:
        """
        更新有效动作
        
        Args:
            last_action: 上一个执行的动作
            budget: 存储预算
            current_storage_consumption: 当前存储消耗
            
        Returns:
            (有效动作掩码数组, 是否还有有效动作)
        """
        action_type, type_specific_id = self.get_action_type(last_action)
        
        if action_type == "INDEX":
            self._update_after_index_action(last_action, type_specific_id)
        else:
            self._update_after_mv_action(last_action, type_specific_id)
        
        # 禁用已选择的动作
        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        if last_action in self._remaining_valid_actions:
            self._remaining_valid_actions.remove(last_action)
        
        # 更新依赖动作
        self._update_dependent_actions(last_action)
        
        # 基于预算更新有效动作
        self._valid_actions_based_on_budget(budget, current_storage_consumption)
        
        is_valid_action_left = len(self._remaining_valid_actions) > 0
        
        return np.array(self.valid_actions), is_valid_action_left
    
    def _update_after_index_action(self, action_id: int, type_specific_id: int):
        """
        索引动作后的更新
        
        Args:
            action_id: 动作ID
            type_specific_id: 索引候选中的ID
        """
        index = self.index_candidates[type_specific_id]
        self.current_indexes.add(index)
        
        # 更新列状态
        if hasattr(index, 'columns'):
            index_width = len(index.columns)
            for i, col in enumerate(index.columns):
                if col in self.column_to_idx:
                    col_idx = self.column_to_idx[col]
                    status_value = 1 / (i + 1)
                    self.current_action_status[col_idx] += status_value
            
            # 添加到当前组合
            self.current_combinations.add(tuple(index.columns))
        
        logging.debug(f"Added index: {index}")
    
    def _update_after_mv_action(self, action_id: int, type_specific_id: int):
        """
        物化视图动作后的更新
        
        Args:
            action_id: 动作ID
            type_specific_id: 物化视图候选中的ID
        """
        mv = self.mv_candidates[type_specific_id]
        self.current_mvs.add(mv)
        
        logging.debug(f"Added materialized view: {mv.name}")
    
    def _update_dependent_actions(self, last_action: int):
        """
        更新依赖动作（启用因依赖关系而可用的新动作）
        
        Args:
            last_action: 上一个执行的动作
        """
        action_type, type_specific_id = self.get_action_type(last_action)
        
        if action_type == "INDEX":
            # 启用依赖于此索引的多列索引
            if type_specific_id in self.index_dependency_map:
                for dependent_action in self.index_dependency_map[type_specific_id]:
                    dependent_index = self.index_candidates[dependent_action]
                    
                    # 检查是否已经选择
                    if hasattr(dependent_index, 'columns'):
                        if tuple(dependent_index.columns) in self.current_combinations:
                            continue
                    
                    # 检查列是否在工作负载中可索引
                    if hasattr(self, 'wl_indexable_columns') and hasattr(dependent_index, 'columns'):
                        if dependent_index.columns[-1] not in self.wl_indexable_columns:
                            continue
                    
                    self.valid_actions[dependent_action] = self.ALLOWED_ACTION
                    if dependent_action not in self._remaining_valid_actions:
                        self._remaining_valid_actions.append(dependent_action)
            
            # 禁用与当前索引冲突的扩展
            self._disable_conflicting_extensions(last_action, type_specific_id)
            
            # 如果启用了重新启用索引，处理相关逻辑
            if self.reenable_indexes:
                self._handle_reenable_indexes(type_specific_id)
            
            # 启用依赖于此索引的物化视图
            if type_specific_id in self.mv_dependency_map:
                for mv_action in self.mv_dependency_map[type_specific_id]:
                    self.valid_actions[mv_action] = self.ALLOWED_ACTION
                    if mv_action not in self._remaining_valid_actions:
                        self._remaining_valid_actions.append(mv_action)
    
    def _disable_conflicting_extensions(self, action_id: int, type_specific_id: int):
        """
        禁用与当前索引冲突的扩展
        
        Args:
            action_id: 动作ID
            type_specific_id: 索引候选中的ID
        """
        current_index = self.index_candidates[type_specific_id]
        if not hasattr(current_index, 'columns'):
            return
        
        current_cols = tuple(current_index.columns)
        current_len = len(current_cols)
        
        if current_len < 2:
            return
        
        prefix = current_cols[:-1]
        
        # 禁用具有相同前缀但不同扩展的索引
        for other_action in copy.copy(self._remaining_valid_actions):
            if other_action >= self.number_of_index_actions:
                continue
            
            other_index = self.index_candidates[other_action]
            if not hasattr(other_index, 'columns'):
                continue
            
            other_cols = tuple(other_index.columns)
            if len(other_cols) != current_len:
                continue
            
            if other_cols[:-1] == prefix and other_cols != current_cols:
                self.valid_actions[other_action] = self.FORBIDDEN_ACTION
                if other_action in self._remaining_valid_actions:
                    self._remaining_valid_actions.remove(other_action)
    
    def _handle_reenable_indexes(self, type_specific_id: int):
        """
        处理重新启用索引的逻辑
        
        Args:
            type_specific_id: 索引候选中的ID
        """
        current_index = self.index_candidates[type_specific_id]
        if not hasattr(current_index, 'columns') or len(current_index.columns) <= 1:
            return
        
        # 重新启用前缀索引
        prefix_cols = tuple(current_index.columns[:-1])
        
        for i, index in enumerate(self.index_candidates):
            if hasattr(index, 'columns') and tuple(index.columns) == prefix_cols:
                # 检查前缀的前缀是否存在（如果需要）
                if len(prefix_cols) > 1:
                    parent_prefix = prefix_cols[:-1]
                    if parent_prefix not in self.current_combinations:
                        return
                
                self.valid_actions[i] = self.ALLOWED_ACTION
                if i not in self._remaining_valid_actions:
                    self._remaining_valid_actions.append(i)
                
                logging.debug(f"Re-enabled index: {index}")
                break
    
    def _valid_actions_based_on_workload(self, workload):
        """
        基于工作负载设置有效动作
        
        Args:
            workload: 工作负载
        """
        # 获取工作负载中可索引的列
        if hasattr(workload, 'indexable_columns'):
            indexable_columns = workload.indexable_columns(return_sorted=False)
            self.wl_indexable_columns = indexable_columns & self.indexable_columns
        else:
            self.wl_indexable_columns = self.indexable_columns
        
        # 启用单列索引
        for i, index in enumerate(self.index_candidates):
            if hasattr(index, 'columns') and len(index.columns) == 1:
                col = index.columns[0]
                if col in self.wl_indexable_columns:
                    self.valid_actions[i] = self.ALLOWED_ACTION
                    self._remaining_valid_actions.append(i)
        
        # 启用所有物化视图（物化视图通常不依赖于特定列）
        for i in range(self.number_of_mv_actions):
            mv_action_id = self.number_of_index_actions + i
            self.valid_actions[mv_action_id] = self.ALLOWED_ACTION
            self._remaining_valid_actions.append(mv_action_id)
    
    def _valid_actions_based_on_budget(self, budget: Optional[float], 
                                       current_storage_consumption: float):
        """
        基于预算设置有效动作
        
        Args:
            budget: 存储预算（MB）
            current_storage_consumption: 当前存储消耗（字节）
        """
        if budget is None:
            return
        
        new_remaining_actions = []
        for action_idx in self._remaining_valid_actions:
            action_cost = self.action_storage_consumptions[action_idx]
            total_cost = b_to_mb(current_storage_consumption + action_cost)
            
            if total_cost > budget:
                self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
            else:
                new_remaining_actions.append(action_idx)
        
        self._remaining_valid_actions = new_remaining_actions
    
    def _valid_actions_based_on_last_action(self, last_action: int):
        """基于上一个动作更新有效动作（由update_valid_actions调用）"""
        self._update_dependent_actions(last_action)
    
    def get_current_configuration(self) -> Tuple[Set[Index], Set[MaterializedView]]:
        """
        获取当前配置
        
        Returns:
            (当前索引集合, 当前物化视图集合)
        """
        return self.current_indexes.copy(), self.current_mvs.copy()
    
    def get_action_info(self, action_id: int) -> Dict[str, Any]:
        """
        获取动作信息
        
        Args:
            action_id: 动作ID
            
        Returns:
            动作信息字典
        """
        action_type, type_specific_id = self.get_action_type(action_id)
        
        info = {
            'action_id': action_id,
            'action_type': action_type,
            'type_specific_id': type_specific_id,
            'storage_cost': self.action_storage_consumptions[action_id],
            'is_valid': self.valid_actions[action_id] == self.ALLOWED_ACTION
        }
        
        if action_type == "INDEX":
            index = self.index_candidates[type_specific_id]
            info['object'] = index
            if hasattr(index, 'columns'):
                info['columns'] = index.columns
        else:
            mv = self.mv_candidates[type_specific_id]
            info['object'] = mv
            info['name'] = mv.name
            info['definition'] = mv.definition_sql
        
        return info
    
    def reset(self):
        """重置动作管理器状态"""
        self.current_indexes = set()
        self.current_mvs = set()
        self.current_combinations = set()
        self.valid_actions = None
        self._remaining_valid_actions = None
        self.current_action_status = None


class HybridActionManagerNonMasking(HybridActionManager):
    """
    非掩码版本的混合动作管理器
    所有动作始终有效，由环境处理无效动作
    """
    
    def get_initial_valid_actions(self, workload, budget: Optional[float]) -> np.ndarray:
        """获取初始有效动作（所有动作都有效）"""
        self.current_action_status = [0 for _ in range(self.number_of_columns)]
        self.valid_actions = [self.ALLOWED_ACTION for _ in range(self.number_of_actions)]
        self._remaining_valid_actions = list(range(self.number_of_actions))
        
        self.current_indexes = set()
        self.current_mvs = set()
        self.current_combinations = set()
        
        return np.array(self.valid_actions)
    
    def update_valid_actions(self, last_action: int, budget: Optional[float],
                            current_storage_consumption: float) -> Tuple[np.ndarray, bool]:
        """更新有效动作（保持所有动作有效）"""
        action_type, type_specific_id = self.get_action_type(last_action)
        
        if action_type == "INDEX":
            self._update_after_index_action(last_action, type_specific_id)
        else:
            self._update_after_mv_action(last_action, type_specific_id)
        
        return np.array(self.valid_actions), True
    
    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        """非掩码版本不基于预算禁用动作"""
        pass


def create_hybrid_action_manager(index_candidates: List,
                                 mv_candidates: List[MaterializedView],
                                 index_storage_costs: List[int],
                                 mv_storage_costs: List[int],
                                 config: Dict[str, Any]) -> HybridActionManager:
    """
    创建混合动作管理器的工厂函数
    
    Args:
        index_candidates: 索引候选列表
        mv_candidates: 物化视图候选列表
        index_storage_costs: 索引存储成本列表
        mv_storage_costs: 物化视图存储成本列表
        config: 配置字典
        
    Returns:
        HybridActionManager实例
    """
    use_masking = config.get('use_action_masking', True)
    
    manager_class = HybridActionManager if use_masking else HybridActionManagerNonMasking
    
    return manager_class(
        index_candidates=index_candidates,
        mv_candidates=mv_candidates,
        index_storage_costs=index_storage_costs,
        mv_storage_costs=mv_storage_costs,
        sb_version=config.get('sb_version', 2),
        max_index_width=config.get('max_index_width', 3),
        reenable_indexes=config.get('reenable_indexes', False)
    )

