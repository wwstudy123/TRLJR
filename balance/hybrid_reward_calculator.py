# balance/hybrid_reward_calculator.py
"""
混合奖励计算器模块
考虑索引和物化视图的联合效果和协同效应
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple

import numpy as np

from index_selection_evaluation.selection.utils import b_to_mb
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.materialized_view import MaterializedView

from .reward_calculator import RewardCalculator


class HybridRewardCalculator(RewardCalculator):
    """
    混合奖励计算器
    考虑索引和物化视图的联合效果和协同效应
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化混合奖励计算器
        
        Args:
            config: 配置字典，包含:
                - index_reward_weight: 索引奖励权重
                - mv_reward_weight: 物化视图奖励权重
                - storage_penalty: 存储惩罚系数
                - synergy_weight: 协同效应权重
                - performance_weight: 性能提升权重
        """
        super().__init__()
        
        self.config = config or {}
        
        # 奖励权重
        self.index_weight = self.config.get("index_reward_weight", 0.5)
        self.mv_weight = self.config.get("mv_reward_weight", 0.5)
        
        # 惩罚和奖励系数
        self.storage_penalty = self.config.get("storage_penalty", 0.01)
        self.synergy_weight = self.config.get("synergy_weight", 0.1)
        self.performance_weight = self.config.get("performance_weight", 1.0)
        
        # 奖励缩放
        self.reward_scale = self.config.get("reward_scale", 100.0)
        
        # 历史记录（用于计算协同效应）
        self.cost_history = []
        self.action_history = []
        
        logging.info(f"HybridRewardCalculator initialized with index_weight={self.index_weight}, "
                    f"mv_weight={self.mv_weight}")
    
    def reset(self):
        """重置奖励计算器"""
        super().reset()
        self.cost_history = []
        self.action_history = []
    
    def calculate_reward(self, environment_state: Dict[str, Any]) -> float:
        """
        计算奖励
        
        Args:
            environment_state: 环境状态字典
            
        Returns:
            奖励值
        """
        # 提取状态信息
        current_cost = environment_state["current_cost"]
        previous_cost = environment_state.get("previous_cost", current_cost)
        initial_cost = environment_state["initial_cost"]
        
        # 获取存储信息
        current_storage = environment_state.get("current_storage_consumption", 0)
        previous_storage = environment_state.get("previous_storage_consumption", 0)
        
        # 获取新增大小
        new_index_size = environment_state.get("new_index_size")
        new_mv_size = environment_state.get("new_mv_size")
        
        # 判断动作类型
        if new_mv_size is not None:
            action_type = "MATERIALIZED_VIEW"
            new_size = new_mv_size
        elif new_index_size is not None:
            action_type = "INDEX"
            new_size = new_index_size
        else:
            action_type = "UNKNOWN"
            new_size = 1
        
        # 计算协同效应
        synergy_bonus = self._calculate_synergy_bonus(environment_state)
        
        # 计算奖励
        reward = self._calculate_hybrid_reward(
            old_cost=previous_cost,
            new_cost=current_cost,
            initial_cost=initial_cost,
            old_storage=previous_storage,
            new_storage=current_storage,
            new_size=new_size,
            action_type=action_type,
            synergy_bonus=synergy_bonus
        )
        
        # 更新历史
        self.cost_history.append(current_cost)
        self.action_history.append(action_type)
        
        # 累积奖励
        self.accumulated_reward += reward
        
        return reward
    
    def _calculate_hybrid_reward(self,
                                old_cost: float,
                                new_cost: float,
                                initial_cost: float,
                                old_storage: float,
                                new_storage: float,
                                new_size: float,
                                action_type: str,
                                synergy_bonus: float) -> float:
        """
        计算联合奖励，考虑索引和物化视图的协同效应
        
        Args:
            old_cost: 旧成本
            new_cost: 新成本
            initial_cost: 初始成本
            old_storage: 旧存储
            new_storage: 新存储
            new_size: 新增大小
            action_type: 动作类型
            synergy_bonus: 协同效应奖励
            
        Returns:
            总奖励
        """
        # 1. 基础性能提升奖励
        if old_cost > 0:
            performance_improvement = (old_cost - new_cost) / old_cost
        else:
            performance_improvement = 0
        
        # 相对于初始成本的改善
        if initial_cost > 0:
            relative_improvement = (old_cost - new_cost) / initial_cost
        else:
            relative_improvement = 0
        
        # 2. 存储成本惩罚
        storage_increase = new_storage - old_storage
        storage_penalty = b_to_mb(storage_increase) * self.storage_penalty
        
        # 3. 存储效率奖励（性能提升/存储增加）
        if new_size > 0:
            storage_efficiency = relative_improvement / b_to_mb(new_size)
        else:
            storage_efficiency = relative_improvement
        
        # 4. 协同效应奖励
        synergy_reward = synergy_bonus * self.synergy_weight
        
        # 5. 根据动作类型调整权重
        if action_type == "INDEX":
            action_weight = self.index_weight
        elif action_type == "MATERIALIZED_VIEW":
            action_weight = self.mv_weight
        else:
            action_weight = 0.5
        
        # 6. 组合奖励
        performance_reward = self.performance_weight * relative_improvement
        
        total_reward = action_weight * (
            performance_reward + 
            storage_efficiency - 
            storage_penalty
        ) + synergy_reward
        
        # 7. 缩放奖励
        total_reward *= self.reward_scale
        
        return total_reward
    
    def _calculate_synergy_bonus(self, environment_state: Dict[str, Any]) -> float:
        """
        计算协同效应奖励
        
        Args:
            environment_state: 环境状态
            
        Returns:
            协同效应奖励
        """
        synergy = 0.0
        
        # 获取当前索引和物化视图
        current_indexes = environment_state.get("current_indexes", set())
        current_mvs = environment_state.get("current_mvs", set())
        
        # 如果同时有索引和物化视图，计算协同效应
        if current_indexes and current_mvs:
            # 1. 索引-MV覆盖协同
            coverage_synergy = self._calculate_coverage_synergy(current_indexes, current_mvs)
            synergy += coverage_synergy
            
            # 2. 成本下降加速协同
            if len(self.cost_history) >= 2:
                acceleration_synergy = self._calculate_acceleration_synergy()
                synergy += acceleration_synergy
            
            # 3. 多样性协同
            diversity_synergy = self._calculate_diversity_synergy(current_indexes, current_mvs)
            synergy += diversity_synergy
        
        return synergy
    
    def _calculate_coverage_synergy(self, 
                                   indexes: Set[Index], 
                                   mvs: Set[MaterializedView]) -> float:
        """
        计算索引和MV的覆盖协同
        
        当索引和物化视图覆盖不同的查询模式时，给予额外奖励
        """
        synergy = 0.0
        
        # 简单的覆盖协同：索引和MV数量的乘积效应
        num_indexes = len(indexes)
        num_mvs = len(mvs)
        
        if num_indexes > 0 and num_mvs > 0:
            # 归一化的协同分数
            synergy = min(num_indexes, num_mvs) / max(num_indexes + num_mvs, 1) * 0.1
        
        return synergy
    
    def _calculate_acceleration_synergy(self) -> float:
        """
        计算成本下降加速协同
        
        如果最近的成本下降速度加快，给予额外奖励
        """
        if len(self.cost_history) < 3:
            return 0.0
        
        # 计算最近两次的成本下降
        recent_decrease = self.cost_history[-2] - self.cost_history[-1]
        previous_decrease = self.cost_history[-3] - self.cost_history[-2]
        
        # 如果下降加速，给予奖励
        if recent_decrease > previous_decrease and previous_decrease > 0:
            acceleration = (recent_decrease - previous_decrease) / previous_decrease
            return min(acceleration * 0.05, 0.1)  # 限制最大值
        
        return 0.0
    
    def _calculate_diversity_synergy(self,
                                    indexes: Set[Index],
                                    mvs: Set[MaterializedView]) -> float:
        """
        计算配置多样性协同
        
        多样化的配置（不同类型的索引和MV）获得额外奖励
        """
        synergy = 0.0
        
        # 索引多样性：不同宽度的索引
        index_widths = set()
        for index in indexes:
            if hasattr(index, 'columns'):
                index_widths.add(len(index.columns))
        
        if len(index_widths) > 1:
            synergy += 0.02 * (len(index_widths) - 1)
        
        # MV多样性：不同类型的MV（基于定义SQL的特征）
        mv_types = set()
        for mv in mvs:
            if hasattr(mv, 'definition_sql'):
                sql = mv.definition_sql.upper()
                if 'GROUP BY' in sql:
                    mv_types.add('aggregation')
                if 'JOIN' in sql:
                    mv_types.add('join')
                if 'WHERE' in sql and 'GROUP BY' not in sql:
                    mv_types.add('filter')
        
        if len(mv_types) > 1:
            synergy += 0.02 * (len(mv_types) - 1)
        
        return synergy
    
    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        """兼容父类接口"""
        if initial_cost > 0:
            reward = ((previous_cost - current_cost) / initial_cost) * self.reward_scale
        else:
            reward = 0
        return reward


class HybridRelativeDifferenceReward(HybridRewardCalculator):
    """
    基于相对差异的混合奖励计算器
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def _calculate_hybrid_reward(self, old_cost, new_cost, initial_cost,
                                old_storage, new_storage, new_size,
                                action_type, synergy_bonus) -> float:
        """相对差异奖励"""
        if initial_cost > 0:
            relative_improvement = (old_cost - new_cost) / initial_cost
        else:
            relative_improvement = 0
        
        # 存储惩罚
        storage_penalty = b_to_mb(new_storage - old_storage) * self.storage_penalty
        
        # 协同奖励
        synergy_reward = synergy_bonus * self.synergy_weight
        
        # 动作权重
        weight = self.index_weight if action_type == "INDEX" else self.mv_weight
        
        reward = weight * (relative_improvement - storage_penalty) + synergy_reward
        return reward * self.reward_scale


class HybridStorageEfficiencyReward(HybridRewardCalculator):
    """
    基于存储效率的混合奖励计算器
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def _calculate_hybrid_reward(self, old_cost, new_cost, initial_cost,
                                old_storage, new_storage, new_size,
                                action_type, synergy_bonus) -> float:
        """存储效率奖励"""
        if initial_cost > 0:
            relative_improvement = (old_cost - new_cost) / initial_cost
        else:
            relative_improvement = 0
        
        # 存储效率
        storage_mb = b_to_mb(new_size) if new_size > 0 else 1
        efficiency = relative_improvement / storage_mb
        
        # 协同奖励
        synergy_reward = synergy_bonus * self.synergy_weight
        
        # 动作权重
        weight = self.index_weight if action_type == "INDEX" else self.mv_weight
        
        reward = weight * efficiency + synergy_reward
        return reward * self.reward_scale


class HybridDRLindaReward(HybridRewardCalculator):
    """
    基于DRLinda的混合奖励计算器
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def _calculate_hybrid_reward(self, old_cost, new_cost, initial_cost,
                                old_storage, new_storage, new_size,
                                action_type, synergy_bonus) -> float:
        """DRLinda风格奖励"""
        if initial_cost > 0:
            total_improvement = (initial_cost - new_cost) / initial_cost * 100
        else:
            total_improvement = 0
        
        # 协同奖励
        synergy_reward = synergy_bonus * self.synergy_weight * 10
        
        return total_improvement + synergy_reward


class HybridCompositeReward(HybridRewardCalculator):
    """
    复合混合奖励计算器
    结合多种奖励策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 各组件权重
        self.performance_component_weight = config.get("performance_component_weight", 0.4)
        self.efficiency_component_weight = config.get("efficiency_component_weight", 0.3)
        self.synergy_component_weight = config.get("synergy_component_weight", 0.2)
        self.progress_component_weight = config.get("progress_component_weight", 0.1)
    
    def _calculate_hybrid_reward(self, old_cost, new_cost, initial_cost,
                                old_storage, new_storage, new_size,
                                action_type, synergy_bonus) -> float:
        """复合奖励"""
        # 1. 性能组件
        if initial_cost > 0:
            performance = (old_cost - new_cost) / initial_cost
        else:
            performance = 0
        
        # 2. 效率组件
        storage_mb = b_to_mb(new_size) if new_size > 0 else 1
        efficiency = performance / storage_mb if storage_mb > 0 else performance
        
        # 3. 协同组件
        synergy = synergy_bonus
        
        # 4. 进度组件（相对于初始成本的总体改善）
        if initial_cost > 0:
            progress = (initial_cost - new_cost) / initial_cost
        else:
            progress = 0
        
        # 动作权重
        weight = self.index_weight if action_type == "INDEX" else self.mv_weight
        
        # 组合奖励
        reward = weight * (
            self.performance_component_weight * performance +
            self.efficiency_component_weight * efficiency +
            self.synergy_component_weight * synergy +
            self.progress_component_weight * progress
        )
        
        return reward * self.reward_scale


class AdaptiveHybridRewardCalculator(HybridRewardCalculator):
    """
    自适应混合奖励计算器
    根据训练进度动态调整奖励策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.episode_count = 0
        self.warmup_episodes = config.get("warmup_episodes", 100)
        self.exploration_bonus_decay = config.get("exploration_bonus_decay", 0.99)
        self.current_exploration_bonus = config.get("initial_exploration_bonus", 0.1)
    
    def reset(self):
        """重置并更新episode计数"""
        super().reset()
        self.episode_count += 1
        
        # 衰减探索奖励
        if self.episode_count > self.warmup_episodes:
            self.current_exploration_bonus *= self.exploration_bonus_decay
    
    def _calculate_hybrid_reward(self, old_cost, new_cost, initial_cost,
                                old_storage, new_storage, new_size,
                                action_type, synergy_bonus) -> float:
        """自适应奖励"""
        # 基础奖励
        base_reward = super()._calculate_hybrid_reward(
            old_cost, new_cost, initial_cost,
            old_storage, new_storage, new_size,
            action_type, synergy_bonus
        )
        
        # 探索奖励（鼓励尝试不同动作）
        exploration_reward = 0
        if action_type == "MATERIALIZED_VIEW" and self.episode_count < self.warmup_episodes:
            # 早期阶段鼓励尝试物化视图
            exploration_reward = self.current_exploration_bonus
        
        # 新颖性奖励
        novelty_reward = self._calculate_novelty_reward(action_type)
        
        return base_reward + exploration_reward + novelty_reward
    
    def _calculate_novelty_reward(self, action_type: str) -> float:
        """计算新颖性奖励"""
        if len(self.action_history) < 2:
            return 0.0
        
        # 如果连续选择不同类型的动作，给予奖励
        if self.action_history[-1] != action_type:
            return self.current_exploration_bonus * 0.5
        
        return 0.0


def create_hybrid_reward_calculator(calculator_type: str = "basic",
                                   config: Dict[str, Any] = None) -> HybridRewardCalculator:
    """
    创建混合奖励计算器的工厂函数
    
    Args:
        calculator_type: 计算器类型
        config: 配置字典
        
    Returns:
        HybridRewardCalculator实例
    """
    calculators = {
        "basic": HybridRewardCalculator,
        "relative": HybridRelativeDifferenceReward,
        "efficiency": HybridStorageEfficiencyReward,
        "drlinda": HybridDRLindaReward,
        "composite": HybridCompositeReward,
        "adaptive": AdaptiveHybridRewardCalculator
    }
    
    calculator_class = calculators.get(calculator_type, HybridRewardCalculator)
    return calculator_class(config)


# 默认配置
DEFAULT_HYBRID_REWARD_CONFIG = {
    "index_reward_weight": 0.5,
    "mv_reward_weight": 0.5,
    "storage_penalty": 0.01,
    "synergy_weight": 0.1,
    "performance_weight": 1.0,
    "reward_scale": 100.0,
    "performance_component_weight": 0.4,
    "efficiency_component_weight": 0.3,
    "synergy_component_weight": 0.2,
    "progress_component_weight": 0.1,
    "warmup_episodes": 100,
    "exploration_bonus_decay": 0.99,
    "initial_exploration_bonus": 0.1
}

