# 混合索引和物化视图推荐模式使用指南

## 概述

本指南说明如何启用和使用索引和物化视图的联合推荐模式。该模式通过强化学习同时优化索引和物化视图的选择，以获得更好的查询性能。

## 修改内容

### 1. 新增文件

#### 1.1 `index_selection_evaluation/selection/materialized_view.py`
- 定义了 `MaterializedView` 类
- 包含物化视图的基本属性：名称、定义SQL、预估大小
- 实现了相等性和哈希方法，便于在集合中操作

#### 1.2 `balance/materialized_view_miner.py`（已存在）
- `MaterializedViewMiner`：使用频繁模式挖掘生成物化视图候选
- `JointCandidateGenerator`：联合生成索引和物化视图候选

#### 1.3 `balance/hybrid_action_manager.py`（已存在）
- `HybridActionManager`：管理索引和物化视图的联合动作空间

#### 1.4 `gym_db/envs/hybrid_db_env.py`（已存在）
- `HybridDBEnv`：支持索引和物化视图的混合环境

### 2. 修改的文件

#### 2.1 `gym_db/__init__.py`
- 新增环境注册：`DB-hybrid-v1` → `HybridDBEnv`

#### 2.2 `balance/experiment.py`
- `prepare()` 方法：添加物化视图候选生成逻辑
- `_generate_materialized_view_candidates()` 方法：新方法，用于生成物化视图候选
- `make_env()` 方法：支持混合模式和标准模式的自动切换

### 3. 配置文件

#### 3.1 `experiments/tpch_hybrid.json`
- 新增混合模式配置示例
- 关键配置项：
  - `enable_hybrid_mode: true`：启用混合模式
  - `materialized_view_config`：物化视图配置
    - `min_support`：最小支持度阈值（0-1）
    - `max_mv_count`：最大物化视图候选数量

## 使用方法

### 方法1：直接运行混合模式实验

```bash
python -m balance experiments/tpch_hybrid.json
```

### 方法2：修改现有配置文件

在现有配置文件中添加以下配置：

```json
{
  "enable_hybrid_mode": true,
  "materialized_view_config": {
    "min_support": 0.3,
    "max_mv_count": 10
  }
}
```

### 方法3：修改main.py参数

```bash
python main.py experiments/tpch_hybrid.json
```

或修改 `main.py` 中的配置文件路径：

```python
CONFIGURATION_FILE = "experiments/tpch_hybrid.json"
```

## 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_hybrid_mode` | bool | false | 是否启用混合模式 |
| `min_support` | float | 0.3 | 频繁模式挖掘的最小支持度 |
| `max_mv_count` | int | 10 | 最大物化视图候选数量 |
| `max_index_width` | int | 3 | 最大索引宽度 |

## 架构说明

### 混合模式的动作空间

混合模式的动作空间由两部分组成：
1. **索引动作**：0 到 N_idx-1
2. **物化视图动作**：N_idx 到 N_idx+N_mv-1

总动作数 = 索引候选数 + 物化视图候选数

### 混合模式的观察空间

观察空间包含：
- 索引状态向量
- 物化视图状态向量
- 查询成本
- 存储消耗（索引+物化视图）

### 混合模式的奖励计算

奖励考虑：
- 查询成本降低
- 索引存储消耗
- 物化视图存储消耗
- 总存储预算约束

## 工作流程

1. **准备阶段**：
   - 生成索引候选
   - 使用频繁模式挖掘生成物化视图候选

2. **训练阶段**：
   - 强化学习智能体在混合环境中学习
   - 智能体可以选择创建索引或物化视图

3. **评估阶段**：
   - 在验证和测试工作负载上评估性能
   - 输出索引和物化视图的联合配置

## 示例输出

```
INFO: Generated 150 index candidates into the environments.
INFO: Generated 10 materialized view candidates
INFO: HybridDBEnv initialized with 10 MV candidates
...
Evaluated Workload (TEST): Test_Workload_1
    Initial cost: 12,345.67, now: 8,765.43 (70.95%). 
    Storage: 12.50 MB (Index: 8.75 MB, MV: 3.75 MB)
    Configuration: 5 indexes, 2 MVs
```

## 注意事项

1. **数据库支持**：确保PostgreSQL支持物化视图（版本 >= 9.3）
2. **存储预算**：混合模式需要更大的存储预算，因为要同时考虑索引和物化视图
3. **计算成本**：物化视图候选生成会增加准备阶段的计算时间
4. **查询重写**：当前版本假设数据库查询优化器能自动使用物化视图

## 故障排除

### 问题1：导入错误
```
ModuleNotFoundError: No module named 'balance.materialized_view_miner'
```
**解决**：确保 `balance` 模块在Python路径中

### 问题2：环境注册错误
```
gym.error.UnregisteredEnv: DB-hybrid-v1 not found
```
**解决**：确保运行时导入了 `gym_db` 模块

### 问题3：物化视图创建失败
```
Failed to create MV mv_join_0: syntax error
```
**解决**：检查物化视图SQL语法，确保表名和列名正确

## 扩展功能

### 自定义物化视图挖掘策略

可以继承 `MaterializedViewMiner` 类实现自定义策略：

```python
from balance.materialized_view_miner import MaterializedViewMiner

class CustomMVMINER(MaterializedViewMiner):
    def mine_frequent_view_candidates(self):
        # 自定义挖掘逻辑
        pass
```

### 自定义奖励函数

使用混合奖励计算器：

```json
{
  "enable_hybrid_mode": true,
  "hybrid_reward_calculator": "HybridRewardCalculator"
}
```

## 性能优化建议

1. **减少候选数量**：降低 `max_mv_count` 和 `max_index_width`
2. **提高支持度阈值**：增大 `min_support` 值
3. **过滤工作负载**：在配置中过滤不必要的查询类
4. **并行环境**：增加 `parallel_environments` 数量

## 参考文献和致谢

本实现基于以下研究：
- Bruno 和 Chaudhuri (2005): "Automatic Physical Database Tuning: A Relaxation-based Approach"
- Sadri et al.: "DRLindex: Deep Reinforcement Learning Index Advisor for a Cluster Database"
