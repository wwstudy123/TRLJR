# TPC-H数据集索引和物化视图联合推荐测试指南

## 概述

本文档说明如何在TPC-H数据集上测试索引和物化视图联合推荐功能。所有hybrid模块已集成到实验类、环境类和主程序中。

## 已完成的集成

### 1. 实验类扩展 (`BALANCE/experiment.py`)

**添加的功能：**
- ✅ 导入hybrid模块（HybridActionManager, HybridObservationManager, HybridRewardCalculator）
- ✅ 初始化JointExperimentConfig配置
- ✅ 在prepare()阶段更新索引和MV候选
- ✅ 创建联合环境（JointDBEnvV1）
- ✅ 详细的日志输出和验证

**关键代码位置：**
- `__init__()`: 初始化联合配置
- `prepare()`: 更新联合配置的索引候选和MV候选
- `_create_joint_env()`: 创建联合环境

### 2. 环境类扩展 (`gym_db/envs/`)

**HybridDBEnv (`hybrid_db_env.py`):**
- ✅ 支持索引和物化视图的联合操作
- ✅ 管理MV的创建和删除
- ✅ 计算联合成本
- ✅ 详细的性能报告

**JointDBEnvV1 (`joint_db_env_v1.py`):**
- ✅ 封装HybridDBEnv
- ✅ 验证hybrid模块类型
- ✅ 专门用于TPC-H测试

### 3. 主程序扩展 (`main.py`)

**添加的功能：**
- ✅ 检测联合推荐启用状态
- ✅ 验证配置完整性
- ✅ 详细的测试信息输出
- ✅ TPC-H数据集测试标识

## 快速开始

### 步骤1: 运行测试脚本

首先运行测试脚本验证所有模块是否正确集成：

```bash
python test_joint_recommendation.py
```

测试脚本会检查：
- Hybrid模块导入
- 配置文件加载
- 实验初始化

### 步骤2: 检查配置文件

确保 `experiments/tpch.json` 中已启用联合推荐：

```json
{
  "materialized_views": {
    "enabled": true,
    ...
  }
}
```

### 步骤3: 运行主程序

```bash
python main.py
```

## 运行时的输出信息

### 启动阶段

程序启动时会显示：

```
======================================================================
Joint Index and Materialized View Recommendation ENABLED
Testing on TPC-H Dataset
  - MV Candidates: 3
  - Index Candidates: Not initialized
======================================================================
```

### 准备阶段

在 `experiment.prepare()` 后会显示：

```
======================================================================
Joint Configuration Updated for TPC-H Dataset
  - Index Candidates: 150
  - MV Candidates: 3
  - Total Actions: 153
    * Index Actions: 150
    * MV Actions: 3
  - Observation Features: 2150
======================================================================
Materialized View Candidates:
  [0] mv_customer_orders: SELECT c_custkey, c_name, COUNT(o_orderkey) as order_count...
  [1] mv_supplier_parts: SELECT s_suppkey, s_name, COUNT(ps_partkey) as part_count...
  [2] mv_lineitem_summary: SELECT l_orderkey, l_partkey, SUM(l_quantity) as total_quantity...
```

### 环境创建阶段

创建环境时会显示：

```
======================================================================
HybridDBEnv Initialized
  - MV Candidates: 3
  - Index Candidates: 150
  - Action Manager Type: HybridActionManager
  - Observation Manager Type: HybridObservationManager
  - Reward Calculator Type: HybridRewardCalculator
======================================================================
```

### Episode结束阶段

每个episode结束时会显示详细的性能报告：

```
======================================================================
Evaluated Workload (TESTING): ...
  Initial cost: 1,234,567.89
  Current cost: 987,654.32
  Cost reduction: 20.00%
  Accumulated reward: 15.2345
  Storage consumption: 1250.50 MB
    - Index storage: 1000.00 MB
    - MV storage: 250.50 MB
  Configuration:
    - Indexes: 12
    - Materialized Views: 2
  Selected MVs:
    * mv_customer_orders
    * mv_lineitem_summary
======================================================================
```

## 配置说明

### 物化视图配置

在 `experiments/tpch.json` 中配置：

```json
{
  "materialized_views": {
    "enabled": true,                    // 必须为true
    "max_count": 5,                     // 最大MV数量
    "auto_mine": false,                 // 是否自动挖掘
    "observation_manager_type": "basic", // 观察管理器类型
    "reward_calculator_type": "basic",  // 奖励计算器类型
    "templates": [                      // MV模板
      {
        "name": "mv_customer_orders",
        "sql": "SELECT ...",
        "estimated_size": 1048576
      }
    ]
  }
}
```

### 观察管理器类型

- `basic`: 基础观察（默认）
- `plan`: 基于查询计划嵌入
- `workload`: 基于工作负载嵌入
- `cost`: 包含成本信息
- `mv_benefit`: 包含MV收益信息
- `full`: 完整特征集

### 奖励计算器类型

- `basic`: 基础混合奖励（默认）
- `relative`: 基于相对差异
- `efficiency`: 基于存储效率
- `drlinda`: DRLinda风格
- `composite`: 复合奖励
- `adaptive`: 自适应奖励

## 验证检查清单

运行前请确认：

- [ ] Hybrid模块文件存在：
  - `BALANCE/hybrid_action_manager.py`
  - `BALANCE/hybrid_observation_manager.py`
  - `BALANCE/hybrid_reward_calculator.py`
  - `BALANCE/materialized_view_miner.py`
  - `BALANCE/joint_experiment_config.py`

- [ ] 环境文件存在：
  - `gym_db/envs/hybrid_db_env.py`
  - `gym_db/envs/joint_db_env_v1.py`

- [ ] 配置文件正确：
  - `experiments/tpch.json` 中 `materialized_views.enabled = true`

- [ ] 数据库连接正常：
  - PostgreSQL数据库 `tpch_1_lhr` 存在
  - 数据库支持物化视图功能

## 故障排除

### 问题1: Hybrid模块导入失败

**错误信息：**
```
Hybrid modules not fully available: ...
```

**解决方案：**
- 检查hybrid模块文件是否存在
- 检查Python路径配置
- 运行 `python test_joint_recommendation.py` 诊断

### 问题2: 联合推荐未启用

**错误信息：**
```
Joint recommendation is DISABLED
```

**解决方案：**
- 检查 `experiments/tpch.json` 中 `materialized_views.enabled` 是否为 `true`
- 检查配置文件格式是否正确

### 问题3: 环境创建失败

**错误信息：**
```
Missing required config keys: ...
```

**解决方案：**
- 确保 `experiment.prepare()` 已调用
- 检查 `joint_config` 是否正确初始化
- 查看日志中的详细错误信息

### 问题4: MV候选为空

**错误信息：**
```
MV Candidates: 0
```

**解决方案：**
- 检查 `templates` 配置是否正确
- 如果使用自动挖掘，检查 `min_support` 设置
- 查看日志中的MV初始化信息

## 性能监控

### 关键指标

程序运行时会输出以下关键指标：

1. **成本改善率**: `Cost reduction: XX%`
2. **存储消耗**: `Storage consumption: XX MB`
3. **索引数量**: `Indexes: XX`
4. **MV数量**: `Materialized Views: XX`
5. **累积奖励**: `Accumulated reward: XX`

### 日志文件

所有日志会保存到：
- `experiment_results/ID_<experiment_id>/report_ID_<experiment_id>.txt`

## 下一步

1. **调整配置**: 根据TPC-H数据集特点调整MV模板和参数
2. **监控训练**: 观察训练过程中的成本改善和存储使用
3. **分析结果**: 查看报告文件分析联合推荐效果
4. **优化参数**: 根据结果调整奖励权重、存储惩罚等参数

## 参考文档

- `JOINT_RECOMMENDATION_GUIDE.md`: 详细的联合推荐使用指南
- `test_joint_recommendation.py`: 测试脚本
- `experiments/tpch.json`: TPC-H配置文件示例

## 联系支持

如遇到问题，请检查：
1. 日志文件中的详细错误信息
2. 测试脚本的输出
3. 配置文件格式
