# balance/mlp_cost_estimator.py
"""
MLP联合成本估算模型
用于估算索引和物化视图联合配置的查询成本
"""

import logging
import os
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CostEstimationDataset(Dataset):
    """成本估算数据集"""
    
    def __init__(self, 
                 query_features: np.ndarray,
                 index_features: np.ndarray,
                 mv_features: np.ndarray,
                 true_costs: np.ndarray):
        """
        初始化数据集
        
        Args:
            query_features: 查询特征 [N, query_dim]
            index_features: 索引特征 [N, index_dim]
            mv_features: 物化视图特征 [N, mv_dim]
            true_costs: 真实成本 [N, 1]
        """
        self.query_features = torch.FloatTensor(query_features)
        self.index_features = torch.FloatTensor(index_features)
        self.mv_features = torch.FloatTensor(mv_features)
        self.true_costs = torch.FloatTensor(true_costs).reshape(-1, 1)
        
    def __len__(self):
        return len(self.true_costs)
    
    def __getitem__(self, idx):
        return (
            self.query_features[idx],
            self.index_features[idx],
            self.mv_features[idx],
            self.true_costs[idx]
        )


class MLPCostEstimator(nn.Module):
    """MLP成本估算模型"""
    
    def __init__(self, 
                 query_dim: int = 128,
                 index_dim: int = 64,
                 mv_dim: int = 32,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True):
        """
        初始化MLP成本估算器
        
        Args:
            query_dim: 查询特征维度
            index_dim: 索引特征维度
            mv_dim: 物化视图特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout率
            use_batch_norm: 是否使用批归一化
        """
        super().__init__()
        
        self.query_dim = query_dim
        self.index_dim = index_dim
        self.mv_dim = mv_dim
        self.input_dim = query_dim + index_dim + mv_dim
        
        # 特征编码器
        self.query_encoder = self._build_encoder(query_dim, query_dim // 2)
        self.index_encoder = self._build_encoder(index_dim, index_dim // 2)
        self.mv_encoder = self._build_encoder(mv_dim, mv_dim // 2)
        
        # 编码后的总维度
        encoded_dim = query_dim // 2 + index_dim // 2 + mv_dim // 2
        
        # 主网络
        layers = []
        prev_dim = encoded_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
        logging.info(f"MLPCostEstimator initialized with input_dim={self.input_dim}, "
                    f"hidden_dims={hidden_dims}")
    
    def _build_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建特征编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, 
                query_features: torch.Tensor,
                index_features: torch.Tensor,
                mv_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_features: 查询计划特征 [batch, query_dim]
            index_features: 候选索引特征 [batch, index_dim]
            mv_features: 候选物化视图特征 [batch, mv_dim]
            
        Returns:
            成本估算 [batch, 1]
        """
        # 编码各部分特征
        query_encoded = self.query_encoder(query_features)
        index_encoded = self.index_encoder(index_features)
        mv_encoded = self.mv_encoder(mv_features)
        
        # 拼接特征
        combined_features = torch.cat([query_encoded, index_encoded, mv_encoded], dim=-1)
        
        # 预测成本
        cost_estimate = self.network(combined_features)
        
        return cost_estimate
    
    def forward_combined(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        使用已拼接的特征进行前向传播
        
        Args:
            combined_features: 拼接后的特征 [batch, input_dim]
            
        Returns:
            成本估算 [batch, 1]
        """
        # 分割特征
        query_features = combined_features[:, :self.query_dim]
        index_features = combined_features[:, self.query_dim:self.query_dim + self.index_dim]
        mv_features = combined_features[:, self.query_dim + self.index_dim:]
        
        return self.forward(query_features, index_features, mv_features)


class MLPCostEstimatorTrainer:
    """MLP成本估算器训练器"""
    
    def __init__(self, 
                 model: MLPCostEstimator,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: str = None):
        """
        初始化训练器
        
        Args:
            model: MLP模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 计算设备
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_model(self, 
                   train_loader: DataLoader,
                   val_loader: DataLoader = None,
                   num_epochs: int = 100,
                   early_stopping_patience: int = 20,
                   save_path: str = None) -> Dict[str, List[float]]:
        """
        训练MLP成本估算模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_path: 模型保存路径
            
        Returns:
            训练历史
        """
        logging.info(f"Starting training for {num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch + 1}/{num_epochs}, "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch + 1}/{num_epochs}, "
                               f"Train Loss: {train_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            query_feat, index_feat, mv_feat, true_cost = batch
            
            # 移动到设备
            query_feat = query_feat.to(self.device)
            index_feat = index_feat.to(self.device)
            mv_feat = mv_feat.to(self.device)
            true_cost = true_cost.to(self.device)
            
            # 前向传播
            predicted_cost = self.model(query_feat, index_feat, mv_feat)
            loss = self.criterion(predicted_cost, true_cost)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                query_feat, index_feat, mv_feat, true_cost = batch
                
                query_feat = query_feat.to(self.device)
                index_feat = index_feat.to(self.device)
                mv_feat = mv_feat.to(self.device)
                true_cost = true_cost.to(self.device)
                
                predicted_cost = self.model(query_feat, index_feat, mv_feat)
                loss = self.criterion(predicted_cost, true_cost)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def predict(self, 
               query_features: np.ndarray,
               index_features: np.ndarray,
               mv_features: np.ndarray) -> np.ndarray:
        """
        预测成本
        
        Args:
            query_features: 查询特征
            index_features: 索引特征
            mv_features: 物化视图特征
            
        Returns:
            预测的成本
        """
        self.model.eval()
        
        with torch.no_grad():
            query_tensor = torch.FloatTensor(query_features).to(self.device)
            index_tensor = torch.FloatTensor(index_features).to(self.device)
            mv_tensor = torch.FloatTensor(mv_features).to(self.device)
            
            # 确保是2D张量
            if query_tensor.dim() == 1:
                query_tensor = query_tensor.unsqueeze(0)
                index_tensor = index_tensor.unsqueeze(0)
                mv_tensor = mv_tensor.unsqueeze(0)
            
            predicted = self.model(query_tensor, index_tensor, mv_tensor)
            
        return predicted.cpu().numpy()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'query_dim': self.model.query_dim,
                'index_dim': self.model.index_dim,
                'mv_dim': self.model.mv_dim
            }
        }, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logging.info(f"Model loaded from {path}")


class JointCostEstimator:
    """联合成本估算器：结合MLP预测和实际成本评估"""
    
    def __init__(self, 
                 mlp_model: MLPCostEstimator,
                 cost_evaluation = None,
                 use_mlp_weight: float = 0.3):
        """
        初始化联合成本估算器
        
        Args:
            mlp_model: MLP成本估算模型
            cost_evaluation: 实际成本评估器
            use_mlp_weight: MLP预测的权重
        """
        self.mlp_model = mlp_model
        self.cost_evaluation = cost_evaluation
        self.use_mlp_weight = use_mlp_weight
        self.device = next(mlp_model.parameters()).device
        
        # 成本缓存
        self.cost_cache = {}
        
    def estimate_cost(self,
                     workload,
                     indexes: List,
                     materialized_views: List,
                     query_features: np.ndarray,
                     index_features: np.ndarray,
                     mv_features: np.ndarray) -> float:
        """
        估算联合配置的成本
        
        Args:
            workload: 工作负载
            indexes: 索引列表
            materialized_views: 物化视图列表
            query_features: 查询特征
            index_features: 索引特征
            mv_features: 物化视图特征
            
        Returns:
            估算成本
        """
        # 检查缓存
        cache_key = self._get_cache_key(workload, indexes, materialized_views)
        if cache_key in self.cost_cache:
            return self.cost_cache[cache_key]
        
        # MLP预测
        mlp_cost = self._mlp_predict(query_features, index_features, mv_features)
        
        # 如果有实际成本评估器，结合两者
        if self.cost_evaluation is not None:
            try:
                actual_cost = self.cost_evaluation.calculate_cost(
                    workload, indexes, materialized_views
                )
                # 加权组合
                final_cost = (self.use_mlp_weight * mlp_cost + 
                            (1 - self.use_mlp_weight) * actual_cost)
            except Exception as e:
                logging.warning(f"Actual cost evaluation failed: {e}, using MLP prediction only")
                final_cost = mlp_cost
        else:
            final_cost = mlp_cost
        
        # 缓存结果
        self.cost_cache[cache_key] = final_cost
        
        return final_cost
    
    def _mlp_predict(self,
                    query_features: np.ndarray,
                    index_features: np.ndarray,
                    mv_features: np.ndarray) -> float:
        """MLP预测成本"""
        self.mlp_model.eval()
        
        with torch.no_grad():
            query_tensor = torch.FloatTensor(query_features).to(self.device)
            index_tensor = torch.FloatTensor(index_features).to(self.device)
            mv_tensor = torch.FloatTensor(mv_features).to(self.device)
            
            if query_tensor.dim() == 1:
                query_tensor = query_tensor.unsqueeze(0)
                index_tensor = index_tensor.unsqueeze(0)
                mv_tensor = mv_tensor.unsqueeze(0)
            
            predicted = self.mlp_model(query_tensor, index_tensor, mv_tensor)
        
        return predicted.item()
    
    def _get_cache_key(self, workload, indexes, materialized_views) -> str:
        """生成缓存键"""
        workload_key = str(id(workload))
        index_key = str(sorted([str(idx) for idx in indexes]))
        mv_key = str(sorted([mv.name for mv in materialized_views]))
        return f"{workload_key}_{index_key}_{mv_key}"
    
    def clear_cache(self):
        """清除缓存"""
        self.cost_cache.clear()
    
    def rank_configurations(self,
                           workload,
                           configurations: List[Tuple[List, List]],
                           query_features: np.ndarray,
                           index_features_list: List[np.ndarray],
                           mv_features_list: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        对配置进行排序
        
        Args:
            workload: 工作负载
            configurations: 配置列表 [(indexes, mvs), ...]
            query_features: 查询特征
            index_features_list: 各配置的索引特征列表
            mv_features_list: 各配置的物化视图特征列表
            
        Returns:
            排序后的(配置索引, 成本)列表
        """
        costs = []
        
        for i, (indexes, mvs) in enumerate(configurations):
            cost = self.estimate_cost(
                workload, indexes, mvs,
                query_features, index_features_list[i], mv_features_list[i]
            )
            costs.append((i, cost))
        
        # 按成本升序排序
        costs.sort(key=lambda x: x[1])
        
        return costs


class CostEstimatorDataCollector:
    """成本估算数据收集器：用于收集训练数据"""
    
    def __init__(self, cost_evaluation):
        """
        初始化数据收集器
        
        Args:
            cost_evaluation: 成本评估器
        """
        self.cost_evaluation = cost_evaluation
        self.collected_data = []
        
    def collect_sample(self,
                      workload,
                      indexes: List,
                      materialized_views: List,
                      query_features: np.ndarray,
                      index_features: np.ndarray,
                      mv_features: np.ndarray):
        """
        收集一个样本
        
        Args:
            workload: 工作负载
            indexes: 索引列表
            materialized_views: 物化视图列表
            query_features: 查询特征
            index_features: 索引特征
            mv_features: 物化视图特征
        """
        try:
            # 获取真实成本
            true_cost = self.cost_evaluation.calculate_cost(
                workload, indexes, materialized_views
            )
            
            self.collected_data.append({
                'query_features': query_features,
                'index_features': index_features,
                'mv_features': mv_features,
                'true_cost': true_cost
            })
            
        except Exception as e:
            logging.warning(f"Failed to collect sample: {e}")
    
    def get_dataset(self) -> CostEstimationDataset:
        """
        获取收集的数据集
        
        Returns:
            CostEstimationDataset
        """
        if not self.collected_data:
            raise ValueError("No data collected")
        
        query_features = np.array([d['query_features'] for d in self.collected_data])
        index_features = np.array([d['index_features'] for d in self.collected_data])
        mv_features = np.array([d['mv_features'] for d in self.collected_data])
        true_costs = np.array([d['true_cost'] for d in self.collected_data])
        
        return CostEstimationDataset(query_features, index_features, mv_features, true_costs)
    
    def save_data(self, path: str):
        """保存收集的数据"""
        np.savez(
            path,
            query_features=np.array([d['query_features'] for d in self.collected_data]),
            index_features=np.array([d['index_features'] for d in self.collected_data]),
            mv_features=np.array([d['mv_features'] for d in self.collected_data]),
            true_costs=np.array([d['true_cost'] for d in self.collected_data])
        )
        logging.info(f"Data saved to {path}")
    
    def load_data(self, path: str):
        """加载数据"""
        data = np.load(path)
        self.collected_data = []
        
        for i in range(len(data['true_costs'])):
            self.collected_data.append({
                'query_features': data['query_features'][i],
                'index_features': data['index_features'][i],
                'mv_features': data['mv_features'][i],
                'true_cost': data['true_costs'][i]
            })
        
        logging.info(f"Loaded {len(self.collected_data)} samples from {path}")
    
    def clear(self):
        """清除收集的数据"""
        self.collected_data.clear()


def create_cost_estimator(query_dim: int,
                         index_dim: int,
                         mv_dim: int,
                         config: Dict[str, Any] = None) -> Tuple[MLPCostEstimator, MLPCostEstimatorTrainer]:
    """
    创建成本估算器和训练器
    
    Args:
        query_dim: 查询特征维度
        index_dim: 索引特征维度
        mv_dim: 物化视图特征维度
        config: 配置字典
        
    Returns:
        (MLPCostEstimator, MLPCostEstimatorTrainer)
    """
    config = config or {}
    
    model = MLPCostEstimator(
        query_dim=query_dim,
        index_dim=index_dim,
        mv_dim=mv_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        dropout_rate=config.get('dropout_rate', 0.2),
        use_batch_norm=config.get('use_batch_norm', True)
    )
    
    trainer = MLPCostEstimatorTrainer(
        model=model,
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5),
        device=config.get('device', None)
    )
    
    return model, trainer


def train_cost_estimator_from_data(train_data_path: str,
                                   val_data_path: str = None,
                                   config: Dict[str, Any] = None,
                                   save_path: str = None) -> MLPCostEstimatorTrainer:
    """
    从数据文件训练成本估算器
    
    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        config: 配置字典
        save_path: 模型保存路径
        
    Returns:
        训练好的MLPCostEstimatorTrainer
    """
    config = config or {}
    
    # 加载训练数据
    train_data = np.load(train_data_path)
    train_dataset = CostEstimationDataset(
        train_data['query_features'],
        train_data['index_features'],
        train_data['mv_features'],
        train_data['true_costs']
    )
    
    # 推断维度
    query_dim = train_data['query_features'].shape[1]
    index_dim = train_data['index_features'].shape[1]
    mv_dim = train_data['mv_features'].shape[1]
    
    # 创建模型和训练器
    model, trainer = create_cost_estimator(query_dim, index_dim, mv_dim, config)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )
    
    val_loader = None
    if val_data_path:
        val_data = np.load(val_data_path)
        val_dataset = CostEstimationDataset(
            val_data['query_features'],
            val_data['index_features'],
            val_data['mv_features'],
            val_data['true_costs']
        )
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32))
    
    # 训练
    trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=config.get('num_epochs', 100),
        early_stopping_patience=config.get('early_stopping_patience', 20),
        save_path=save_path
    )
    
    return trainer

