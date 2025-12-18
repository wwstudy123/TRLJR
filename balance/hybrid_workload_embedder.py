# balance/hybrid_workload_embedder.py
"""
混合工作负载嵌入器模块
融合查询嵌入、索引特征和物化视图特征
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import gensim
from sklearn.decomposition import PCA

from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.materialized_view import MaterializedView
from index_selection_evaluation.selection.workload import Workload, Query
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation

from .workload_embedder import WorkloadEmbedder, PlanEmbedder
from .boo import BagOfOperators


class MLPCostEstimator:
    """MLP成本估算器"""
    
    def __init__(self, input_dim: int = 100, hidden_dims: List[int] = [64, 32], output_dim: int = 50):
        """
        初始化MLP成本估算器
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # 简化实现：使用随机初始化的权重
        # 实际应用中应该训练这些权重
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hidden_dim) * 0.01)
            self.biases.append(np.zeros(hidden_dim))
            prev_dim = hidden_dim
        
        self.weights.append(np.random.randn(prev_dim, output_dim) * 0.01)
        self.biases.append(np.zeros(output_dim))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            输出特征
        """
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU激活
        
        return x
    
    def estimate_cost_features(self, query_features: np.ndarray, 
                              index_features: np.ndarray,
                              mv_features: np.ndarray) -> np.ndarray:
        """
        估算成本特征
        
        Args:
            query_features: 查询特征
            index_features: 索引特征
            mv_features: 物化视图特征
            
        Returns:
            成本估算特征
        """
        # 拼接输入特征
        combined = np.concatenate([
            query_features.flatten()[:self.input_dim // 3],
            index_features.flatten()[:self.input_dim // 3],
            mv_features.flatten()[:self.input_dim // 3]
        ])
        
        # 填充或截断到input_dim
        if len(combined) < self.input_dim:
            combined = np.pad(combined, (0, self.input_dim - len(combined)))
        else:
            combined = combined[:self.input_dim]
        
        return self.forward(combined)


class HybridWorkloadEmbedder(WorkloadEmbedder):
    """
    混合工作负载嵌入器
    融合查询嵌入、索引候选特征和物化视图候选特征
    """
    
    def __init__(self, 
                 query_texts: List,
                 representation_size: int,
                 database_connector,
                 index_candidates: List[Index],
                 mv_candidates: List[MaterializedView],
                 columns: List = None,
                 cost_feature_dim: int = 50):
        """
        初始化混合工作负载嵌入器
        
        Args:
            query_texts: 查询文本列表
            representation_size: 基础表示维度
            database_connector: 数据库连接器
            index_candidates: 索引候选列表
            mv_candidates: 物化视图候选列表
            columns: 列信息
            cost_feature_dim: 成本特征维度
        """
        super().__init__(query_texts, representation_size, database_connector, columns)
        
        self.index_candidates = index_candidates
        self.mv_candidates = mv_candidates
        self.cost_feature_dim = cost_feature_dim
        
        # 初始化MLP成本估算器
        self.cost_estimator = MLPCostEstimator(
            input_dim=representation_size + len(index_candidates) + len(mv_candidates),
            output_dim=cost_feature_dim
        )
        
        # 总表示维度 = 查询嵌入 + 索引特征 + 物化视图特征 + MLP成本估算特征
        self.true_representation_size = (
            representation_size +           # 原始查询嵌入
            len(index_candidates) +         # 索引候选特征
            len(mv_candidates) +            # 物化视图候选特征
            cost_feature_dim                # MLP成本估算特征
        )
        
        # BOO创建器用于提取查询计划特征
        self.boo_creator = BagOfOperators()
        
        # 缓存
        self.embedding_cache = {}
        self.index_feature_cache = {}
        self.mv_feature_cache = {}
        
        # 构建索引列映射
        self._build_index_column_mapping()
        
        # 构建物化视图模式映射
        self._build_mv_pattern_mapping()
        
        logging.info(f"HybridWorkloadEmbedder initialized with representation size: {self.true_representation_size}")
    
    def _build_index_column_mapping(self):
        """构建索引到列的映射"""
        self.index_column_map = {}
        
        for i, index in enumerate(self.index_candidates):
            if hasattr(index, 'columns'):
                col_names = tuple(str(col) for col in index.columns)
                self.index_column_map[i] = col_names
    
    def _build_mv_pattern_mapping(self):
        """构建物化视图模式映射"""
        self.mv_patterns = {}
        
        for i, mv in enumerate(self.mv_candidates):
            patterns = self._extract_mv_patterns(mv)
            self.mv_patterns[i] = patterns
    
    def _extract_mv_patterns(self, mv: MaterializedView) -> Dict[str, Any]:
        """
        提取物化视图的模式特征
        
        Args:
            mv: 物化视图对象
            
        Returns:
            模式特征字典
        """
        patterns = {
            'tables': set(),
            'columns': set(),
            'aggregations': set(),
            'has_group_by': False,
            'has_join': False,
            'predicates': []
        }
        
        sql = mv.definition_sql.upper()
        
        # 提取表名
        from_pattern = r'FROM\s+([\w\s,]+?)(?:WHERE|GROUP|ORDER|JOIN|$)'
        from_match = re.search(from_pattern, sql)
        if from_match:
            tables_str = from_match.group(1)
            for table in tables_str.split(','):
                table_name = table.strip().split()[0].lower()
                if table_name:
                    patterns['tables'].add(table_name)
        
        # 检测JOIN
        if 'JOIN' in sql:
            patterns['has_join'] = True
            join_pattern = r'(\w+)\s+JOIN\s+(\w+)'
            join_matches = re.findall(join_pattern, sql)
            for t1, t2 in join_matches:
                patterns['tables'].add(t1.lower())
                patterns['tables'].add(t2.lower())
        
        # 检测GROUP BY
        if 'GROUP BY' in sql:
            patterns['has_group_by'] = True
        
        # 检测聚合函数
        for agg in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']:
            if agg in sql:
                patterns['aggregations'].add(agg)
        
        return patterns
    
    def get_embeddings(self, workload_or_plans) -> List[np.ndarray]:
        """
        获取工作负载嵌入
        
        Args:
            workload_or_plans: 工作负载对象或查询计划列表
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        
        # 判断输入类型
        if isinstance(workload_or_plans, Workload):
            queries = workload_or_plans.queries
        elif isinstance(workload_or_plans, list):
            if len(workload_or_plans) > 0 and isinstance(workload_or_plans[0], Query):
                queries = workload_or_plans
            else:
                # 假设是查询计划
                return self._get_embeddings_from_plans(workload_or_plans)
        else:
            queries = [workload_or_plans]
        
        for query in queries:
            embedding = self._get_single_query_embedding(query)
            embeddings.append(embedding)
        
        return embeddings
    
    def _get_single_query_embedding(self, query: Query) -> np.ndarray:
        """
        获取单个查询的嵌入
        
        Args:
            query: 查询对象
            
        Returns:
            嵌入向量
        """
        cache_key = str(query.text) if hasattr(query, 'text') else str(query)
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # 1. 获取基础查询嵌入
        base_embedding = self._get_base_query_embedding(query)
        
        # 2. 生成索引候选特征
        index_features = self._generate_index_features(query)
        
        # 3. 生成物化视图候选特征
        mv_features = self._generate_mv_features(query)
        
        # 4. MLP成本估算特征
        cost_features = self._estimate_cost_features(base_embedding, index_features, mv_features)
        
        # 5. 融合所有特征
        full_embedding = np.concatenate([
            base_embedding, index_features, mv_features, cost_features
        ])
        
        self.embedding_cache[cache_key] = full_embedding
        return full_embedding
    
    def _get_embeddings_from_plans(self, plans: List) -> List[np.ndarray]:
        """
        从查询计划获取嵌入
        
        Args:
            plans: 查询计划列表
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        
        for plan in plans:
            # 从计划提取BOO特征
            boo = self.boo_creator.boo_from_plan(plan)
            
            # 转换为基础嵌入
            base_embedding = self._boo_to_embedding(boo)
            
            # 生成索引和MV特征
            index_features = self._generate_index_features_from_plan(plan)
            mv_features = self._generate_mv_features_from_plan(plan)
            
            # 成本特征
            cost_features = self._estimate_cost_features(base_embedding, index_features, mv_features)
            
            # 融合
            full_embedding = np.concatenate([
                base_embedding, index_features, mv_features, cost_features
            ])
            
            embeddings.append(full_embedding)
        
        return embeddings
    
    def _get_base_query_embedding(self, query: Query) -> np.ndarray:
        """
        获取基础查询嵌入
        
        Args:
            query: 查询对象
            
        Returns:
            基础嵌入向量
        """
        query_text = query.text if hasattr(query, 'text') else str(query)
        
        # 使用简单的词袋模型
        tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
        
        # 创建特征向量
        embedding = np.zeros(self.representation_size)
        
        # 基于token计算特征
        for i, token in enumerate(tokens[:self.representation_size]):
            # 使用token的hash值作为特征
            hash_val = hash(token) % self.representation_size
            embedding[hash_val] += 1
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _boo_to_embedding(self, boo: List[str]) -> np.ndarray:
        """
        将BOO转换为嵌入向量
        
        Args:
            boo: Bag of Operators列表
            
        Returns:
            嵌入向量
        """
        embedding = np.zeros(self.representation_size)
        
        for op in boo:
            hash_val = hash(op) % self.representation_size
            embedding[hash_val] += 1
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _generate_index_features(self, query: Query) -> np.ndarray:
        """
        生成索引候选特征
        
        Args:
            query: 查询对象
            
        Returns:
            索引特征向量
        """
        index_features = np.zeros(len(self.index_candidates))
        query_text = query.text.upper() if hasattr(query, 'text') else str(query).upper()
        
        # 提取查询中涉及的列
        query_columns = self._extract_columns_from_query(query_text)
        
        for i, index in enumerate(self.index_candidates):
            # 计算索引与查询的匹配度
            match_score = self._calculate_index_match_score(index, query_columns, query_text)
            index_features[i] = match_score
        
        return index_features
    
    def _generate_index_features_from_plan(self, plan: Dict) -> np.ndarray:
        """
        从查询计划生成索引特征
        
        Args:
            plan: 查询计划
            
        Returns:
            索引特征向量
        """
        index_features = np.zeros(len(self.index_candidates))
        
        # 从计划中提取涉及的表和列
        tables, columns = self._extract_tables_columns_from_plan(plan)
        
        for i, index in enumerate(self.index_candidates):
            if hasattr(index, 'columns'):
                # 检查索引列是否在计划涉及的列中
                index_cols = set(str(col) for col in index.columns)
                match_count = len(index_cols & columns)
                index_features[i] = match_count / len(index_cols) if index_cols else 0
        
        return index_features
    
    def _extract_tables_columns_from_plan(self, plan: Dict) -> Tuple[set, set]:
        """
        从查询计划中提取表和列
        
        Args:
            plan: 查询计划
            
        Returns:
            (表集合, 列集合)
        """
        tables = set()
        columns = set()
        
        def traverse(node):
            if 'Relation Name' in node:
                tables.add(node['Relation Name'].lower())
            
            if 'Index Cond' in node:
                cols = re.findall(r'(\w+\.\w+|\w+)', node['Index Cond'])
                columns.update(col.lower() for col in cols)
            
            if 'Filter' in node:
                cols = re.findall(r'(\w+\.\w+|\w+)', node['Filter'])
                columns.update(col.lower() for col in cols)
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse(child)
        
        traverse(plan)
        return tables, columns
    
    def _extract_columns_from_query(self, query_text: str) -> set:
        """
        从查询文本中提取列名
        
        Args:
            query_text: 查询文本
            
        Returns:
            列名集合
        """
        columns = set()
        
        # 匹配表.列 或 单独的列名
        col_pattern = r'(\w+\.\w+|\b[a-z_]\w*\b)'
        matches = re.findall(col_pattern, query_text.lower())
        
        # 过滤SQL关键字
        sql_keywords = {'select', 'from', 'where', 'and', 'or', 'join', 'on', 
                       'group', 'by', 'order', 'having', 'limit', 'as', 'in',
                       'between', 'like', 'is', 'null', 'not', 'exists'}
        
        for match in matches:
            if match not in sql_keywords:
                columns.add(match)
        
        return columns
    
    def _calculate_index_match_score(self, index: Index, query_columns: set, 
                                    query_text: str) -> float:
        """
        计算索引与查询的匹配分数
        
        Args:
            index: 索引对象
            query_columns: 查询涉及的列
            query_text: 查询文本
            
        Returns:
            匹配分数 (0-1)
        """
        if not hasattr(index, 'columns') or not index.columns:
            return 0.0
        
        index_cols = set(str(col).lower() for col in index.columns)
        
        # 计算列重叠度
        overlap = len(index_cols & query_columns)
        if not index_cols:
            return 0.0
        
        base_score = overlap / len(index_cols)
        
        # 检查是否在WHERE子句中
        where_bonus = 0.0
        if 'WHERE' in query_text:
            for col in index_cols:
                if col in query_text.lower():
                    where_bonus += 0.1
        
        # 检查是否在ORDER BY中
        order_bonus = 0.0
        if 'ORDER BY' in query_text:
            for col in index_cols:
                if col in query_text.lower():
                    order_bonus += 0.1
        
        return min(1.0, base_score + where_bonus + order_bonus)
    
    def _generate_mv_features(self, query: Query) -> np.ndarray:
        """
        生成物化视图候选特征
        
        Args:
            query: 查询对象
            
        Returns:
            物化视图特征向量
        """
        mv_features = np.zeros(len(self.mv_candidates))
        query_text = query.text if hasattr(query, 'text') else str(query)
        
        for i, mv_candidate in enumerate(self.mv_candidates):
            # 计算查询与物化视图的匹配度
            match_score = self._calculate_mv_match_score(query_text, mv_candidate)
            mv_features[i] = match_score
        
        return mv_features
    
    def _generate_mv_features_from_plan(self, plan: Dict) -> np.ndarray:
        """
        从查询计划生成物化视图特征
        
        Args:
            plan: 查询计划
            
        Returns:
            物化视图特征向量
        """
        mv_features = np.zeros(len(self.mv_candidates))
        
        # 从计划中提取特征
        tables, columns = self._extract_tables_columns_from_plan(plan)
        has_aggregation = self._plan_has_aggregation(plan)
        has_sort = self._plan_has_sort(plan)
        
        for i, mv in enumerate(self.mv_candidates):
            patterns = self.mv_patterns.get(i, {})
            
            # 计算匹配分数
            score = 0.0
            
            # 表重叠
            mv_tables = patterns.get('tables', set())
            if mv_tables and tables:
                table_overlap = len(mv_tables & tables) / len(mv_tables)
                score += table_overlap * 0.4
            
            # 聚合匹配
            if has_aggregation and patterns.get('aggregations'):
                score += 0.3
            
            # GROUP BY匹配
            if patterns.get('has_group_by'):
                score += 0.2
            
            # JOIN匹配
            if patterns.get('has_join') and len(tables) > 1:
                score += 0.1
            
            mv_features[i] = min(1.0, score)
        
        return mv_features
    
    def _plan_has_aggregation(self, plan: Dict) -> bool:
        """检查计划是否包含聚合"""
        def check(node):
            if node.get('Node Type') in ['Aggregate', 'HashAggregate', 'GroupAggregate']:
                return True
            if 'Plans' in node:
                return any(check(child) for child in node['Plans'])
            return False
        return check(plan)
    
    def _plan_has_sort(self, plan: Dict) -> bool:
        """检查计划是否包含排序"""
        def check(node):
            if node.get('Node Type') == 'Sort':
                return True
            if 'Plans' in node:
                return any(check(child) for child in node['Plans'])
            return False
        return check(plan)
    
    def _calculate_mv_match_score(self, query_text: str, mv: MaterializedView) -> float:
        """
        计算查询与物化视图的匹配分数
        
        Args:
            query_text: 查询文本
            mv: 物化视图对象
            
        Returns:
            匹配分数 (0-1)
        """
        query_upper = query_text.upper()
        mv_sql = mv.definition_sql.upper()
        
        score = 0.0
        
        # 1. 表名匹配
        query_tables = self._extract_tables_from_sql(query_upper)
        mv_tables = self._extract_tables_from_sql(mv_sql)
        
        if query_tables and mv_tables:
            table_overlap = len(query_tables & mv_tables) / max(len(query_tables), 1)
            score += table_overlap * 0.3
        
        # 2. 聚合函数匹配
        query_aggs = self._extract_aggregations(query_upper)
        mv_aggs = self._extract_aggregations(mv_sql)
        
        if query_aggs and mv_aggs:
            agg_overlap = len(query_aggs & mv_aggs) / max(len(query_aggs), 1)
            score += agg_overlap * 0.25
        
        # 3. GROUP BY匹配
        query_has_group = 'GROUP BY' in query_upper
        mv_has_group = 'GROUP BY' in mv_sql
        
        if query_has_group and mv_has_group:
            score += 0.2
        
        # 4. JOIN匹配
        query_has_join = 'JOIN' in query_upper
        mv_has_join = 'JOIN' in mv_sql
        
        if query_has_join and mv_has_join:
            score += 0.15
        
        # 5. 列名匹配
        query_cols = self._extract_columns_from_query(query_upper)
        mv_cols = self._extract_columns_from_query(mv_sql)
        
        if query_cols and mv_cols:
            col_overlap = len(query_cols & mv_cols) / max(len(query_cols), 1)
            score += col_overlap * 0.1
        
        return min(1.0, score)
    
    def _extract_tables_from_sql(self, sql: str) -> set:
        """从SQL中提取表名"""
        tables = set()
        
        # FROM子句
        from_pattern = r'FROM\s+([\w\s,]+?)(?:WHERE|GROUP|ORDER|JOIN|HAVING|LIMIT|$)'
        from_match = re.search(from_pattern, sql)
        if from_match:
            for table in from_match.group(1).split(','):
                table_name = table.strip().split()[0].lower()
                if table_name:
                    tables.add(table_name)
        
        # JOIN子句
        join_pattern = r'JOIN\s+(\w+)'
        join_matches = re.findall(join_pattern, sql)
        tables.update(t.lower() for t in join_matches)
        
        return tables
    
    def _extract_aggregations(self, sql: str) -> set:
        """从SQL中提取聚合函数"""
        aggs = set()
        for agg in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']:
            if agg in sql:
                aggs.add(agg)
        return aggs
    
    def _estimate_cost_features(self, base_embedding: np.ndarray,
                               index_features: np.ndarray,
                               mv_features: np.ndarray) -> np.ndarray:
        """
        估算成本特征
        
        Args:
            base_embedding: 基础嵌入
            index_features: 索引特征
            mv_features: 物化视图特征
            
        Returns:
            成本估算特征
        """
        return self.cost_estimator.estimate_cost_features(
            base_embedding, index_features, mv_features
        )
    
    def get_workload_representation(self, workload: Workload) -> np.ndarray:
        """
        获取完整工作负载表示
        
        Args:
            workload: 工作负载对象
            
        Returns:
            工作负载表示向量
        """
        embeddings = self.get_embeddings(workload)
        
        # 聚合所有查询的嵌入
        if embeddings:
            # 加权平均（按查询频率）
            weights = []
            for query in workload.queries:
                weights.append(query.frequency if hasattr(query, 'frequency') else 1)
            
            weights = np.array(weights) / sum(weights)
            workload_repr = np.average(embeddings, axis=0, weights=weights)
        else:
            workload_repr = np.zeros(self.true_representation_size)
        
        return workload_repr
    
    def clear_cache(self):
        """清除缓存"""
        self.embedding_cache.clear()
        self.index_feature_cache.clear()
        self.mv_feature_cache.clear()


class HybridPlanEmbedder(PlanEmbedder):
    """
    混合查询计划嵌入器
    扩展PlanEmbedder以支持索引和物化视图特征
    """
    
    def __init__(self, 
                 query_texts: List,
                 representation_size: int,
                 database_connector,
                 columns: List,
                 index_candidates: List[Index],
                 mv_candidates: List[MaterializedView],
                 cost_feature_dim: int = 50):
        """
        初始化混合计划嵌入器
        
        Args:
            query_texts: 查询文本列表
            representation_size: 基础表示维度
            database_connector: 数据库连接器
            columns: 列信息
            index_candidates: 索引候选列表
            mv_candidates: 物化视图候选列表
            cost_feature_dim: 成本特征维度
        """
        super().__init__(query_texts, representation_size, database_connector, columns)
        
        self.index_candidates = index_candidates
        self.mv_candidates = mv_candidates
        self.cost_feature_dim = cost_feature_dim
        
        # 更新总表示维度
        self.true_representation_size = (
            representation_size + 10 +      # 原始 + 值编码
            len(index_candidates) +         # 索引特征
            len(mv_candidates) +            # 物化视图特征
            cost_feature_dim                # 成本特征
        )
        
        # 创建辅助嵌入器用于特征生成
        self.hybrid_embedder = HybridWorkloadEmbedder(
            query_texts, representation_size, None,
            index_candidates, mv_candidates, columns, cost_feature_dim
        )
    
    def get_embeddings(self, plans: List) -> List[np.ndarray]:
        """
        获取计划嵌入（带索引和MV特征）
        
        Args:
            plans: 查询计划列表
            
        Returns:
            嵌入向量列表
        """
        # 获取基础嵌入
        base_embeddings = super().get_embeddings(plans)
        
        # 获取索引和MV特征
        hybrid_embeddings = self.hybrid_embedder._get_embeddings_from_plans(plans)
        
        # 融合
        result = []
        for base_emb, hybrid_emb in zip(base_embeddings, hybrid_embeddings):
            # 提取hybrid中的索引、MV和成本特征
            offset = self.hybrid_embedder.representation_size
            index_mv_cost_features = hybrid_emb[offset:]
            
            # 拼接
            full_emb = np.concatenate([base_emb, index_mv_cost_features])
            result.append(full_emb)
        
        return result


def create_hybrid_embedder(query_texts: List,
                          representation_size: int,
                          database_connector,
                          index_candidates: List[Index],
                          mv_candidates: List[MaterializedView],
                          columns: List = None,
                          config: Dict[str, Any] = None) -> HybridWorkloadEmbedder:
    """
    创建混合工作负载嵌入器的工厂函数
    
    Args:
        query_texts: 查询文本列表
        representation_size: 表示维度
        database_connector: 数据库连接器
        index_candidates: 索引候选列表
        mv_candidates: 物化视图候选列表
        columns: 列信息
        config: 配置字典
        
    Returns:
        HybridWorkloadEmbedder实例
    """
    config = config or {}
    
    return HybridWorkloadEmbedder(
        query_texts=query_texts,
        representation_size=representation_size,
        database_connector=database_connector,
        index_candidates=index_candidates,
        mv_candidates=mv_candidates,
        columns=columns,
        cost_feature_dim=config.get('cost_feature_dim', 50)
    )

