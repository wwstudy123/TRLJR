# balance/materialized_view_miner.py
"""
物化视图候选生成模块
使用频繁模式挖掘方法生成物化视图候选
"""

import re
import logging
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
from itertools import combinations

from index_selection_evaluation.selection.workload import Workload, Query
from index_selection_evaluation.selection.materialized_view import MaterializedView


class MaterializedViewMiner:
    """物化视图挖掘器：使用频繁模式挖掘生成物化视图候选"""
    
    def __init__(self, workload: Workload, min_support: float = 0.3, max_mv_count: int = 10):
        """
        初始化物化视图挖掘器
        
        Args:
            workload: 工作负载对象
            min_support: 最小支持度阈值
            max_mv_count: 最大物化视图候选数量
        """
        self.workload = workload
        self.min_support = min_support
        self.max_mv_count = max_mv_count
        self.total_frequency = sum(q.frequency for q in self.workload.queries)
        
    def mine_frequent_view_candidates(self) -> List[MaterializedView]:
        """
        使用频繁模式挖掘生成物化视图候选
        
        Returns:
            物化视图候选列表
        """
        logging.info(f"Mining frequent view candidates with min_support={self.min_support}")
        
        # 1. 提取查询中的表连接模式
        join_patterns = self._extract_join_patterns()
        logging.info(f"Extracted {len(join_patterns)} frequent join patterns")
        
        # 2. 提取频繁谓词组合
        frequent_predicates = self._mine_frequent_predicates()
        logging.info(f"Extracted {len(frequent_predicates)} frequent predicate patterns")
        
        # 3. 提取频繁聚合模式
        frequent_aggregations = self._mine_frequent_aggregations()
        logging.info(f"Extracted {len(frequent_aggregations)} frequent aggregation patterns")
        
        # 4. 提取频繁GROUP BY模式
        frequent_group_by = self._mine_frequent_group_by()
        logging.info(f"Extracted {len(frequent_group_by)} frequent GROUP BY patterns")
        
        # 5. 生成物化视图候选
        mv_candidates = self._generate_mv_candidates(
            join_patterns, frequent_predicates, frequent_aggregations, frequent_group_by
        )
        
        logging.info(f"Generated {len(mv_candidates)} materialized view candidates")
        return mv_candidates
    
    def _extract_join_patterns(self) -> Dict[Tuple[str, ...], int]:
        """
        提取表连接模式
        
        Returns:
            频繁连接模式字典 {(table1, table2): frequency}
        """
        join_frequency = {}
        
        for query in self.workload.queries:
            joins = self._parse_joins_from_query(query.text)
            for join in joins:
                join_key = tuple(sorted(join))
                join_frequency[join_key] = join_frequency.get(join_key, 0) + query.frequency
        
        # 返回频率超过阈值的连接模式
        return {
            join: freq for join, freq in join_frequency.items() 
            if freq / self.total_frequency >= self.min_support
        }
    
    def _parse_joins_from_query(self, query_text: str) -> List[Tuple[str, str]]:
        """
        从查询中解析表连接
        
        Args:
            query_text: 查询SQL文本
            
        Returns:
            连接的表对列表
        """
        joins = []
        query_upper = query_text.upper()
        
        # 解析显式JOIN
        # 匹配 table1 JOIN table2 ON condition
        join_pattern = r'(\w+)\s+(?:INNER\s+)?(?:LEFT\s+)?(?:RIGHT\s+)?(?:OUTER\s+)?JOIN\s+(\w+)'
        matches = re.findall(join_pattern, query_upper)
        for match in matches:
            table1, table2 = match[0].lower(), match[1].lower()
            joins.append((table1, table2))
        
        # 解析FROM子句中的隐式连接 (FROM table1, table2 WHERE ...)
        from_pattern = r'FROM\s+([\w\s,]+?)(?:WHERE|GROUP|ORDER|HAVING|LIMIT|$)'
        from_match = re.search(from_pattern, query_upper)
        if from_match:
            tables_str = from_match.group(1)
            # 提取表名（忽略别名）
            tables = []
            for part in tables_str.split(','):
                table_parts = part.strip().split()
                if table_parts:
                    tables.append(table_parts[0].lower())
            
            # 生成表对
            if len(tables) > 1:
                for i in range(len(tables)):
                    for j in range(i + 1, len(tables)):
                        joins.append((tables[i], tables[j]))
        
        return joins
    
    def _mine_frequent_predicates(self) -> Dict[str, int]:
        """
        挖掘频繁谓词组合
        
        Returns:
            频繁谓词字典 {predicate_pattern: frequency}
        """
        predicate_frequency = defaultdict(int)
        
        for query in self.workload.queries:
            predicates = self._extract_predicates(query.text)
            for predicate in predicates:
                predicate_frequency[predicate] += query.frequency
        
        # 返回频率超过阈值的谓词
        return {
            pred: freq for pred, freq in predicate_frequency.items()
            if freq / self.total_frequency >= self.min_support
        }
    
    def _extract_predicates(self, query_text: str) -> List[str]:
        """
        从查询中提取谓词
        
        Args:
            query_text: 查询SQL文本
            
        Returns:
            谓词列表
        """
        predicates = []
        query_upper = query_text.upper()
        
        # 提取WHERE子句
        where_pattern = r'WHERE\s+(.+?)(?:GROUP|ORDER|HAVING|LIMIT|$)'
        where_match = re.search(where_pattern, query_upper, re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            # 提取比较谓词 (column op value)
            comparison_pattern = r'(\w+\.\w+|\w+)\s*(=|<>|!=|<=|>=|<|>|LIKE|IN|BETWEEN)\s*'
            comparisons = re.findall(comparison_pattern, where_clause)
            
            for col, op in comparisons:
                # 标准化谓词（移除具体值，保留列和操作符）
                predicate = f"{col.lower()}_{op}"
                predicates.append(predicate)
        
        return predicates
    
    def _mine_frequent_aggregations(self) -> Dict[str, int]:
        """
        挖掘频繁聚合模式
        
        Returns:
            频繁聚合字典 {aggregation_pattern: frequency}
        """
        aggregation_frequency = defaultdict(int)
        aggregation_functions = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']
        
        for query in self.workload.queries:
            query_upper = query.text.upper()
            
            for agg_func in aggregation_functions:
                # 匹配 AGG_FUNC(column) 模式
                pattern = rf'{agg_func}\s*\(\s*(\w+\.?\w*|\*)\s*\)'
                matches = re.findall(pattern, query_upper)
                
                for match in matches:
                    agg_pattern = f"{agg_func}({match.lower()})"
                    aggregation_frequency[agg_pattern] += query.frequency
        
        # 返回频率超过阈值的聚合模式
        return {
            agg: freq for agg, freq in aggregation_frequency.items()
            if freq / self.total_frequency >= self.min_support
        }
    
    def _mine_frequent_group_by(self) -> Dict[Tuple[str, ...], int]:
        """
        挖掘频繁GROUP BY模式
        
        Returns:
            频繁GROUP BY字典 {(col1, col2, ...): frequency}
        """
        group_by_frequency = defaultdict(int)
        
        for query in self.workload.queries:
            group_by_cols = self._extract_group_by(query.text)
            if group_by_cols:
                group_key = tuple(sorted(group_by_cols))
                group_by_frequency[group_key] += query.frequency
        
        # 返回频率超过阈值的GROUP BY模式
        return {
            group: freq for group, freq in group_by_frequency.items()
            if freq / self.total_frequency >= self.min_support
        }
    
    def _extract_group_by(self, query_text: str) -> List[str]:
        """
        从查询中提取GROUP BY列
        
        Args:
            query_text: 查询SQL文本
            
        Returns:
            GROUP BY列列表
        """
        query_upper = query_text.upper()
        
        # 匹配 GROUP BY col1, col2, ...
        group_pattern = r'GROUP\s+BY\s+([\w\s,\.]+?)(?:HAVING|ORDER|LIMIT|$)'
        group_match = re.search(group_pattern, query_upper)
        
        if group_match:
            cols_str = group_match.group(1)
            cols = [col.strip().lower() for col in cols_str.split(',')]
            return cols
        
        return []
    
    def _generate_mv_candidates(self, 
                               join_patterns: Dict[Tuple[str, ...], int],
                               frequent_predicates: Dict[str, int],
                               frequent_aggregations: Dict[str, int],
                               frequent_group_by: Dict[Tuple[str, ...], int]) -> List[MaterializedView]:
        """
        基于频繁模式生成物化视图候选
        
        Args:
            join_patterns: 频繁连接模式
            frequent_predicates: 频繁谓词
            frequent_aggregations: 频繁聚合
            frequent_group_by: 频繁GROUP BY
            
        Returns:
            物化视图候选列表
        """
        mv_candidates = []
        mv_id = 0
        
        # 策略1: 基于频繁连接模式生成物化视图
        for join_tables, freq in sorted(join_patterns.items(), key=lambda x: -x[1]):
            if len(mv_candidates) >= self.max_mv_count:
                break
                
            mv_name = f"mv_join_{mv_id}"
            tables_str = ", ".join(join_tables)
            
            # 构建简单的连接物化视图
            if len(join_tables) == 2:
                t1, t2 = join_tables
                mv_sql = f"SELECT * FROM {t1} JOIN {t2} ON {t1}.id = {t2}.{t1}_id"
            else:
                mv_sql = f"SELECT * FROM {tables_str}"
            
            mv = MaterializedView(
                name=mv_name,
                definition_sql=mv_sql,
                estimated_size=self._estimate_mv_size(join_tables, freq)
            )
            mv_candidates.append(mv)
            mv_id += 1
        
        # 策略2: 基于频繁聚合+GROUP BY模式生成物化视图
        for group_cols, group_freq in sorted(frequent_group_by.items(), key=lambda x: -x[1]):
            if len(mv_candidates) >= self.max_mv_count:
                break
            
            # 找到与此GROUP BY相关的聚合函数
            related_aggs = []
            for agg, agg_freq in frequent_aggregations.items():
                related_aggs.append(agg)
            
            if related_aggs:
                mv_name = f"mv_agg_{mv_id}"
                cols_str = ", ".join(group_cols)
                aggs_str = ", ".join(related_aggs[:3])  # 最多3个聚合
                
                # 从GROUP BY列推断表名
                table_name = self._infer_table_from_columns(group_cols)
                
                mv_sql = f"SELECT {cols_str}, {aggs_str} FROM {table_name} GROUP BY {cols_str}"
                
                mv = MaterializedView(
                    name=mv_name,
                    definition_sql=mv_sql,
                    estimated_size=self._estimate_mv_size(group_cols, group_freq)
                )
                mv_candidates.append(mv)
                mv_id += 1
        
        # 策略3: 基于频繁谓词生成选择性物化视图
        predicate_groups = self._group_predicates_by_table(frequent_predicates)
        for table, predicates in predicate_groups.items():
            if len(mv_candidates) >= self.max_mv_count:
                break
            
            mv_name = f"mv_filter_{mv_id}"
            # 构建带过滤条件的物化视图
            where_conditions = self._build_where_from_predicates(predicates[:3])
            
            if where_conditions:
                mv_sql = f"SELECT * FROM {table} WHERE {where_conditions}"
                
                mv = MaterializedView(
                    name=mv_name,
                    definition_sql=mv_sql,
                    estimated_size=1024 * 1024  # 默认1MB
                )
                mv_candidates.append(mv)
                mv_id += 1
        
        return mv_candidates[:self.max_mv_count]
    
    def _estimate_mv_size(self, pattern: Tuple, frequency: int) -> int:
        """
        估计物化视图大小
        
        Args:
            pattern: 模式元组
            frequency: 频率
            
        Returns:
            估计大小（字节）
        """
        # 基础大小 + 基于模式复杂度的因子
        base_size = 512 * 1024  # 512KB
        complexity_factor = len(pattern) * 0.5
        frequency_factor = min(frequency / self.total_frequency, 1.0)
        
        return int(base_size * (1 + complexity_factor) * (1 + frequency_factor))
    
    def _infer_table_from_columns(self, columns: Tuple[str, ...]) -> str:
        """
        从列名推断表名
        
        Args:
            columns: 列名元组
            
        Returns:
            推断的表名
        """
        for col in columns:
            if '.' in col:
                return col.split('.')[0]
        
        # 如果无法推断，返回通用名称
        return "source_table"
    
    def _group_predicates_by_table(self, predicates: Dict[str, int]) -> Dict[str, List[str]]:
        """
        按表分组谓词
        
        Args:
            predicates: 谓词字典
            
        Returns:
            按表分组的谓词 {table: [predicates]}
        """
        table_predicates = defaultdict(list)
        
        for predicate, freq in predicates.items():
            # 从谓词中提取表名
            if '.' in predicate:
                table = predicate.split('.')[0]
            else:
                # 尝试从列名推断表
                col_part = predicate.split('_')[0]
                table = self._infer_table_from_column_prefix(col_part)
            
            table_predicates[table].append(predicate)
        
        return dict(table_predicates)
    
    def _infer_table_from_column_prefix(self, column_prefix: str) -> str:
        """
        从列前缀推断表名
        
        Args:
            column_prefix: 列名前缀
            
        Returns:
            推断的表名
        """
        # 常见的列前缀到表名映射
        prefix_mapping = {
            'c_': 'customer',
            'o_': 'orders',
            'l_': 'lineitem',
            'p_': 'part',
            's_': 'supplier',
            'n_': 'nation',
            'r_': 'region',
            'ps_': 'partsupp',
        }
        
        for prefix, table in prefix_mapping.items():
            if column_prefix.startswith(prefix):
                return table
        
        return "unknown_table"
    
    def _build_where_from_predicates(self, predicates: List[str]) -> str:
        """
        从谓词列表构建WHERE子句
        
        Args:
            predicates: 谓词列表
            
        Returns:
            WHERE子句字符串
        """
        conditions = []
        
        for predicate in predicates:
            # 解析谓词格式: column_operator
            parts = predicate.rsplit('_', 1)
            if len(parts) == 2:
                col, op = parts
                
                # 根据操作符添加占位符值
                if op in ('=', '<>', '!='):
                    conditions.append(f"{col} {op} 'placeholder'")
                elif op in ('<', '<=', '>', '>='):
                    conditions.append(f"{col} {op} 0")
                elif op == 'LIKE':
                    conditions.append(f"{col} LIKE '%pattern%'")
                elif op == 'IN':
                    conditions.append(f"{col} IN ('value1', 'value2')")
                elif op == 'BETWEEN':
                    conditions.append(f"{col} BETWEEN 0 AND 100")
        
        return " AND ".join(conditions) if conditions else ""


class JointCandidateGenerator:
    """联合候选生成器：索引 + 物化视图"""
    
    def __init__(self, max_index_width: int = 3, max_mv_count: int = 10, min_support: float = 0.3):
        """
        初始化联合候选生成器
        
        Args:
            max_index_width: 最大索引宽度
            max_mv_count: 最大物化视图数量
            min_support: 最小支持度阈值
        """
        self.max_index_width = max_index_width
        self.max_mv_count = max_mv_count
        self.min_support = min_support
    
    def generate_joint_candidates(self, workload: Workload) -> Tuple[List, List[MaterializedView]]:
        """
        生成联合候选集：索引 + 物化视图
        
        Args:
            workload: 工作负载
            
        Returns:
            (索引候选列表, 物化视图候选列表)
        """
        from index_selection_evaluation.selection.candidate_generation import syntactically_relevant_indexes
        
        # 生成索引候选
        index_candidates = []
        for query in workload.queries:
            query_indexes = syntactically_relevant_indexes(query, self.max_index_width)
            index_candidates.extend(query_indexes)
        
        # 去重索引候选
        unique_indexes = list(set(index_candidates))
        
        # 使用频繁模式挖掘生成物化视图候选
        mv_miner = MaterializedViewMiner(workload, self.min_support, self.max_mv_count)
        mv_candidates = mv_miner.mine_frequent_view_candidates()
        
        logging.info(f"Generated {len(unique_indexes)} index candidates and {len(mv_candidates)} MV candidates")
        
        return unique_indexes, mv_candidates


def create_mv_miner(workload: Workload, config: Dict[str, Any] = None) -> MaterializedViewMiner:
    """
    创建物化视图挖掘器的工厂函数
    
    Args:
        workload: 工作负载
        config: 配置字典
        
    Returns:
        MaterializedViewMiner实例
    """
    config = config or {}
    return MaterializedViewMiner(
        workload=workload,
        min_support=config.get('min_support', 0.3),
        max_mv_count=config.get('max_mv_count', 10)
    )

