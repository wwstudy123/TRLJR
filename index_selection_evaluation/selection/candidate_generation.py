import itertools
import logging

from .index import Index
from .materialized_view import MaterializedView


def candidates_per_query(workload, max_index_width, candidate_generator):
    candidates = []

    for query in workload.queries:
        candidates.append(candidate_generator(query, max_index_width))

    return candidates


def joint_candidates_per_query(workload, max_index_width, max_mv_count, 
                              index_generator, mv_generator):
    """
    为每个查询生成联合候选：索引 + 物化视图
    
    Args:
        workload: 工作负载
        max_index_width: 最大索引宽度
        max_mv_count: 最大物化视图数量
        index_generator: 索引候选生成器
        mv_generator: 物化视图候选生成器
        
    Returns:
        (索引候选列表, 物化视图候选列表)
    """
    index_candidates = []
    mv_candidates = []
    
    for query in workload.queries:
        # 生成索引候选
        query_indexes = index_generator(query, max_index_width)
        index_candidates.append(query_indexes)
        
        # 生成物化视图候选
        query_mvs = mv_generator(query, max_mv_count)
        mv_candidates.append(query_mvs)
    
    return index_candidates, mv_candidates


def syntactically_relevant_materialized_views(query, max_mv_count=3):
    """
    基于语法相关性生成物化视图候选
    
    Args:
        query: 查询对象
        max_mv_count: 最大物化视图数量
        
    Returns:
        物化视图候选列表
    """
    mv_candidates = []
    
    # 简单的物化视图生成策略
    # 策略1: 基于GROUP BY的物化视图
    if "GROUP BY" in query.text.upper():
        mv_name = f"mv_group_by_{query.nr}"
        mv_sql = f"SELECT * FROM ({query.text}) AS {mv_name}"
        mv = MaterializedView(mv_name, mv_sql, estimated_size=1024*1024)
        mv_candidates.append(mv)
    
    # 策略2: 基于JOIN的物化视图
    if "JOIN" in query.text.upper():
        mv_name = f"mv_join_{query.nr}"
        mv_sql = f"SELECT * FROM ({query.text}) AS {mv_name}"
        mv = MaterializedView(mv_name, mv_sql, estimated_size=2*1024*1024)
        mv_candidates.append(mv)
    
    # 策略3: 基于聚合函数的物化视图
    aggregation_functions = ["SUM", "COUNT", "AVG", "MAX", "MIN"]
    for func in aggregation_functions:
        if func in query.text.upper():
            mv_name = f"mv_agg_{func.lower()}_{query.nr}"
            mv_sql = f"SELECT * FROM ({query.text}) AS {mv_name}"
            mv = MaterializedView(mv_name, mv_sql, estimated_size=512*1024)
            mv_candidates.append(mv)
            break
    
    return mv_candidates[:max_mv_count]


def syntactically_relevant_indexes(query, max_index_width):
    # "SAEFIS" or "BFI" see paper linked in DB2Advis algorithm
    # This implementation is "BFI" and uses all syntactically relevant indexes.
    columns = query.columns
    logging.debug(f"{query}")
    logging.debug(f"Indexable columns: {len(columns)}")

    indexable_columns_per_table = {}
    for column in columns:
        if column.table not in indexable_columns_per_table:
            indexable_columns_per_table[column.table] = set()
        indexable_columns_per_table[column.table].add(column)

    possible_column_combinations = set()
    for table in indexable_columns_per_table:
        columns = indexable_columns_per_table[table]
        for index_length in range(1, max_index_width + 1):
            possible_column_combinations |= set(
                itertools.permutations(columns, index_length)
            )

    logging.debug(f"Potential indexes: {len(possible_column_combinations)}")
    return [Index(p) for p in possible_column_combinations]

def syntactically_relevant_indexes_only_one(query, max_index_width):
    # "SAEFIS" or "BFI" see paper linked in DB2Advis algorithm
    # This implementation is "BFI" and uses all syntactically relevant indexes.
    columns = query.columns
    
    return [Index(p) for p in [[i] for i in columns]]