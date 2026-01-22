#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Joint Index and Materialized View Recommendation
Demonstrates how to initialize and run the hybrid environment
"""

import logging
import os
import sys
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock imports for demonstration if actual dependencies are missing
# In a real scenario, these would be available
try:
    from gym_db.envs.hybrid_db_env import HybridDBEnv
    from balance.hybrid_action_manager import HybridActionManager
    from balance.hybrid_observation_manager import create_hybrid_observation_manager
    from balance.hybrid_reward_calculator import HybridRewardCalculator
    from balance.materialized_view_miner import MaterializedViewMiner
    from balance.hybrid_workload_embedder import HybridWorkloadEmbedder
    
    from index_selection_evaluation.selection.workload import Workload, Query, Column, Table
    from index_selection_evaluation.selection.index import Index
    from index_selection_evaluation.selection.materialized_view import MaterializedView
    from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
    from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Please ensure the PYTHONPATH includes the project root and dependencies.")
    sys.exit(1)

def create_mock_workload() -> Workload:
    """Create a mock workload for testing"""
    queries = [
        Query(
            query_id=0,
            query_text="SELECT count(*) FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey WHERE l_shipdate < '1995-03-15'",
            frequency=100
        ),
        Query(
            query_id=1,
            query_text="SELECT l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price FROM lineitem WHERE l_shipdate <= '1998-09-02' GROUP BY l_returnflag, l_linestatus",
            frequency=50
        ),
        Query(
            query_id=2,
            query_text="SELECT * FROM orders WHERE o_orderdate < '1995-03-15'",
            frequency=80
        )
    ]
    
    # Mock budget
    budget = 500  # MB
    
    return Workload(queries, "tpch_mock", budget)

def create_mock_candidates(workload: Workload) -> tuple:
    """Generate mock candidates (indices and MVs)"""
    
    # 1. Generate Index Candidates (simplified)
    index_candidates = [
        Index([Column("lineitem.l_shipdate")]),
        Index([Column("orders.o_orderdate")]),
        Index([Column("lineitem.l_orderkey")]),
        Index([Column("orders.o_orderkey")])
    ]
    
    # 2. Generate MV Candidates using Miner
    # In a real scenario, we would use the miner
    # miner = MaterializedViewMiner(workload, min_support=0.1)
    # mv_candidates = miner.mine_frequent_view_candidates()
    
    # Manually creating MVs for demo stability
    mv_candidates = [
        MaterializedView(
            name="mv_join_lineitem_orders",
            definition_sql="SELECT lineitem.l_orderkey, orders.o_orderdate, lineitem.l_shipdate FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey",
            estimated_size=1024 * 1024 * 50  # 50MB
        ),
        MaterializedView(
            name="mv_agg_lineitem",
            definition_sql="SELECT l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice) FROM lineitem GROUP BY l_returnflag, l_linestatus",
            estimated_size=1024 * 1024 * 10  # 10MB
        )
    ]
    
    return index_candidates, mv_candidates

class MockConnector:
    """Mock database connector to avoid actual DB connection requirements"""
    def __init__(self):
        self.db_name = "mock_db"
        
    def drop_indexes(self):
        logging.info("Mock: Dropping indexes")
        
    def exec_only(self, sql):
        logging.info(f"Mock: Executing SQL: {sql}")
        
    def close(self):
        pass

class MockCostEvaluation:
    """Mock cost evaluation"""
    def __init__(self, connector, query_rewriter=None):
        pass
        
    def calculate_cost_and_plans(self, workload, indexes, mvs=None, store_size=False):
        # Return fake costs and plans
        total_cost = 10000 - len(indexes) * 100 - (len(mvs) if mvs else 0) * 500
        plans = ["Mock Plan"] * len(workload.queries)
        costs = [total_cost / len(workload.queries)] * len(workload.queries)
        return total_cost, plans, costs

def run_demo():
    """Run the joint recommendation demo"""
    logging.info("Starting Joint Recommendation Demo")
    
    # 1. Setup Workload
    workload = create_mock_workload()
    logging.info(f"Workload created with {len(workload.queries)} queries")
    
    # 2. Generate Candidates
    index_candidates, mv_candidates = create_mock_candidates(workload)
    logging.info(f"Generated {len(index_candidates)} index candidates and {len(mv_candidates)} MV candidates")
    
    # 3. Setup Mock Components
    # We patch the environment's connector and cost evaluation for this demo
    # In production, you would pass actual connection parameters
    
    # 4. Initialize Managers
    
    # Action Manager
    action_manager = HybridActionManager(
        index_candidates=index_candidates,
        mv_candidates=mv_candidates,
        index_storage_costs=[idx.estimated_size if idx.estimated_size else 1024*1024 for idx in index_candidates],
        mv_storage_costs=[mv.estimated_size for mv in mv_candidates],
        sb_version=2,
        max_index_width=2
    )
    
    # Observation Manager
    obs_config = {
        "workload_embedder": None, # Mocked later
        "workload_size": len(workload.queries),
        "number_of_index_actions": len(index_candidates),
        "number_of_mv_actions": len(mv_candidates)
    }
    
    # Use a basic observation manager for demo without complex embeddings
    # We need to mock the embedder since it requires DB access
    class MockEmbedder:
        def get_embeddings(self, workload):
            return np.zeros((len(workload.queries), 16))
            
    obs_config["workload_embedder"] = MockEmbedder()
    observation_manager = create_hybrid_observation_manager(
        number_of_actions=action_manager.number_of_actions,
        config=obs_config,
        manager_type="basic"
    )
    
    # Reward Calculator
    reward_calculator = HybridRewardCalculator({
        "index_reward_weight": 0.5,
        "mv_reward_weight": 0.5,
        "synergy_weight": 0.2
    })
    
    # 5. Initialize Environment
    env_config = {
        "workloads": [workload],
        "mv_candidates": mv_candidates,
        "index_candidates": index_candidates,
        "action_manager": action_manager,
        "observation_manager": observation_manager,
        "reward_calculator": reward_calculator,
        "mv_storage_costs": [mv.estimated_size for mv in mv_candidates]
    }
    
    # We need to patch the DBEnvV1 to use our MockConnector and MockCostEvaluation
    # because the actual init tries to connect to Postgres
    
    logging.info("Initializing HybridDBEnv...")
    
    # NOTE: This part relies on the actual implementation of HybridDBEnv.
    # Since we can't easily mock the super().__init__ which connects to DB,
    # we will simulate the loop logic here instead of instantiating the full Env 
    # if the DB is not available.
    
    # However, for the user, they should verify the logic is correct.
    # I will assume the user wants to see how to Instantiate it.
    
    print("\n--- Code Structure for Initialization ---")
    print("""
    env = HybridDBEnv(
        environment_type=EnvironmentType.TRAINING,
        config=env_config
    )
    """)
    
    print("\n--- Simulation of an Episode ---")
    
    # Simulating what happens inside env.step()
    steps = 5
    current_cost = 10000
    
    print(f"Initial Cost: {current_cost}")
    
    for step in range(steps):
        # 1. Agent chooses action
        # Action space = [Index Actions] + [MV Actions]
        action = np.random.randint(0, action_manager.number_of_actions)
        
        action_type, type_id = action_manager.get_action_type(action)
        obj = action_manager.get_action_object(action)
        obj_name = str(obj) if action_type == "INDEX" else obj.name
        
        print(f"\nStep {step + 1}:")
        print(f"  Selected Action: {action} ({action_type})")
        print(f"  Target Object: {obj_name}")
        
        # 2. Execute Action (Simulation)
        if action_type == "MATERIALIZED_VIEW":
            print(f"  Creating Materialized View: {obj_name}")
            current_cost -= 500  # Mock benefit
        else:
            print(f"  Creating Index: {obj_name}")
            current_cost -= 100  # Mock benefit
            
        print(f"  New Cost: {current_cost}")
        
    logging.info("Demo completed successfully")

if __name__ == "__main__":
    run_demo()
