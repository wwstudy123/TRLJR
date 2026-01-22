import datetime
import logging

from .what_if_index_creation import WhatIfIndexCreation
from .query_rewriter import NoOpQueryRewriter


class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="whatif", query_rewriter=None):
        logging.debug("Init cost evaluation")
        self.db_connector = db_connector
        self.cost_estimation = cost_estimation
        logging.info("Cost estimation with " + self.cost_estimation)
        self.what_if = WhatIfIndexCreation(db_connector)
        self.current_indexes = set()
        self.current_materialized_views = set()
        
        # Initialize query rewriter for MV support
        self.query_rewriter = query_rewriter or NoOpQueryRewriter()

        assert len(self.what_if.all_simulated_indexes()) == len(self.current_indexes)

        self.cost_requests = 0
        self.cache_hits = 0
        # Cache structure:
        # {(query_object, relevant_indexes, relevant_mvs): cost}
        self.cache = {}

        # Cache structure:
        # {(query_object, relevant_indexes, relevant_mvs): (cost, plan)}
        self.cache_plans = {}

        self.completed = False
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        self.relevant_indexes_cache = {}
        self.relevant_mvs_cache = {}

        self.costing_time = datetime.timedelta(0)

    def estimate_size(self, index):
        # TODO: Refactor: It is currently too complicated to compute
        # We must search in current indexes to get an index object with .hypopg_oid
        result = None
        for i in self.current_indexes:
            if index == i:
                result = i
                break
        if result:
            # Index does currently exist and size can be queried
            if not index.estimated_size:
                index.estimated_size = self.what_if.estimate_index_size(result.hypopg_oid)
        else:
            self._simulate_or_create_index(index, store_size=True)

    def which_indexes_utilized_and_cost(self, query, indexes):
        self._prepare_cost_calculation(indexes, store_size=True)

        plan = self.db_connector.get_plan(query)
        cost = plan["Total Cost"]
        plan_str = str(plan)

        recommended_indexes = set()

        # We are iterating over the CostEvalution's indexes and not over `indexes`
        # because it is not guaranteed that hypopg_name is set for all items in
        # `indexes`. This is caused by _prepare_cost_calculation that only creates
        # indexes which are not yet existing. If there is no hypothetical index
        # created for an index object, there is no hypopg_name assigned to it. However,
        # all items in current_indexes must also have an equivalent in `indexes`.
        for index in self.current_indexes:
            assert (
                index in indexes
            ), "Something went wrong with _prepare_cost_calculation."

            if index.hypopg_name not in plan_str:
                continue
            recommended_indexes.add(index)

        return recommended_indexes, cost

    def calculate_cost(self, workload, indexes, mvs=None, store_size=False):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        self._prepare_cost_calculation_mvs(mvs or [])
        total_cost = 0

        # TODO: Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._request_cache(query, indexes, mvs or []) * query.frequency
        return total_cost

    def calculate_cost_and_plans(self, workload, indexes, mvs=None, store_size=False):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        start_time = datetime.datetime.now()

        self._prepare_cost_calculation(indexes, store_size=store_size)
        self._prepare_cost_calculation_mvs(mvs or [])
        total_cost = 0
        plans = []
        costs = []

        for query in workload.queries:
            self.cost_requests += 1
            cost, plan = self._request_cache_plans(query, indexes, mvs or [])
            total_cost += cost * query.frequency
            plans.append(plan)
            costs.append(cost)

        end_time = datetime.datetime.now()
        self.costing_time += end_time - start_time

        return total_cost, plans, costs

    # Creates the current index combination by simulating/creating
    # missing indexes and unsimulating/dropping indexes
    # that exist but are not in the combination.
    def _prepare_cost_calculation(self, indexes, store_size=False):
        for index in set(indexes) - self.current_indexes:
            self._simulate_or_create_index(index, store_size=store_size)
        for index in self.current_indexes - set(indexes):
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set(indexes)

    def _prepare_cost_calculation_mvs(self, mvs):
        """Prepare materialized views for cost calculation"""
        for mv in set(mvs) - self.current_materialized_views:
            self._create_materialized_view(mv)
        for mv in self.current_materialized_views - set(mvs):
            self._drop_materialized_view(mv)
        self.current_materialized_views = set(mvs)

    def _create_materialized_view(self, mv):
        """Create a materialized view"""
        self.db_connector.create_materialized_view(mv)
        self.current_materialized_views.add(mv)

    def _drop_materialized_view(self, mv):
        """Drop a materialized view"""
        self.db_connector.drop_materialized_view(mv)
        self.current_materialized_views.remove(mv)

    def _simulate_or_create_index(self, index, store_size=False):
        if self.cost_estimation == "whatif":
            self.what_if.simulate_index(index, store_size=store_size)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.create_index(index)
        self.current_indexes.add(index)

    def _unsimulate_or_drop_index(self, index):
        if self.cost_estimation == "whatif":
            self.what_if.drop_simulated_index(index)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.drop_index(index)
        self.current_indexes.remove(index)

    def _get_cost(self, query):
        if self.cost_estimation == "whatif":
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            runtime = self.db_connector.exec_query(query)[0]
            return runtime

    def _get_cost_plan(self, query):
        query_plan = self.db_connector.get_plan(query)
        return query_plan["Total Cost"], query_plan

    def complete_cost_estimation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._unsimulate_or_drop_index(index)
        
        for mv in self.current_materialized_views.copy():
            self._drop_materialized_view(mv)

        assert self.current_indexes == set()
        assert self.current_materialized_views == set()

    def _request_cache(self, query, indexes, mvs=None):
        mvs = mvs or []
        q_i_m_hash = (query, frozenset(indexes), frozenset(mvs))
        
        if q_i_m_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_m_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_m_hash] = relevant_indexes
            
        if q_i_m_hash in self.relevant_mvs_cache:
            relevant_mvs = self.relevant_mvs_cache[q_i_m_hash]
        else:
            relevant_mvs = self._relevant_mvs(query, mvs)
            self.relevant_mvs_cache[q_i_m_hash] = relevant_mvs

        # Check if query and corresponding relevant indexes/mvs in cache
        cache_key = (query, relevant_indexes, relevant_mvs)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        # If no cache hit request cost from database system
        else:
            cost = self._get_cost_with_rewriting(query, relevant_mvs)
            self.cache[cache_key] = cost
            return cost

    def _request_cache_plans(self, query, indexes, mvs=None):
        mvs = mvs or []
        q_i_m_hash = (query, frozenset(indexes), frozenset(mvs))
        
        if q_i_m_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_m_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_m_hash] = relevant_indexes
            
        if q_i_m_hash in self.relevant_mvs_cache:
            relevant_mvs = self.relevant_mvs_cache[q_i_m_hash]
        else:
            relevant_mvs = self._relevant_mvs(query, mvs)
            self.relevant_mvs_cache[q_i_m_hash] = relevant_mvs

        # Check if query and corresponding relevant indexes/mvs in cache
        cache_key = (query, relevant_indexes, relevant_mvs)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        # If no cache hit request cost from database system
        else:
            cost, plan = self._get_cost_plan_with_rewriting(query, relevant_mvs)
            self.cache[cache_key] = (cost, plan)
            return cost, plan

    def _get_cost_with_rewriting(self, query, relevant_mvs):
        """Get cost considering query rewriting for materialized views"""
        if not relevant_mvs:
            return self._get_cost(query)
        
        # Get candidate rewritten queries
        rewritten_queries = self.query_rewriter.rewrite(query, relevant_mvs)
        
        # Find the best cost among all candidates
        best_cost = None
        for candidate_query in rewritten_queries:
            cost = self._get_cost(candidate_query)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                
        return best_cost

    def _get_cost_plan_with_rewriting(self, query, relevant_mvs):
        """Get cost and plan considering query rewriting for materialized views"""
        if not relevant_mvs:
            return self._get_cost_plan(query)
        
        # Get candidate rewritten queries
        rewritten_queries = self.query_rewriter.rewrite(query, relevant_mvs)
        
        # Find the best cost and plan among all candidates
        best_cost = None
        best_plan = None
        for candidate_query in rewritten_queries:
            cost, plan = self._get_cost_plan(candidate_query)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_plan = plan
                
        return best_cost, best_plan

    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)

    @staticmethod
    def _relevant_mvs(query, mvs):
        """Determine which materialized views are relevant for a query"""
        # For now, return all MVs - more sophisticated logic can be added later
        # This could include checking if the query can be rewritten to use the MV
        return frozenset(mvs)
