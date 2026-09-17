"""Component 5: Intelligent Relocation Planner -- the project's core
novel contribution. Turns Stage 4's predictions + Stage 5's dependency
graph + current shard load into a concrete, cost-justified relocation
plan: which records to move, from which shard, to which shard, and why.

For each candidate record i and destination shard j:

    ExpectedValue(i, j) = P(hotspot_i) * Benefit(i, j)
                        - RelocationCost(i, j)
                        - (1 - P(hotspot_i)) * WastedCost(i, j)
                        - CrossShardPenalty(i, j)

Benefit(i, j) is defined directly as the reduction in cluster load
variance a hypothetical move would produce, so "destination chosen to
minimize resulting load variance" falls out of the formula rather than
needing a separate rule. Only positive-expected-value moves make the
plan. Exact optimal assignment is NP-hard, so this uses the
methodology's greedy heuristic: at each step, pick the single best
(candidate, destination) pair by expected value, apply it, update the
shard-load and placement estimates, then recompute every remaining
candidate's expected value against that updated state before picking
the next move.
"""

from dataclasses import dataclass

import numpy as np

from common.shard_client import ShardCluster
from planner.dependency_graph import DependencyGraph

RELOCATION_COST = 1.0  # flat cost per record moved
WASTED_COST_MULTIPLIER = 1.5  # wasted-effort penalty beyond the move cost if the prediction was wrong
CROSS_SHARD_PENALTY_WEIGHT = 0.05  # penalty per unit of co-access weight split across shards


@dataclass
class RelocationCandidate:
    record_id: str
    source_shard: str
    p_hotspot: float
    confidence: float
    predicted_load: float


@dataclass
class RelocationMove:
    record_id: str
    source_shard: str
    destination_shard: str
    predicted_cost: float
    predicted_benefit: float
    expected_value: float
    confidence: float


class RelocationPlanner:
    def __init__(self, cluster: ShardCluster = None, graph: DependencyGraph = None):
        self.cluster = cluster or ShardCluster()
        self.graph = graph

    @staticmethod
    def _variance(loads: dict) -> float:
        return float(np.var(list(loads.values())))

    def _cross_shard_penalty(self, record_id: str, destination: str, placement: dict) -> float:
        if self.graph is None:
            return 0.0
        penalty = 0.0
        for neighbor, weight in self.graph.neighbors(record_id).items():
            neighbor_shard = placement.get(neighbor, self.cluster.shard_for_key(neighbor))
            if neighbor_shard != destination:
                penalty += weight * CROSS_SHARD_PENALTY_WEIGHT
        return penalty

    def _evaluate(self, candidate: RelocationCandidate, destination: str, loads: dict, placement: dict):
        """Returns (expected_value, benefit, total_cost) for moving `candidate` to `destination`."""
        source = candidate.source_shard
        if destination == source:
            return None

        hypothetical = dict(loads)
        hypothetical[source] -= candidate.predicted_load
        hypothetical[destination] += candidate.predicted_load

        benefit = max(0.0, self._variance(loads) - self._variance(hypothetical))
        wasted_cost = RELOCATION_COST * WASTED_COST_MULTIPLIER
        cross_shard_penalty = self._cross_shard_penalty(candidate.record_id, destination, placement)
        total_cost = RELOCATION_COST + (1 - candidate.p_hotspot) * wasted_cost + cross_shard_penalty

        expected_value = candidate.p_hotspot * benefit - total_cost
        return expected_value, benefit, total_cost

    def plan(self, candidates: list, initial_loads: dict) -> list:
        """Greedily assigns each move to whichever (candidate, destination)
        pair currently has the highest expected value, re-evaluating
        everything remaining after each assignment. Returns a list of
        RelocationMove, empty if no candidate ever has a positive EV."""
        loads = dict(initial_loads)
        placement = {c.record_id: c.source_shard for c in candidates}
        remaining = list(candidates)
        moves = []

        while remaining:
            best = None
            for candidate in remaining:
                for destination in self.cluster.shard_names():
                    result = self._evaluate(candidate, destination, loads, placement)
                    if result is None:
                        continue
                    ev, benefit, cost = result
                    if ev <= 0:
                        continue
                    if best is None or ev > best[0]:
                        best = (ev, candidate, destination, benefit, cost)

            if best is None:
                break

            ev, candidate, destination, benefit, cost = best
            moves.append(
                RelocationMove(
                    record_id=candidate.record_id,
                    source_shard=candidate.source_shard,
                    destination_shard=destination,
                    predicted_cost=cost,
                    predicted_benefit=benefit,
                    expected_value=ev,
                    confidence=candidate.confidence,
                )
            )

            loads[candidate.source_shard] -= candidate.predicted_load
            loads[destination] += candidate.predicted_load
            placement[candidate.record_id] = destination
            remaining.remove(candidate)

        return moves
