from __future__ import annotations

from unittest import TestCase

import numpy as np

from edgeconbce_real_benchmark_v2 import (
    carbon_guard_priority_scores,
    ranked_route_mask,
    route_count_from_rate,
)


class LegacyBP3RealBenchmarkV3Tests(TestCase):
    def test_carbon_guard_priority_scores_penalise_longer_sequences(self) -> None:
        priority_scores = carbon_guard_priority_scores(
            route_probability_values=np.asarray([0.95, 0.93], dtype=np.float64),
            sequence_lengths=np.asarray([250.0, 100.0], dtype=np.float64),
            length_penalty=0.20,
        )
        self.assertLess(priority_scores[0], priority_scores[1])

    def test_ranked_route_mask_selects_top_scores(self) -> None:
        routed_mask = ranked_route_mask(
            np.asarray([0.30, 0.90, 0.90, 0.10], dtype=np.float64),
            route_count=2,
        )
        self.assertEqual(routed_mask.tolist(), [False, True, True, False])

    def test_route_count_from_rate_rounds_nonzero_target_rate(self) -> None:
        self.assertEqual(route_count_from_rate(15, 0.101), 2)
        self.assertEqual(route_count_from_rate(15, 0.0), 0)
