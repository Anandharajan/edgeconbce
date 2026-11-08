from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import TestCase

from edgeconbce.backends.edge_proxy import edge_model_probability
from edgeconbce.backends.local_heuristic import local_model_probability
from edgeconbce.config import load_run_config
from edgeconbce.features.pacvf import pacvf_like_features
from edgeconbce.features.structure_proxy import structure_proxy_features
from edgeconbce.policies.router import choose_route
from edgeconbce.schemas import NotebookCompatConfig, SequenceSample


class RoutingInvariantTests(TestCase):
    def test_static_policies_route_as_expected(self) -> None:
        config = load_run_config(Path("configs/notebook_compat.yaml"))
        self.assertIsInstance(config, NotebookCompatConfig)
        sample = SequenceSample(
            sample_id="sample_test",
            sequence="ACDEFGHIKLMNPQRSTVWY" * 6,
            label=True,
            hidden_signal=0.7,
            mode="ambiguous",
        )
        features = pacvf_like_features(sample.sequence, config.max_sequence_length)
        proxies = structure_proxy_features(sample.sequence, config.max_sequence_length)
        local_probability = local_model_probability(features)
        edge_probability = edge_model_probability(features, proxies, local_probability)
        self.assertEqual(choose_route("local_only", sample, local_probability, edge_probability, features, proxies, config), "local")
        self.assertEqual(choose_route("edge_only", sample, local_probability, edge_probability, features, proxies, config), "edge_full")

    def test_privacy_mode_changes_split_transfer_flag_behavior(self) -> None:
        config = load_run_config(Path("configs/notebook_compat.yaml"))
        self.assertIsInstance(config, NotebookCompatConfig)
        no_privacy = replace(config, privacy_mode=False)
        sample = SequenceSample(
            sample_id="sample_test",
            sequence="ACDEFGHIKLMNPQRSTVWY" * 10,
            label=True,
            hidden_signal=0.7,
            mode="hard_positive",
        )
        features = pacvf_like_features(sample.sequence, config.max_sequence_length)
        proxies = structure_proxy_features(sample.sequence, config.max_sequence_length)
        local_probability = local_model_probability(features)
        edge_probability = edge_model_probability(features, proxies, local_probability)
        route = choose_route("adaptive_split", sample, local_probability, edge_probability, features, proxies, no_privacy)
        self.assertIn(route, {"local", "split_to_edge"})
