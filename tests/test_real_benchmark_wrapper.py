from __future__ import annotations

from unittest import TestCase

from edgeconbce.pipeline.real_benchmark import _augment_legacy_results, _policy_data_movement_summary
from edgeconbce.schemas import DeviceCarbonConfig, LegacyBP3RealV3Config


class RealBenchmarkWrapperTests(TestCase):
    def test_policy_data_movement_summary_estimates_ascii_bytes(self) -> None:
        data_movement = _policy_data_movement_summary(
            [
                {"length": 120, "raw_sequence_shared": False},
                {"length": 80, "raw_sequence_shared": True},
            ]
        )
        self.assertEqual(data_movement["sequence_count"], 2)
        self.assertEqual(data_movement["shared_sequence_count"], 1)
        self.assertEqual(data_movement["total_bytes_transferred"], 80)
        self.assertEqual(data_movement["max_bytes_transferred"], 80)
        self.assertEqual(data_movement["raw_sequence_share_rate"], 0.5)
        self.assertEqual(data_movement["mean_bytes_transferred"], 40.0)

    def test_augment_legacy_results_adds_provenance_and_summary_fields(self) -> None:
        config = LegacyBP3RealV3Config(
            mode="legacy_bp3_real_v3",
            output_json_path="outputs/test.json",
            seed=42,
            train_fasta_path="train.fasta",
            test_fasta_path="test.fasta",
            validation_fraction=0.2,
            route_budget_max=0.5,
            threshold_grid=(0.5,),
            route_probability_grid=(0.5,),
            edge_blend_weight_grid=(1.0,),
            smoothing_radius_grid=(0,),
            smoothing_alpha_grid=(0.0,),
            local_device=DeviceCarbonConfig("local", 10.0, 1.0, 100),
            edge_device=DeviceCarbonConfig("edge", 20.0, 2.0, 100),
            grid_intensity_gco2_per_kwh=500.0,
        )
        payload = {
            "external_test": {
                "adaptive_split": {
                    "summary": {"mcc": 0.2},
                    "per_sequence": [
                        {"length": 50, "raw_sequence_shared": False},
                        {"length": 100, "raw_sequence_shared": True},
                    ],
                }
            }
        }

        augmented = _augment_legacy_results(payload, config)
        self.assertEqual(augmented["artifact_version"], 2)
        self.assertEqual(augmented["benchmark_mode"], "legacy_bp3_real_v3")
        self.assertIn("measurement_provenance", augmented)
        self.assertEqual(
            augmented["external_test"]["adaptive_split"]["summary"]["raw_sequence_share_rate"],
            0.5,
        )
        self.assertEqual(
            augmented["external_test"]["adaptive_split"]["summary"]["mean_bytes_transferred"],
            50.0,
        )
