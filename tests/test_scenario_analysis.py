from __future__ import annotations

from pathlib import Path
from unittest import TestCase
import json
import shutil
import uuid

from edgeconbce.pipeline.scenario_analysis import ScenarioSpec, _policy_sequence_rows, generate_scenario_analysis


class ScenarioAnalysisTests(TestCase):
    def test_policy_sequence_rows_respect_plm_policy_thresholds_and_routing(self) -> None:
        run_payload = {
            "config": {
                "grid_intensity_gco2_per_kwh": 100.0,
                "edge_blend_weight_grid": [1.0],
                "local_device": {
                    "operational_power_w": 10.0,
                    "embodied_kgco2e_total": 1.0,
                    "amortization_requests": 100,
                },
                "edge_device": {
                    "operational_power_w": 20.0,
                    "embodied_kgco2e_total": 2.0,
                    "amortization_requests": 100,
                },
            },
            "model_selection": {
                "edge_optimisation": {
                    "selected_edge_weight": 1.0,
                }
            },
            "external_test": {
                "local_only": {
                    "decision_threshold": 0.5,
                    "per_sequence": [],
                },
                "edge_only": {
                    "decision_threshold": 0.5,
                    "per_sequence": [],
                },
                "plm_local_only": {
                    "decision_threshold": 0.7,
                    "per_sequence": [
                        {
                            "sequence_id": "seq1",
                            "length": 1,
                            "labels": [1],
                            "local_probabilities": [0.6],
                            "edge_probabilities": [0.4],
                            "local_latency_ms": 10.0,
                            "edge_latency_ms": 20.0,
                        }
                    ],
                },
                "plm_edge_only": {
                    "decision_threshold": 0.3,
                    "per_sequence": [
                        {
                            "sequence_id": "seq1",
                            "length": 1,
                            "labels": [1],
                            "local_probabilities": [0.6],
                            "edge_probabilities": [0.4],
                            "local_latency_ms": 10.0,
                            "edge_latency_ms": 20.0,
                        }
                    ],
                },
                "plm_adaptive_split": {
                    "decision_thresholds": {
                        "local": 0.7,
                        "edge": 0.3,
                    },
                    "per_sequence": [
                        {
                            "sequence_id": "seq1",
                            "length": 1,
                            "labels": [1],
                            "local_probabilities": [0.6],
                            "edge_probabilities": [0.4],
                            "local_latency_ms": 10.0,
                            "edge_latency_ms": 20.0,
                            "route": False,
                            "route_probability": 0.2,
                        },
                        {
                            "sequence_id": "seq2",
                            "length": 1,
                            "labels": [1],
                            "local_probabilities": [0.6],
                            "edge_probabilities": [0.4],
                            "local_latency_ms": 10.0,
                            "edge_latency_ms": 20.0,
                            "route": True,
                            "route_probability": 0.9,
                        },
                    ],
                },
            },
        }
        scenario = ScenarioSpec(name="baseline", description="Original benchmark configuration.")

        plm_local_rows = _policy_sequence_rows(run_payload, "plm_local_only", scenario)
        plm_edge_rows = _policy_sequence_rows(run_payload, "plm_edge_only", scenario)
        plm_adaptive_rows = _policy_sequence_rows(run_payload, "plm_adaptive_split", scenario)

        self.assertFalse(plm_local_rows[0]["edge_used"])
        self.assertEqual(plm_local_rows[0]["predictions"], [False])
        self.assertEqual(plm_local_rows[0]["active_latency_ms"], 10.0)

        self.assertTrue(plm_edge_rows[0]["edge_used"])
        self.assertEqual(plm_edge_rows[0]["predictions"], [True])
        self.assertEqual(plm_edge_rows[0]["active_latency_ms"], 20.0)

        self.assertFalse(plm_adaptive_rows[0]["edge_used"])
        self.assertEqual(plm_adaptive_rows[0]["predictions"], [False])
        self.assertTrue(plm_adaptive_rows[1]["edge_used"])
        self.assertEqual(plm_adaptive_rows[1]["predictions"], [True])
        self.assertEqual(plm_adaptive_rows[1]["active_latency_ms"], 30.0)

    def test_generate_scenario_analysis_reports_edge_fallback_and_grid_sensitivity(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"scenario_{uuid.uuid4().hex}"
        repeated_path = temp_dir / "repeated.json"
        run_path = temp_dir / "run_seed_1.json"
        output_dir = temp_dir / "analysis"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            run_payload = {
                "config": {
                    "seed": 1,
                    "grid_intensity_gco2_per_kwh": 100.0,
                    "edge_blend_weight_grid": [1.0],
                    "local_device": {
                        "operational_power_w": 360000.0,
                        "embodied_kgco2e_total": 1.0,
                        "amortization_requests": 100,
                    },
                    "edge_device": {
                        "operational_power_w": 720000.0,
                        "embodied_kgco2e_total": 2.0,
                        "amortization_requests": 100,
                    },
                },
                "model_selection": {
                    "edge_optimisation": {
                        "selected_edge_weight": 1.0,
                    }
                },
                "external_test": {
                    "local_only": {
                        "decision_threshold": 0.5,
                        "per_sequence": [
                            {
                                "sequence_id": "seq1",
                                "length": 10,
                                "labels": [1, 0],
                                "local_probabilities": [0.9, 0.1],
                                "edge_probabilities": [0.8, 0.2],
                                "local_latency_ms": 1000.0,
                                "edge_latency_ms": 2000.0,
                            },
                            {
                                "sequence_id": "seq2",
                                "length": 20,
                                "labels": [0, 1],
                                "local_probabilities": [0.2, 0.8],
                                "edge_probabilities": [0.1, 0.9],
                                "local_latency_ms": 1000.0,
                                "edge_latency_ms": 2000.0,
                            },
                        ],
                    },
                    "edge_only": {
                        "decision_threshold": 0.5,
                        "per_sequence": [
                            {
                                "sequence_id": "seq1",
                                "length": 10,
                                "labels": [1, 0],
                                "local_probabilities": [0.9, 0.1],
                                "edge_probabilities": [0.8, 0.2],
                                "local_latency_ms": 1000.0,
                                "edge_latency_ms": 2000.0,
                            },
                            {
                                "sequence_id": "seq2",
                                "length": 20,
                                "labels": [0, 1],
                                "local_probabilities": [0.2, 0.8],
                                "edge_probabilities": [0.1, 0.9],
                                "local_latency_ms": 1000.0,
                                "edge_latency_ms": 2000.0,
                            },
                        ],
                    },
                    "adaptive_split": {
                        "per_sequence": [
                            {
                                "sequence_id": "seq1",
                                "length": 10,
                                "labels": [1, 0],
                                "local_probabilities": [0.9, 0.1],
                                "edge_probabilities": [0.8, 0.2],
                                "local_latency_ms": 1000.0,
                                "edge_latency_ms": 2000.0,
                                "route": False,
                                "route_probability": 0.2,
                            },
                            {
                                "sequence_id": "seq2",
                                "length": 20,
                                "labels": [0, 1],
                                "local_probabilities": [0.2, 0.8],
                                "edge_probabilities": [0.1, 0.9],
                                "local_latency_ms": 1000.0,
                                "edge_latency_ms": 2000.0,
                                "route": True,
                                "route_probability": 0.9,
                            },
                        ],
                    },
                },
            }
            repeated_payload = {
                "run_artifacts": [
                    {
                        "seed": 1,
                        "output_json_path": str(run_path),
                    }
                ]
            }
            run_path.write_text(json.dumps(run_payload), encoding="utf-8")
            repeated_path.write_text(json.dumps(repeated_payload), encoding="utf-8")

            reports = generate_scenario_analysis(
                repeated_results_path=repeated_path,
                output_dir=output_dir,
            )

            payload = json.loads(Path(reports["reports"]["summary_json"]).read_text(encoding="utf-8"))
            by_key = {
                (row["scenario"], row["policy"]): row
                for row in payload["aggregate_policy_metrics"]
            }
            baseline_adaptive = by_key[("baseline", "adaptive_split")]
            edge_unavailable_adaptive = by_key[("edge_unavailable", "adaptive_split")]
            baseline_edge = by_key[("baseline", "edge_only")]
            high_grid_edge = by_key[("high_grid_1p5x", "edge_only")]

            self.assertEqual(baseline_adaptive["raw_sequence_share_rate_mean"], 0.5)
            self.assertEqual(baseline_adaptive["mean_bytes_transferred_mean"], 10.0)
            self.assertEqual(edge_unavailable_adaptive["route_rate_mean"], 0.0)
            self.assertEqual(edge_unavailable_adaptive["mean_bytes_transferred_mean"], 0.0)
            self.assertGreater(high_grid_edge["mean_total_gco2e_mean"], baseline_edge["mean_total_gco2e_mean"])
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
