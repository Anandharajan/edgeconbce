from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgeconbce.config import load_run_config
from edgeconbce.pipeline.notebook_compat import run_benchmark as run_notebook_compat
from edgeconbce.pipeline.repeated_eval import run_repeated_bp3_eval
from edgeconbce.pipeline.real_benchmark import run_legacy_bp3_real_v3
from edgeconbce.schemas import LegacyBP3RealV3Config, NotebookCompatConfig, RepeatedBP3EvalConfig


def run_from_config_path(config_path: str | Path) -> dict[str, Any]:
    config = load_run_config(config_path)
    if isinstance(config, NotebookCompatConfig):
        return run_notebook_compat(config)
    if isinstance(config, LegacyBP3RealV3Config):
        return run_legacy_bp3_real_v3(config)
    if isinstance(config, RepeatedBP3EvalConfig):
        return run_repeated_bp3_eval(config)
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EdgeConBCE benchmark configs.")
    parser.add_argument("--config", required=True, help="Path to a YAML benchmark config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = run_from_config_path(args.config)
    if "policy_results" in results:
        output_path = results["config"]["output_json_path"]
        print(Path(output_path).resolve())
        return 0
    if "reports" in results:
        print(Path(results["reports"]["summary_markdown"]).resolve())
        return 0
    print("Benchmark finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
