"""Score a preselected large-replication S2 population without re-matching."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen14b_native_s2 import sequence_loglik  # noqa: E402
from run_qwen14b_reproduction import LocalQwen  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    population = read_jsonl(args.population)
    baselines = {row["trajectory_id"]: row for row in read_jsonl(args.native_dir / "server_native_baselines.jsonl")}
    surgeries = {
        (row["trajectory_id"], int(row["target_turn"])): row
        for row in read_jsonl(args.native_dir / "server_native_surgeries.jsonl")
    }
    if not population:
        raise RuntimeError("Empty S2 population")
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, max_new_tokens=1)
    config = {
        "stage": "S2_large_replication_exact_metric_v1",
        "population": str(args.population),
        "population_version": "FORMAL_HELDOUT_118",
        "rows": len(population),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode,
        "continuation_scoring": "content_tokens_only_teacher_forced",
        "eos_included": False,
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for selected in population:
        key = (selected["trajectory_id"], int(selected["target_turn"]))
        row = surgeries[key]
        baseline = baselines[key[0]]
        c_orig = baseline["generated_history"][:-1]
        c_surg = row["surgery_history"][:-1]
        y_failure = row["original_output_local"]
        y_success = row["surgery_output_local"]
        success_orig = sequence_loglik(engine, c_orig, y_success)
        failure_orig = sequence_loglik(engine, c_orig, y_failure)
        success_surg = sequence_loglik(engine, c_surg, y_success)
        failure_surg = sequence_loglik(engine, c_surg, y_failure)
        margin_orig = success_orig["mean_log_likelihood"] - failure_orig["mean_log_likelihood"]
        margin_surg = success_surg["mean_log_likelihood"] - failure_surg["mean_log_likelihood"]
        append_jsonl(
            args.output_dir / "s2_sequence_scores.jsonl",
            {
                "population_version": "FORMAL_HELDOUT_118",
                "match_id": selected["match_id"],
                "control_rank": selected.get("control_rank"),
                "trajectory_id": key[0],
                "target_turn": key[1],
                "task": row["task"],
                "s2_group": selected["s2_group"],
                "effect_class_local": row["effect_class_local"],
                "ll_success_original_context": success_orig,
                "ll_failure_original_context": failure_orig,
                "ll_success_surgery_context": success_surg,
                "ll_failure_surgery_context": failure_surg,
                "margin_original": margin_orig,
                "margin_surgery": margin_surg,
                "delta_margin": margin_surg - margin_orig,
            },
        )
        print(json.dumps({"trajectory_id": key[0], "target_turn": key[1], "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()

