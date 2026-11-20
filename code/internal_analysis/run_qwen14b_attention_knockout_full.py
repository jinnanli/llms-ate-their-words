"""Run full native-history target-turn attention knockout on Qwen3-14B."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import socket
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen14b_attention_probe import render_with_offsets  # noqa: E402
from run_qwen14b_attention_knockout import CONDITIONS, generate_condition, mask_self_test  # noqa: E402
from run_qwen14b_reproduction import LocalQwen, extract_and_evaluate  # noqa: E402


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate-first-baseline", action="store_true")
    args = parser.parse_args()

    mask_self_test()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest)
    unique_keys = {(row["trajectory_id"], row["target_turn"]) for row in manifest}
    if not manifest or len(unique_keys) != len(manifest):
        raise RuntimeError("S3 manifest must contain non-empty unique branch keys")
    baselines = {row["trajectory_id"]: row for row in read_jsonl(args.native_dir / "server_native_baselines.jsonl")}
    payloads = {
        row["trajectory_id"]: row
        for row in (json.loads(line) for line in args.payload.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["trajectory_id"]].append(row)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)
    config = {
        "stage": "S3_full_native_history_target_turn_knockout_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "native_dir": str(args.native_dir),
        "payload": str(args.payload),
        "payload_sha256": sha256(args.payload),
        "branch_count": len(manifest),
        "trajectory_count": len(grouped),
        "conditions": list(CONDITIONS),
        "primary_label": "local_effect_class",
        "context_source": "server_native_baselines.generated_history",
        "baseline_generation_policy": "once_per_trajectory_reused_across_branches",
        "mask_scope": "target_assistant_content_tokens_only",
        "direct_only": "final_answer_decoding_queries_to_target_content",
        "full_downstream": "all_queries_after_target_content_to_target_content",
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    first_runtime_case = True
    for trajectory_id, branch_rows in sorted(grouped.items()):
        messages = baselines[trajectory_id]["generated_history"][:-1]
        reference_output = baselines[trajectory_id]["original_output_local"]
        sample = samples[payloads[trajectory_id]["source_instance_id"]]
        first_target = int(branch_rows[0]["target_turn"])
        first_span = render_with_offsets(engine.tokenizer, messages, first_target, args.thinking_mode)
        baseline_output, baseline_metadata = generate_condition(
            engine, messages, [], "baseline", args.prefill_chunk_size
        )
        _, baseline_evaluator, baseline_score = extract_and_evaluate(branch_rows[0]["task"], baseline_output, sample)
        if first_runtime_case and args.validate_first_baseline:
            standard_output, standard_metadata = engine.generate(messages)
            _, _, standard_score = extract_and_evaluate(branch_rows[0]["task"], standard_output, sample)
            validation = {
                "trajectory_id": trajectory_id,
                "target_turn": first_target,
                "manual_vs_standard_exact_output_match": baseline_output == standard_output,
                "manual_vs_standard_score_match": abs(baseline_score - standard_score) <= 1e-12,
                "manual_vs_native_reference_exact_output_match": baseline_output == reference_output,
                "standard_score": standard_score,
                "manual_score": baseline_score,
                "standard_generation_metadata": standard_metadata,
            }
            (args.output_dir / "baseline_invariance_check.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8"
            )
            if not validation["manual_vs_standard_exact_output_match"] or not validation["manual_vs_standard_score_match"]:
                raise RuntimeError("Manual baseline generation differs from standard HF generation")
            first_runtime_case = False

        for row in sorted(branch_rows, key=lambda item: int(item["target_turn"])):
            target_turn = int(row["target_turn"])
            span = render_with_offsets(engine.tokenizer, messages, target_turn, args.thinking_mode)
            target_indices = [
                token["token_index"] for token in span["token_rows"] if token["segment"] == "target_assistant_turn"
            ]
            if not target_indices:
                raise RuntimeError(f"Empty target span for {trajectory_id}:{target_turn}")
            append_jsonl(
                args.output_dir / "token_spans.jsonl",
                {
                    "population_version": row.get(
                        "population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"
                    ),
                    "source_shard": row.get("source_shard", "discovery"),
                    "trajectory_id": trajectory_id,
                    "target_turn": target_turn,
                    "target_token_indices": target_indices,
                    "target_token_start": min(target_indices),
                    "target_token_end_exclusive": max(target_indices) + 1,
                    "target_token_count": len(target_indices),
                    "prompt_token_count": len(span["input_ids"]),
                    "span_validation": "exact_render_offsets_and_template_id_match",
                },
            )
            condition_results = {
                "baseline": (baseline_output, baseline_score, baseline_evaluator, baseline_metadata)
            }
            for condition in ("direct_only", "full_downstream"):
                output, metadata = generate_condition(
                    engine, messages, target_indices, condition, args.prefill_chunk_size
                )
                _, evaluator, score = extract_and_evaluate(row["task"], output, sample)
                condition_results[condition] = (output, score, evaluator, metadata)
            for condition in CONDITIONS:
                output, score, evaluator, metadata = condition_results[condition]
                append_jsonl(
                    args.output_dir / "s3_knockout_results.jsonl",
                    {
                        "population_version": row.get(
                            "population_version", "DISCOVERY_FORMAL_ALIGNED_NATIVE_24"
                        ),
                        "source_shard": row.get("source_shard", "discovery"),
                        "trajectory_id": trajectory_id,
                        "target_turn": target_turn,
                        "task": row["task"],
                        "local_effect_class": row["local_effect_class"],
                        "local_delta": float(row["local_delta"]),
                        "formal_effect_class": row["formal_effect_class"],
                        "condition": condition,
                        "generated_output": output,
                        "evaluator_score": score,
                        "delta_vs_baseline": score - baseline_score,
                        "target_token_start": min(target_indices),
                        "target_token_end_exclusive": max(target_indices) + 1,
                        "number_masked_keys": 0 if condition == "baseline" else len(target_indices),
                        "evaluator_output": evaluator,
                        "generation_metadata": metadata,
                    },
                )
            print(json.dumps({"trajectory_id": trajectory_id, "target_turn": target_turn, "status": "completed"}), flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

