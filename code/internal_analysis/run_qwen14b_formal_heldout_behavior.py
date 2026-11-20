"""Run exact server-native SHARDED baseline and uncapped surgery for FORMAL_HELDOUT_118."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_qwen14b_reproduction import ACK, LocalQwen, effect_class, extract_and_evaluate  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def generate_baseline(system: str, users: list[str], engine: LocalQwen) -> tuple[list[dict], list[dict]]:
    history = [{"role": "system", "content": system}]
    calls = []
    for turn, user in enumerate(users):
        history.append({"role": "user", "content": user})
        assistant, metadata = engine.generate(history)
        calls.append({"turn_index": turn, **metadata})
        history.append({"role": "assistant", "content": assistant})
    return history, calls


def generate_surgery(
    system: str,
    users: list[str],
    baseline_history: list[dict],
    target_turn: int,
    engine: LocalQwen,
) -> tuple[list[dict], list[dict]]:
    assistants = [row["content"] for row in baseline_history if row["role"] == "assistant"]
    history = [{"role": "system", "content": system}]
    calls = []
    for turn, user in enumerate(users):
        history.append({"role": "user", "content": user})
        if turn < target_turn:
            assistant = assistants[turn]
        elif turn == target_turn:
            assistant = ACK
        else:
            assistant, metadata = engine.generate(history)
            calls.append({"turn_index": turn, **metadata})
        history.append({"role": "assistant", "content": assistant})
    return history, calls


def content_tokens(engine: LocalQwen, text: str) -> int:
    return len(engine.tokenizer.encode(text, add_special_tokens=False))


def context_tokens(engine: LocalQwen, messages: list[dict]) -> int:
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    if engine.thinking_mode != "template_default":
        kwargs["enable_thinking"] = engine.thinking_mode == "enabled"
    return len(engine.tokenizer.apply_chat_template(messages, **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=["disabled", "enabled", "template_default"], default="disabled")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(args.source_manifest)
    payloads = {row["trajectory_id"]: row for row in read_jsonl(args.source_payload)}
    if not source_rows or len({row["trajectory_id"] for row in source_rows}) != len(source_rows):
        raise RuntimeError("Source manifest must contain unique trajectories")
    if set(payloads) != {row["trajectory_id"] for row in source_rows}:
        raise RuntimeError("Source manifest and payload trajectory sets differ")
    if any(row["population_version"] != "FORMAL_HELDOUT_118" for row in source_rows):
        raise RuntimeError("Unexpected population version")
    samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = LocalQwen(args.model, args.revision, args.thinking_mode, args.max_new_tokens)
    config = {
        "stage": "formal_heldout_118_native_behavior_v1",
        "population_version": "FORMAL_HELDOUT_118",
        "source_shard": source_rows[0]["source_shard"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_count": len(source_rows),
        "expected_branch_count": sum(int(row["eligible_target_turn_count"]) for row in source_rows),
        "selection_rule": "recovered_original_formal_degraded_membership_no_native_refilter",
        "target_turn_rule": "all_intermediate_assistant_turns_0_to_n_user_turns_minus_2",
        "baseline_generation_policy": "once_per_trajectory",
        "surgery_pre_target_history": "server_native_baseline",
        "model_revision": getattr(engine.model.config, "_commit_hash", None) or args.revision,
        "thinking_mode": args.thinking_mode,
        "do_sample": False,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_id": os.getenv("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for source in source_rows:
        trajectory_id = source["trajectory_id"]
        payload = payloads[trajectory_id]
        try:
            sample = samples[source["reference_id"]]
            system = payload["system_message"]
            users = payload["user_messages"]
            if len(users) != int(source["user_message_count"]):
                raise RuntimeError("Recovered user-message count changed")
            baseline_history, baseline_calls = generate_baseline(system, users, engine)
            baseline_output = baseline_history[-1]["content"]
            _, baseline_evaluator, baseline_score = extract_and_evaluate(source["task"], baseline_output, sample)
            baseline_context = baseline_history[:-1]
            assistant_history_count = sum(
                content_tokens(engine, row["content"]) for row in baseline_context if row["role"] == "assistant"
            )
            baseline_context_count = context_tokens(engine, baseline_context)
            baseline_row = {
                "trajectory_id": trajectory_id,
                "source_instance_id": source["instance_id"],
                "task": source["task"],
                "population_version": source["population_version"],
                "source_shard": source["source_shard"],
                "original_score_local": baseline_score,
                "original_output_local": baseline_output,
                "generated_history": baseline_history,
                "generation_calls": baseline_calls,
                "evaluator_output": baseline_evaluator,
                "formal_full_score": float(source["formal_full_score"]),
                "formal_sharded_score": float(source["formal_sharded_score"]),
                "context_token_count": baseline_context_count,
                "assistant_history_token_count": assistant_history_count,
                "status": "completed",
            }
            append_jsonl(args.output_dir / "server_native_baselines.jsonl", baseline_row)
            append_jsonl(
                args.output_dir / "formal_heldout_payloads.jsonl",
                {
                    "trajectory_id": trajectory_id,
                    "source_instance_id": source["instance_id"],
                    "task": source["task"],
                    "population_version": source["population_version"],
                    "source_shard": source["source_shard"],
                    "original_messages": baseline_history,
                },
            )
            assistants = [row["content"] for row in baseline_history if row["role"] == "assistant"]
            for target_turn in range(len(users) - 1):
                surgery_history, surgery_calls = generate_surgery(system, users, baseline_history, target_turn, engine)
                surgery_output = surgery_history[-1]["content"]
                _, evaluator, surgery_score = extract_and_evaluate(source["task"], surgery_output, sample)
                delta = surgery_score - baseline_score
                branch = {
                    "trajectory_id": trajectory_id,
                    "target_turn": target_turn,
                    "task": source["task"],
                    "population_version": source["population_version"],
                    "source_shard": source["source_shard"],
                    "local_effect_class": effect_class(delta),
                    "local_delta": delta,
                    "effect_class": effect_class(delta),
                    "delta": delta,
                    "formal_effect_class": "not_applicable",
                    "formal_delta": "",
                    "original_score": baseline_score,
                    "surgery_score": surgery_score,
                    "n_user_turns": len(users),
                    "n_downstream_user_anchors": len(users) - target_turn - 1,
                    "relative_position": target_turn / (len(users) - 1),
                    "distance_to_final": len(users) - 1 - target_turn,
                    "assistant_target_token_count": content_tokens(engine, assistants[target_turn]),
                    "assistant_history_token_count": assistant_history_count,
                    "original_context_token_count": baseline_context_count,
                    "surgery_context_token_count": context_tokens(engine, surgery_history[:-1]),
                    "formal_full_score": source["formal_full_score"],
                    "formal_sharded_score": source["formal_sharded_score"],
                }
                append_csv(args.output_dir / "formal_heldout_branch_manifest.csv", branch)
                append_jsonl(
                    args.output_dir / "server_native_surgeries.jsonl",
                    {
                        **branch,
                        "original_score_local": baseline_score,
                        "surgery_score_local": surgery_score,
                        "delta_local": delta,
                        "effect_class_local": effect_class(delta),
                        "original_output_local": baseline_output,
                        "surgery_output_local": surgery_output,
                        "generated_history": baseline_history,
                        "surgery_history": surgery_history,
                        "generation_calls": surgery_calls,
                        "evaluator_output": evaluator,
                        "status": "completed",
                    },
                )
            print(json.dumps({"trajectory_id": trajectory_id, "branches": len(users) - 1}), flush=True)
        except Exception as error:
            append_jsonl(
                args.output_dir / "formal_heldout_failures.jsonl",
                {
                    "trajectory_id": trajectory_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()

