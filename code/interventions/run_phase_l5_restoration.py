"""Execute the frozen Phase L5 Original-Turn Restoration manifest.

The runner reuses the P5/RQ2 replay stack. It never regenerates SHORT_ACK or
intermediate assistant turns: all non-target intermediate slots are the frozen
neutral phrase, exactly one target slot contains canonical original text, and
only the final assistant response is generated and evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from p5_counterfactual_runner import (  # noqa: E402
    ACK,
    DATA,
    append_jsonl,
    call_assistant,
    evaluate,
    read_jsonl,
    user_turns,
)


def load_source_exact(path: str, instruction_id: str, model: str, run_id: str) -> dict:
    matches = [
        row for row in read_jsonl(Path(path))
        if row.get("task_id") == instruction_id
        and (row.get("model_alias") or row.get("assistant_model")) == model
        and row.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact frozen source trajectory, found {len(matches)}")
    return matches[0]


def restoration_branch(source: dict, row: dict) -> tuple[list[dict], list[dict]]:
    system, users, assistants = user_turns(source)
    target = int(row["target_turn"])
    n_intermediate = len(users) - 1
    if n_intermediate != int(row["n_intermediate_turns"]):
        raise ValueError("Manifest/source intermediate-turn count mismatch")
    if target < 0 or target >= n_intermediate:
        raise ValueError("Restoration target is not a non-final assistant turn")
    if assistants[target] != row["target_assistant_text"]:
        raise ValueError("Manifest target text is not exact canonical original A_t")

    history = [dict(system)]
    for index, user in enumerate(users[:-1]):
        history.append({"role": "user", "content": user})
        text = assistants[index] if index == target else ACK
        history.append({"role": "assistant", "content": text})
    history.append({"role": "user", "content": users[-1]})

    # Intervention-semantic assertions occur before the only model call.
    replay_users = [m["content"] for m in history if m["role"] == "user"]
    replay_assistants = [m["content"] for m in history if m["role"] == "assistant"]
    if replay_users != users:
        raise ValueError("Frozen realized user trajectory changed")
    if len(replay_assistants) != n_intermediate:
        raise ValueError("Restoration context has wrong number of intermediate slots")
    for index, text in enumerate(replay_assistants):
        expected = assistants[target] if index == target else ACK
        if text != expected:
            raise ValueError("Restoration context violates one-original/all-other-neutral rule")

    result = call_assistant(history, row["model"])
    history.append({"role": "assistant", "content": result["message"]})
    generated = [{
        "turn_index": len(users) - 1,
        "assistant": result["message"],
        "returned_model": result.get("returned_model"),
        "response_id": result.get("response_id"),
    }]
    return history, generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--exclude-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.validation_only and args.exclude_validation:
        raise ValueError("Choose at most one validation filter")

    with open(args.manifest, encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if args.validation_only:
        manifest = [row for row in manifest if row["validation_sample"].lower() == "true"]
    if args.exclude_validation:
        manifest = [row for row in manifest if row["validation_sample"].lower() != "true"]
    manifest = [
        row for index, row in enumerate(manifest)
        if index % args.worker_count == args.worker_index
    ]
    output = Path(args.output)
    existing = read_jsonl(output) if output.exists() else []
    completed = {row["branch_id"] for row in existing if row.get("status") == "completed"}
    attempted = {row["branch_id"] for row in existing}
    if args.dry_run:
        print(json.dumps({
            "planned_worker": len(manifest),
            "completed": len(completed),
            "validation_only": args.validation_only,
            "exclude_validation": args.exclude_validation,
            "ack": ACK,
        }))
        return

    samples = {row["task_id"]: row for row in json.loads(DATA.read_text(encoding="utf-8"))}
    os.environ["P1_CANONICAL_LOG"] = str(output.with_suffix(".api.jsonl"))
    os.environ["P1_MAX_ATTEMPTS"] = "3"
    os.environ["P1_PROVIDER_ALIAS"] = "configured_provider"
    os.environ["P1_SUMMARY_EVALUATOR_MODEL"] = "gpt-4o"

    for row in manifest:
        branch_id = row["restoration_branch_id"]
        if branch_id in completed or (branch_id in attempted and not args.retry_failed):
            continue
        os.environ["P1_RUN_CONTEXT"] = json.dumps({
            "experiment_id": "phase_l5_original_turn_restoration_v1",
            "run_id": branch_id,
            "case_id": row["source_instance_id"],
            "task": row["task"],
            "mode": "original_turn_restoration",
            "assistant_model": row["model"],
            "call_role": "ASSISTANT_UNDER_TEST",
        })
        try:
            source = load_source_exact(
                row["raw_sharded_path"], row["source_instance_id"], row["model"],
                row["raw_sharded_run_id"],
            )
            history, generated = restoration_branch(source, row)
            answer, evaluator, score = evaluate(
                history, row["task"], row["model"], samples[row["source_instance_id"]]
            )
            short_score = float(row["short_ack_score"])
            record = {
                "branch_id": branch_id,
                "trajectory_id": row["trajectory_id"],
                "target_turn": int(row["target_turn"]),
                "source_instance_id": row["source_instance_id"],
                "model": row["model"],
                "task": row["task"],
                "cohort": row["cohort"],
                "sample_role": row["sample_role"],
                "match_set_id": row["match_set_id"],
                "mode": "original_turn_restoration",
                "neutral_phrase": ACK,
                "fixed_user_sequence": True,
                "future_user_simulator_calls": 0,
                "only_final_assistant_generated": True,
                "source_raw_path": row["raw_sharded_path"],
                "source_run_id": source.get("run_id"),
                "source_trace_hash": hashlib.sha256(
                    json.dumps(source.get("trace", []), sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                "counterfactual_history": history,
                "generated_branch": generated,
                "extracted_answer": answer,
                "evaluator_output": evaluator,
                "restoration_score": score,
                "short_ack_score": short_score,
                "restoration_delta": score - short_score,
                "status": "completed",
            }
        except Exception as error:
            record = {
                "branch_id": branch_id,
                "trajectory_id": row.get("trajectory_id"),
                "target_turn": int(row["target_turn"]),
                "cohort": row.get("cohort"),
                "mode": "original_turn_restoration",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        append_jsonl(output, record)
        print(json.dumps({"branch_id": branch_id, "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
