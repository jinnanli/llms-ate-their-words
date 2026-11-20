"""Frozen history counterfactual runner for P5 and the focused P7-R control.

This runner never invokes the user simulator. It replays user messages from a
canonical SHARDED trace and evaluates only with the upstream task evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))
from model_openai import generate
from system_agent import SystemAgent
from tasks import get_task

ACK = "Acknowledged."
DATA = ROOT / "data" / "sharded_instructions_600.json"
load_dotenv()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def semantic_messages(conversation: dict):
    return [
        {"role": row["role"], "content": row["content"]}
        for row in conversation["trace"]
        if row.get("role") in {"system", "user", "assistant"}
    ]


def user_turns(conversation: dict):
    messages = semantic_messages(conversation)
    system = next(row for row in messages if row["role"] == "system")
    users = [row["content"] for row in messages if row["role"] == "user"]
    assistants = [row["content"] for row in messages if row["role"] == "assistant"]
    if len(users) != len(assistants):
        raise ValueError("Frozen SHARDED trace lacks one assistant response per user turn")
    return system, users, assistants


def returned_family_ok(requested: str, returned: str | None):
    if not returned:
        return True
    if requested == "gpt-4o":
        return returned.startswith("gpt-4o")
    if requested == "gpt-5.6-luna":
        return returned.startswith("gpt-5.6-luna")
    return requested == returned


def call_assistant(history, model):
    result = generate(
        [dict(row) for row in history],
        model=model,
        timeout=60,
        max_retries=1,
        temperature=0,
        max_tokens=1000,
        return_metadata=True,
        call_role="ASSISTANT_UNDER_TEST",
    )
    if not returned_family_ok(model, result.get("returned_model")):
        raise RuntimeError(
            f"returned-model family conflict: {model} -> {result.get('returned_model')}"
        )
    return result


def neutral_ack_branch(conversation: dict, model: str):
    system, users, _ = user_turns(conversation)
    history = [system]
    for user in users[:-1]:
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": ACK}])
    history.append({"role": "user", "content": users[-1]})
    result = call_assistant(history, model)
    history.append({"role": "assistant", "content": result["message"]})
    return history, [{"turn_index": len(users) - 1, "assistant": result["message"], "returned_model": result.get("returned_model"), "response_id": result.get("response_id")}]


def uniform_length_estimate(text: str) -> int:
    """Frozen model-agnostic whitespace-token estimate used by P7-R."""
    return len(re.findall(r"\S+", text or ""))


def length_matched_ack(original: str) -> tuple[str, dict]:
    target = uniform_length_estimate(original)
    # Canonical traces contain non-empty assistant responses. Keep a valid
    # acknowledgement for the defensive zero-length case and disclose it.
    count = max(1, target)
    filler = " ".join([ACK] * count)
    actual = uniform_length_estimate(filler)
    return filler, {
        "estimator": "unicode_whitespace_token_count_v1",
        "original_estimated_tokens": target,
        "replacement_estimated_tokens": actual,
        "length_ratio": (actual / target) if target else None,
    }


def length_matched_ack_branch(conversation: dict, model: str):
    system, users, assistants = user_turns(conversation)
    history = [system]
    slots = []
    for index, user in enumerate(users[:-1]):
        filler, metadata = length_matched_ack(assistants[index])
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": filler}])
        slots.append({"turn_index": index, **metadata})
    history.append({"role": "user", "content": users[-1]})
    result = call_assistant(history, model)
    history.append({"role": "assistant", "content": result["message"]})
    branch = [{"turn_index": len(users) - 1, "assistant": result["message"], "returned_model": result.get("returned_model"), "response_id": result.get("response_id")}]
    return history, branch, slots


def surgery_branch(conversation: dict, model: str, surgery_turn: int):
    system, users, assistants = user_turns(conversation)
    if surgery_turn < 0 or surgery_turn >= len(users) - 1:
        raise ValueError("Surgery turn must be a non-final assistant turn")
    history = [system]
    branch = []
    for index, user in enumerate(users):
        history.append({"role": "user", "content": user})
        if index < surgery_turn:
            assistant = assistants[index]
            history.append({"role": "assistant", "content": assistant})
        elif index == surgery_turn:
            history.append({"role": "assistant", "content": ACK})
        else:
            result = call_assistant(history, model)
            history.append({"role": "assistant", "content": result["message"]})
            branch.append({"turn_index": index, "assistant": result["message"], "returned_model": result.get("returned_model"), "response_id": result.get("response_id")})
    return history, branch


def evaluate(history, task: str, model: str, sample: dict):
    agent = SystemAgent(task, "gpt-5.6-luna", sample)
    answer = agent.extract_answer(history)
    result = get_task(task).evaluator_function(answer, sample)
    score = result.get("score")
    if score is None and isinstance(result.get("is_correct"), bool):
        score = float(result["is_correct"])
    if score is None:
        raise ValueError("Task evaluator did not return a score")
    return answer, result, float(score)


def load_source(path: str, instruction_id: str, model: str):
    matches = [
        row
        for row in read_jsonl(Path(path))
        if row.get("task_id") == instruction_id
        and (row.get("model_alias") or row.get("assistant_model")) == model
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen source trajectory, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["neutral_ack", "length_matched_ack", "surgery"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    manifest = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    samples = {row["task_id"]: row for row in json.loads(DATA.read_text(encoding="utf-8"))}
    output = Path(args.output)
    completed = {
        row["branch_id"]
        for row in read_jsonl(output)
        if row.get("status") == "completed"
    }
    attempted = {row["branch_id"] for row in read_jsonl(output)}
    planned = []
    for row in manifest:
        turns = [-1] if args.mode in {"neutral_ack", "length_matched_ack"} else [int(row["surgery_turn"])]
        for turn in turns:
            key = f"{args.mode}|{row['trajectory_id']}|{turn}|{row.get('repeat_index','0')}"
            planned.append((hashlib.sha256(key.encode()).hexdigest()[:24], row, turn))
    planned = [row for index, row in enumerate(planned) if index % args.worker_count == args.worker_index]
    if args.dry_run:
        print(json.dumps({"planned_worker": len(planned), "completed": len(completed), "ack": ACK}))
        return
    os.environ["P1_CANONICAL_LOG"] = str(output.with_suffix(".api.jsonl"))
    os.environ["P1_MAX_ATTEMPTS"] = "3"
    os.environ["P1_PROVIDER_ALIAS"] = "configured_provider"
    os.environ["P1_SUMMARY_EVALUATOR_MODEL"] = "gpt-4o"
    for branch_id, row, turn in planned:
        if branch_id in completed or (branch_id in attempted and not args.retry_failed):
            continue
        experiment_id = "P7R_length_matched_ack_v1" if args.mode == "length_matched_ack" else "P5_counterfactual_v1"
        os.environ["P1_RUN_CONTEXT"] = json.dumps({"experiment_id": experiment_id, "run_id": branch_id, "case_id": row["instruction_id"], "task": row["task"], "mode": args.mode, "assistant_model": row["tested_model"], "call_role": "ASSISTANT_UNDER_TEST"})
        try:
            source = load_source(row["raw_sharded_path"], row["instruction_id"], row["tested_model"])
            if args.mode == "neutral_ack":
                history, branch = neutral_ack_branch(source, row["tested_model"])
                length_slots = None
            elif args.mode == "length_matched_ack":
                history, branch, length_slots = length_matched_ack_branch(source, row["tested_model"])
            else:
                history, branch = surgery_branch(source, row["tested_model"], turn)
                length_slots = None
            answer, evaluator, score = evaluate(history, row["task"], row["tested_model"], samples[row["instruction_id"]])
            record = {"branch_id": branch_id, "trajectory_id": row["trajectory_id"], "instruction_id": row["instruction_id"], "task": row["task"], "tested_model": row["tested_model"], "mode": args.mode, "surgery_turn": None if turn < 0 else turn, "ack_text": ACK, "length_estimator": "unicode_whitespace_token_count_v1" if length_slots is not None else None, "length_slots": length_slots, "fixed_user_sequence": True, "future_user_simulator_calls": 0, "source_raw_path": row["raw_sharded_path"], "source_run_id": source.get("run_id"), "source_config_hash": source.get("config_hash"), "source_trace_hash": hashlib.sha256(json.dumps(source.get("trace", []), sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "counterfactual_history": history, "branch": branch, "extracted_answer": answer, "evaluator_output": evaluator, "final_score": score, "original_sharded_score": float(row["SHARDED_score"]), "effect": score - float(row["SHARDED_score"]), "repeat_index": int(row.get("repeat_index", 0)), "status": "completed"}
        except Exception as error:
            record = {"branch_id": branch_id, "trajectory_id": row.get("trajectory_id"), "mode": args.mode, "surgery_turn": None if turn < 0 else turn, "status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        append_jsonl(output, record)
        print(json.dumps({"branch_id": branch_id, "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
