"""Validate one completed FORMAL_HELDOUT_118 behavior shard."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "positive"
    if delta < -tolerance:
        return "negative"
    return "zero"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = read_csv(args.source_manifest)
    payloads = {row["trajectory_id"]: row for row in read_jsonl(args.source_payload)}
    baselines = read_jsonl(args.behavior_dir / "server_native_baselines.jsonl")
    surgeries = read_jsonl(args.behavior_dir / "server_native_surgeries.jsonl")
    branches = read_csv(args.behavior_dir / "formal_heldout_branch_manifest.csv")
    failures = read_jsonl(args.behavior_dir / "formal_heldout_failures.jsonl")

    expected_trajectories = {row["trajectory_id"] for row in source}
    expected_keys = {
        (row["trajectory_id"], target)
        for row in source
        for target in range(int(row["eligible_target_turn_count"]))
    }
    baseline_by_id = {row["trajectory_id"]: row for row in baselines}
    surgery_by_key = {(row["trajectory_id"], int(row["target_turn"])): row for row in surgeries}
    branch_keys = {(row["trajectory_id"], int(row["target_turn"])) for row in branches}
    errors = []
    if failures:
        errors.append(f"failure_rows={len(failures)}")
    if len(baseline_by_id) != len(baselines):
        errors.append("duplicate_baseline_trajectory")
    if len(surgery_by_key) != len(surgeries) or len(branch_keys) != len(branches):
        errors.append("duplicate_branch_key")
    if set(baseline_by_id) != expected_trajectories:
        errors.append("baseline_trajectory_set_mismatch")
    if set(surgery_by_key) != expected_keys or branch_keys != expected_keys:
        errors.append("branch_key_set_mismatch")

    for trajectory_id, baseline in baseline_by_id.items():
        expected = payloads[trajectory_id]
        systems = [row["content"] for row in baseline["generated_history"] if row["role"] == "system"]
        users = [row["content"] for row in baseline["generated_history"] if row["role"] == "user"]
        if systems != [expected["system_message"]] or users != expected["user_messages"]:
            errors.append(f"recovered_user_sequence_mismatch:{trajectory_id}")
        if baseline.get("status") != "completed" or not isinstance(baseline.get("evaluator_output"), dict):
            errors.append(f"baseline_evaluator_invalid:{trajectory_id}")

    for key, surgery in surgery_by_key.items():
        baseline = baseline_by_id[key[0]]
        expected = payloads[key[0]]
        users = [row["content"] for row in surgery["surgery_history"] if row["role"] == "user"]
        if users != expected["user_messages"]:
            errors.append(f"surgery_user_sequence_mismatch:{key}")
        baseline_score = float(baseline["original_score_local"])
        surgery_score = float(surgery["surgery_score_local"])
        delta = surgery_score - baseline_score
        if abs(float(surgery["original_score_local"]) - baseline_score) > 1e-12:
            errors.append(f"wrong_baseline_score:{key}")
        if abs(float(surgery["delta_local"]) - delta) > 1e-12:
            errors.append(f"wrong_delta:{key}")
        if surgery["effect_class_local"] != classify(delta):
            errors.append(f"wrong_effect_class:{key}")
        if surgery.get("status") != "completed" or not isinstance(surgery.get("evaluator_output"), dict):
            errors.append(f"surgery_evaluator_invalid:{key}")

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "source_shard": source[0]["source_shard"],
        "expected_trajectories": len(expected_trajectories),
        "actual_trajectories": len(baseline_by_id),
        "expected_branches": len(expected_keys),
        "actual_branches": len(surgery_by_key),
        "tasks": dict(Counter(row["task"] for row in source)),
        "effect_classes": dict(Counter(row["effect_class_local"] for row in surgeries)),
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

