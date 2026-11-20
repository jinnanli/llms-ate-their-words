"""Materialize and validate the complete FORMAL_HELDOUT_118 behavior ledger."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "positive"
    if delta < -tolerance:
        return "negative"
    return "zero"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")

    source_manifests = sorted((args.source_root / "behavior_shards").glob("heldout_shard_*.csv"))
    source_payloads = sorted((args.source_root / "behavior_shards").glob("heldout_shard_*.jsonl"))
    if len(source_manifests) != 8 or len(source_payloads) != 8:
        raise RuntimeError("Expected eight source shard pairs")
    expected_rows = [row for path in source_manifests for row in read_csv(path)]
    expected_payloads = [row for path in source_payloads for row in read_jsonl(path)]
    expected_by_id = {row["trajectory_id"]: row for row in expected_rows}
    payload_by_id = {row["trajectory_id"]: row for row in expected_payloads}

    behavior_dirs = sorted({path.parent for path in args.result_root.glob("L40_*/behavior/heldout_shard_*/server_native_baselines.jsonl")})
    if len(behavior_dirs) != 8:
        raise RuntimeError(f"Expected eight completed behavior directories, found {len(behavior_dirs)}")
    baselines = [row for directory in behavior_dirs for row in read_jsonl(directory / "server_native_baselines.jsonl")]
    surgeries = [row for directory in behavior_dirs for row in read_jsonl(directory / "server_native_surgeries.jsonl")]
    branches = [row for directory in behavior_dirs for row in read_csv(directory / "formal_heldout_branch_manifest.csv")]
    native_payloads = [row for directory in behavior_dirs for row in read_jsonl(directory / "formal_heldout_payloads.jsonl")]
    failures = [
        row
        for directory in behavior_dirs
        if (directory / "formal_heldout_failures.jsonl").exists()
        for row in read_jsonl(directory / "formal_heldout_failures.jsonl")
    ]
    if failures:
        raise RuntimeError(f"Behavior failures exist: {len(failures)}")

    baseline_by_id = {row["trajectory_id"]: row for row in baselines}
    surgery_by_key = {(row["trajectory_id"], int(row["target_turn"])): row for row in surgeries}
    branch_by_key = {(row["trajectory_id"], int(row["target_turn"])): row for row in branches}
    expected_keys = {
        (row["trajectory_id"], target)
        for row in expected_rows
        for target in range(int(row["eligible_target_turn_count"]))
    }
    if len(baseline_by_id) != len(baselines) or set(baseline_by_id) != set(expected_by_id):
        raise RuntimeError("Baseline coverage or uniqueness failed")
    if len(surgery_by_key) != len(surgeries) or set(surgery_by_key) != expected_keys:
        raise RuntimeError("Surgery coverage or uniqueness failed")
    if len(branch_by_key) != len(branches) or set(branch_by_key) != expected_keys:
        raise RuntimeError("Branch manifest coverage or uniqueness failed")
    if len({row["trajectory_id"] for row in native_payloads}) != 118:
        raise RuntimeError("Native payload coverage failed")

    for trajectory_id, baseline in baseline_by_id.items():
        payload = payload_by_id[trajectory_id]
        systems = [row["content"] for row in baseline["generated_history"] if row["role"] == "system"]
        users = [row["content"] for row in baseline["generated_history"] if row["role"] == "user"]
        if systems != [payload["system_message"]] or users != payload["user_messages"]:
            raise RuntimeError(f"Recovered message mismatch: {trajectory_id}")
    for key, surgery in surgery_by_key.items():
        baseline_score = float(baseline_by_id[key[0]]["original_score_local"])
        delta = float(surgery["surgery_score_local"]) - baseline_score
        if abs(float(surgery["delta_local"]) - delta) > 1e-12 or surgery["effect_class_local"] != classify(delta):
            raise RuntimeError(f"Delta/effect mismatch: {key}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "server_native_baselines.jsonl", sorted(baselines, key=lambda row: row["trajectory_id"]))
    write_jsonl(
        args.output_dir / "server_native_surgeries.jsonl",
        sorted(surgeries, key=lambda row: (row["trajectory_id"], int(row["target_turn"]))),
    )
    write_csv(
        args.output_dir / "formal_heldout_branch_manifest.csv",
        sorted(branches, key=lambda row: (row["trajectory_id"], int(row["target_turn"]))),
    )
    write_jsonl(args.output_dir / "formal_heldout_payloads.jsonl", sorted(native_payloads, key=lambda row: row["trajectory_id"]))

    by_trajectory = defaultdict(set)
    for row in surgeries:
        by_trajectory[row["trajectory_id"]].add(row["effect_class_local"])
    task_classes = {
        task: dict(Counter(row["effect_class_local"] for row in surgeries if row["task"] == task))
        for task in ("actions", "code", "data2text")
    }
    summary = {
        "status": "PASS",
        "population_version": "FORMAL_HELDOUT_118",
        "trajectories": len(baselines),
        "branches": len(surgeries),
        "task_trajectories": dict(Counter(row["task"] for row in baselines)),
        "task_branches": dict(Counter(row["task"] for row in surgeries)),
        "effect_classes": dict(Counter(row["effect_class_local"] for row in surgeries)),
        "task_effect_classes": task_classes,
        "trajectories_with_positive": sum("positive" in classes for classes in by_trajectory.values()),
        "trajectories_with_zero": sum("zero" in classes for classes in by_trajectory.values()),
        "mixed_positive_zero_trajectories": sum({"positive", "zero"}.issubset(classes) for classes in by_trajectory.values()),
        "trajectories_with_negative": sum("negative" in classes for classes in by_trajectory.values()),
    }
    (args.output_dir / "behavior_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

