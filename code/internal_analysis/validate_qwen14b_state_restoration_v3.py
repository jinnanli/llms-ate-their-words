"""Validate v3 sham and coarse restoration outputs against the frozen pair manifest."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


LAYERS = (0, 6, 11, 17, 22, 28, 33, 39)
DIRECTIONS = ("rescue", "reversion")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_sham(path: Path) -> dict:
    rows = read_jsonl(path)
    expected_keys = {
        (task, context, layer)
        for task in ("actions", "code", "data2text")
        for context in ("original", "surgery")
        for layer in LAYERS
    }
    keys = [(row["task"], row["context"], int(row["layer"])) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    actual_keys = set(keys)
    invalid = []
    for index, row in enumerate(rows):
        if row.get("exact_output_match") is not True or row.get("exact_score_match") is not True:
            invalid.append({"row": index, "reason": "invariance_mismatch"})
        if not finite(row.get("expected_score")) or not finite(row.get("patched_score")):
            invalid.append({"row": index, "reason": "nonfinite_score"})
        if not isinstance(row.get("evaluator"), dict):
            invalid.append({"row": index, "reason": "missing_evaluator"})
        if row.get("metadata", {}).get("patch_apply_count") != 1:
            invalid.append({"row": index, "reason": "patch_apply_count"})
    return {
        "expected_rows": len(expected_keys),
        "actual_rows": len(rows),
        "missing_keys": sorted(expected_keys - actual_keys),
        "unexpected_keys": sorted(actual_keys - expected_keys),
        "duplicate_keys": duplicates,
        "invalid_rows": invalid,
        "status": "PASS" if len(rows) == len(expected_keys) and not duplicates and actual_keys == expected_keys and not invalid else "FAIL",
    }


def validate_shard(pairs_path: Path, results_path: Path, shard_index: int, shard_count: int) -> dict:
    pair_rows = read_csv(pairs_path)
    pair_ids = sorted({row["pair_id"] for row in pair_rows})
    selected = {pair_id for index, pair_id in enumerate(pair_ids) if index % shard_count == shard_index}
    selected_rows = [row for row in pair_rows if row["pair_id"] in selected]
    expected_keys = {
        (
            row["pair_id"], row["trajectory_id"], int(row["target_turn"]),
            row["sample_role"], direction, layer,
        )
        for row in selected_rows
        for direction in DIRECTIONS
        for layer in LAYERS
    }
    rows = read_jsonl(results_path)
    keys = [
        (
            row["pair_id"], row["trajectory_id"], int(row["target_turn"]),
            row["sample_role"], row["patch_direction"], int(row["layer"]),
        )
        for row in rows
    ]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    actual_keys = set(keys)
    invalid = []
    for index, row in enumerate(rows):
        for field in ("original_score", "surgery_score", "patched_score", "local_delta"):
            if not finite(row.get(field)):
                invalid.append({"row": index, "reason": f"nonfinite_{field}"})
        if not isinstance(row.get("generated_output"), str):
            invalid.append({"row": index, "reason": "missing_generated_output"})
        if not isinstance(row.get("evaluator"), dict):
            invalid.append({"row": index, "reason": "missing_evaluator"})
        if row.get("metadata", {}).get("patch_apply_count") != 1:
            invalid.append({"row": index, "reason": "patch_apply_count"})
    return {
        "shard_index": shard_index,
        "pairs": len(selected),
        "branches": len(selected_rows),
        "expected_rows": len(expected_keys),
        "actual_rows": len(rows),
        "missing_key_count": len(expected_keys - actual_keys),
        "unexpected_key_count": len(actual_keys - expected_keys),
        "duplicate_keys": duplicates,
        "invalid_rows": invalid,
        "status": "PASS" if len(rows) == len(expected_keys) and not duplicates and actual_keys == expected_keys and not invalid else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, action="append", default=[])
    args = parser.parse_args()
    report = {
        "sham": validate_sham(args.result_root / "sham_L40" / "sham_validation.jsonl"),
        "shards": {},
    }
    locations = {0: "L40_shard_00", 1: "gpujl_shard_01", 2: "L40_shard_02", 3: "gpujl_shard_03"}
    for shard in args.shard:
        report["shards"][str(shard)] = validate_shard(
            args.pairs,
            args.result_root / locations[shard] / "state_restoration_results.jsonl",
            shard,
            4,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["sham"]["status"] != "PASS" or any(row["status"] != "PASS" for row in report["shards"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

