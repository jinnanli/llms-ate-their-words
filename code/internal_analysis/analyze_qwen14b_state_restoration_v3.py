"""Analyze complete positive or signed-negative v3 restoration outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


COARSE_LAYERS = (0, 6, 11, 17, 22, 28, 33, 39)
DEPTHS = {"early": (0, 6), "mid": (11, 17, 22, 28), "late": (33, 39)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def bootstrap_pair_differences(
    pairs: dict[str, tuple[float, float]], replicates: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    trajectories = sorted(pairs)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(trajectories) for _ in trajectories]
        estimates.append(sum(pairs[trajectory][0] - pairs[trajectory][1] for trajectory in sampled) / len(sampled))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def summarize(rows: list[dict], metric: str, replicates: int, seed: int) -> dict:
    by_trajectory: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_trajectory[row["trajectory_id"]][row["sample_role"]] = float(row[metric])
    roles = {row["sample_role"] for row in rows}
    treatment_role = next(role for role in roles if role != "matched_zero")
    pairs = {
        trajectory: (values[treatment_role], values["matched_zero"])
        for trajectory, values in by_trajectory.items()
        if treatment_role in values and "matched_zero" in values
    }
    if len(pairs) != len(by_trajectory):
        raise RuntimeError("Incomplete treatment/control pairing")
    treatment = [value[0] for value in pairs.values()]
    zero = [value[1] for value in pairs.values()]
    differences = [a - b for a, b in pairs.values()]
    low, high = bootstrap_pair_differences(pairs, replicates, seed)
    return {
        "treatment_role": treatment_role,
        "trajectory_pairs": len(pairs),
        "treatment_mean": sum(treatment) / len(treatment),
        "treatment_median": quantile(treatment, 0.5),
        "zero_mean": sum(zero) / len(zero),
        "zero_median": quantile(zero, 0.5),
        "difference": sum(differences) / len(differences),
        "paired_median_difference": quantile(differences, 0.5),
        "ci_low": low,
        "ci_high": high,
        "fraction_treatment_positive": sum(value > 0 for value in treatment) / len(treatment),
        "fraction_zero_positive": sum(value > 0 for value in zero) / len(zero),
    }


def macro_summary(rows: list[dict], metric: str, replicates: int, seed: int) -> dict:
    by_task = {task: [row for row in rows if row["task"] == task] for task in ("actions", "code", "data2text")}
    point = {task: summarize(task_rows, metric, 100, seed) for task, task_rows in by_task.items()}
    rng = random.Random(seed)
    estimates = []
    task_pairs = {}
    for task, task_rows in by_task.items():
        by_trajectory = defaultdict(dict)
        for row in task_rows:
            by_trajectory[row["trajectory_id"]][row["sample_role"]] = float(row[metric])
        treatment_role = next(role for role in {row["sample_role"] for row in task_rows} if role != "matched_zero")
        task_pairs[task] = {
            trajectory: (values[treatment_role], values["matched_zero"])
            for trajectory, values in by_trajectory.items()
        }
    for _ in range(replicates):
        task_differences = []
        for task in ("actions", "code", "data2text"):
            trajectories = sorted(task_pairs[task])
            sampled = [rng.choice(trajectories) for _ in trajectories]
            task_differences.append(
                sum(task_pairs[task][trajectory][0] - task_pairs[task][trajectory][1] for trajectory in sampled)
                / len(sampled)
            )
        estimates.append(sum(task_differences) / 3)
    return {
        "trajectory_pairs": sum(value["trajectory_pairs"] for value in point.values()),
        "treatment_mean": sum(value["treatment_mean"] for value in point.values()) / 3,
        "zero_mean": sum(value["zero_mean"] for value in point.values()) / 3,
        "difference": sum(value["difference"] for value in point.values()) / 3,
        "ci_low": quantile(estimates, 0.025),
        "ci_high": quantile(estimates, 0.975),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_complete(manifest: list[dict[str, str]], results: list[dict], mode: str) -> dict:
    directions = ("rescue", "reversion") if mode == "positive" else ("harm_transfer", "rescue_reversal")
    expected = {
        (row["trajectory_id"], int(row["target_turn"]), row["sample_role"], direction, layer)
        for row in manifest for direction in directions for layer in COARSE_LAYERS
    }
    actual = [
        (row["trajectory_id"], int(row["target_turn"]), row["sample_role"], row["patch_direction"], int(row["layer"]))
        for row in results
    ]
    invalid = []
    for index, row in enumerate(results):
        for field in ("original_score", "surgery_score", "patched_score", "local_delta"):
            if not math.isfinite(float(row[field])):
                invalid.append((index, field))
        if not isinstance(row.get("evaluator"), dict) or row.get("metadata", {}).get("patch_apply_count") != 1:
            invalid.append((index, "structure"))
    return {
        "expected_rows": len(expected), "actual_rows": len(actual),
        "missing": len(expected - set(actual)), "unexpected": len(set(actual) - expected),
        "duplicates": sum(count - 1 for count in Counter(actual).values() if count > 1),
        "invalid": len(invalid),
        "status": "PASS" if len(actual) == len(expected) and set(actual) == expected and len(set(actual)) == len(actual) and not invalid else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--mode", choices=["positive", "negative"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest)
    results = [row for path in args.result for row in read_jsonl(path)]
    integrity = validate_complete(manifest, results, args.mode)
    (args.output_dir / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Incomplete restoration population: {integrity}")

    metric_by_direction = (
        {"rescue": "rescue_effect", "reversion": "reversion_effect"}
        if args.mode == "positive"
        else {"harm_transfer": "harm_transfer_effect", "rescue_reversal": "rescue_reversal_effect"}
    )
    summaries = []
    for direction, metric in metric_by_direction.items():
        direction_rows = [row for row in results if row["patch_direction"] == direction]
        for layer in COARSE_LAYERS:
            layer_rows = [row for row in direction_rows if int(row["layer"]) == layer]
            for scope in ("pooled", "actions", "code", "data2text"):
                scope_rows = layer_rows if scope == "pooled" else [row for row in layer_rows if row["task"] == scope]
                summary = summarize(scope_rows, metric, args.bootstrap, args.seed + layer)
                summaries.append({"direction": direction, "depth": "layer", "layer": layer, "scope": scope, **summary})
            summaries.append({
                "direction": direction, "depth": "layer", "layer": layer, "scope": "task_macro",
                **macro_summary(layer_rows, metric, args.bootstrap, args.seed + layer),
            })
        for depth, layers in DEPTHS.items():
            branch_values = defaultdict(list)
            branch_metadata = {}
            for row in direction_rows:
                if int(row["layer"]) in layers:
                    branch_key = (row["trajectory_id"], int(row["target_turn"]), row["sample_role"])
                    branch_values[branch_key].append(float(row[metric]))
                    branch_metadata[branch_key] = row
            depth_rows = []
            for branch_key, values in branch_values.items():
                row = dict(branch_metadata[branch_key])
                row[metric] = sum(values) / len(values)
                depth_rows.append(row)
            for scope in ("pooled", "actions", "code", "data2text"):
                scope_rows = depth_rows if scope == "pooled" else [row for row in depth_rows if row["task"] == scope]
                summary = summarize(scope_rows, metric, args.bootstrap, args.seed + len(depth))
                summaries.append({"direction": direction, "depth": depth, "layer": "", "scope": scope, **summary})
            summaries.append({
                "direction": direction, "depth": depth, "layer": "", "scope": "task_macro",
                **macro_summary(depth_rows, metric, args.bootstrap, args.seed + len(depth)),
            })

    flip_rows = []
    for direction, metric in metric_by_direction.items():
        reference = "original_score" if direction in {"rescue", "harm_transfer"} else "surgery_score"
        for layer in COARSE_LAYERS:
            for task in ("actions", "code"):
                for role in sorted({row["sample_role"] for row in results}):
                    selected = [
                        row for row in results
                        if row["patch_direction"] == direction and int(row["layer"]) == layer
                        and row["task"] == task and row["sample_role"] == role
                    ]
                    flip_rows.append({
                        "direction": direction, "layer": layer, "task": task, "sample_role": role,
                        "n": len(selected),
                        "flip_rate": sum(float(row["patched_score"]) != float(row[reference]) for row in selected) / len(selected),
                    })
    write_csv(args.output_dir / "layer_and_depth_results.csv", summaries)
    write_csv(args.output_dir / "binary_flip_rates.csv", flip_rows)
    (args.output_dir / "analysis_config.json").write_text(json.dumps({
        "mode": args.mode, "layers": COARSE_LAYERS, "depths": DEPTHS,
        "bootstrap_replicates": args.bootstrap, "cluster": "trajectory_id",
        "task_macro": "equal_weight_three_tasks",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()

