"""Compare predefined coarse State Propagation and restoration depth profiles."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


LAYERS = (0, 6, 11, 17, 22, 28, 33, 39)
SCOPES = ("pooled", "actions", "code", "data2text", "task_macro")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1) + 1
        for index in order[cursor:end]:
            output[index] = rank
        cursor = end
    return output


def pearson(a: list[float], b: list[float]) -> float | None:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else None


def state_contrast(rows: list[dict[str, str]], manifest: list[dict[str, str]], layer: int, scope: str) -> float:
    role_by_key = {(row["trajectory_id"], int(row["target_turn"])): row["sample_role"] for row in manifest}
    selected = []
    for row in rows:
        key = (row["trajectory_id"], int(row["target_turn"]))
        if key not in role_by_key or int(row["layer"]) != layer:
            continue
        if int(row["downstream_anchor_ordinal"]) != int(row["downstream_anchor_count"]) - 1:
            continue
        if scope not in {"pooled", "task_macro"} and row["task"] != scope:
            continue
        selected.append((row, role_by_key[key]))
    if scope == "task_macro":
        values = [state_contrast(rows, manifest, layer, task) for task in ("actions", "code", "data2text")]
        return sum(values) / 3
    by_trajectory = defaultdict(dict)
    for row, role in selected:
        by_trajectory[row["trajectory_id"]][role] = float(row["cosine_distance"])
    differences = [values["positive"] - values["matched_zero"] for values in by_trajectory.values()]
    return sum(differences) / len(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-anchors", type=Path, required=True)
    parser.add_argument("--restoration-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.manifest)
    state_rows = read_csv(args.state_anchors)
    restoration = read_csv(args.restoration_results)
    profile = []
    correlations = []
    for scope in SCOPES:
        state_values = []
        for layer in LAYERS:
            state_value = state_contrast(state_rows, manifest, layer, scope)
            state_values.append(state_value)
            row = {"scope": scope, "layer": layer, "state_positive_minus_zero": state_value}
            for direction in ("rescue", "reversion"):
                match = next(
                    item for item in restoration
                    if item["direction"] == direction and item["depth"] == "layer"
                    and int(item["layer"]) == layer and item["scope"] == scope
                )
                row[f"{direction}_positive_minus_zero"] = float(match["difference"])
            profile.append(row)
        for direction in ("rescue", "reversion"):
            restoration_values = [row[f"{direction}_positive_minus_zero"] for row in profile if row["scope"] == scope]
            correlations.append({
                "scope": scope,
                "direction": direction,
                "layers": len(LAYERS),
                "spearman_profile_correlation": pearson(ranks(state_values), ranks(restoration_values)),
            })
    with (args.output_dir / "state_restoration_layer_alignment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(profile)
    with (args.output_dir / "state_restoration_profile_correlations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlations[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(correlations)
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()

