"""Analyze complete formal full-layer and prospective coarse restoration runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


FORMAL_LAYERS = tuple(range(40))
COARSE_LAYERS = (0, 6, 11, 17, 22, 28, 33, 39)
FORMAL_REGION = tuple(range(6, 29))
COARSE_REGION = (6, 11, 17, 22, 28)
TASKS = ("actions", "code", "data2text")
PREDEFINED_REGIONS = {
    "3-12": tuple(range(3, 13)),
    "16-31": tuple(range(16, 32)),
    "3-12_union_16-31": tuple(range(3, 13)) + tuple(range(16, 32)),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def paired_values(rows: list[dict], metric: str) -> dict[str, tuple[float, float]]:
    by_trajectory: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_trajectory[row["trajectory_id"]][row["sample_role"]] = float(row[metric])
    output = {}
    for trajectory, values in by_trajectory.items():
        if "positive" not in values or "matched_zero" not in values:
            raise RuntimeError(f"Incomplete positive/zero pair for {trajectory}")
        output[trajectory] = (values["positive"], values["matched_zero"])
    return output


def bootstrap_difference(pairs: dict[str, tuple[float, float]], replicates: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    trajectories = sorted(pairs)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(trajectories) for _ in trajectories]
        estimates.append(sum(pairs[key][0] - pairs[key][1] for key in sampled) / len(sampled))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def summarize(rows: list[dict], metric: str, replicates: int, seed: int) -> dict:
    pairs = paired_values(rows, metric)
    positive = [values[0] for values in pairs.values()]
    zero = [values[1] for values in pairs.values()]
    differences = [a - b for a, b in pairs.values()]
    low, high = bootstrap_difference(pairs, replicates, seed)
    return {
        "trajectory_pairs": len(pairs),
        "positive_mean": sum(positive) / len(positive),
        "positive_median": quantile(positive, 0.5),
        "zero_mean": sum(zero) / len(zero),
        "zero_median": quantile(zero, 0.5),
        "difference": sum(differences) / len(differences),
        "paired_median_difference": quantile(differences, 0.5),
        "ci_low": low,
        "ci_high": high,
        "positive_nonzero_rate": sum(value != 0 for value in positive) / len(positive),
        "zero_nonzero_rate": sum(value != 0 for value in zero) / len(zero),
    }


def macro_summary(rows: list[dict], metric: str, replicates: int, seed: int) -> dict:
    task_pairs = {task: paired_values([row for row in rows if row["task"] == task], metric) for task in TASKS}
    task_points = {
        task: sum(a - b for a, b in pairs.values()) / len(pairs)
        for task, pairs in task_pairs.items()
    }
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled_tasks = []
        for task in TASKS:
            pairs = task_pairs[task]
            trajectories = sorted(pairs)
            sampled = [rng.choice(trajectories) for _ in trajectories]
            sampled_tasks.append(sum(pairs[key][0] - pairs[key][1] for key in sampled) / len(sampled))
        estimates.append(sum(sampled_tasks) / len(TASKS))
    return {
        "trajectory_pairs": sum(len(pairs) for pairs in task_pairs.values()),
        "positive_mean": "",
        "positive_median": "",
        "zero_mean": "",
        "zero_median": "",
        "difference": sum(task_points.values()) / len(TASKS),
        "paired_median_difference": "",
        "ci_low": quantile(estimates, 0.025),
        "ci_high": quantile(estimates, 0.975),
        "positive_nonzero_rate": "",
        "zero_nonzero_rate": "",
    }


def validate(manifest: list[dict], results: list[dict], layers: tuple[int, ...]) -> dict:
    expected = {
        (row["trajectory_id"], int(row["target_turn"]), row["sample_role"], direction, layer)
        for row in manifest for direction in ("rescue", "reversion") for layer in layers
    }
    actual = [
        (row["trajectory_id"], int(row["target_turn"]), row["sample_role"], row["patch_direction"], int(row["layer"]))
        for row in results
    ]
    invalid = 0
    for row in results:
        if any(not math.isfinite(float(row[field])) for field in ("original_score", "surgery_score", "patched_score", "local_delta")):
            invalid += 1
        elif not isinstance(row.get("evaluator"), dict) or row.get("metadata", {}).get("patch_apply_count") != 1:
            invalid += 1
    result = {
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "missing": len(expected - set(actual)),
        "unexpected": len(set(actual) - expected),
        "duplicates": sum(count - 1 for count in Counter(actual).values() if count > 1),
        "invalid": invalid,
    }
    result["status"] = "PASS" if all(result[key] == 0 for key in ("missing", "unexpected", "duplicates", "invalid")) and len(actual) == len(expected) else "FAIL"
    return result


def branch_layer_average(rows: list[dict], direction: str, metric: str, layers: tuple[int, ...]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    metadata = {}
    for row in rows:
        if row["patch_direction"] != direction or int(row["layer"]) not in layers:
            continue
        key = (row["trajectory_id"], int(row["target_turn"]), row["sample_role"])
        grouped[key].append(float(row[metric]))
        metadata[key] = row
    output = []
    for key, values in grouped.items():
        if len(values) != len(layers):
            raise RuntimeError(f"Incomplete region for {key}: {len(values)}/{len(layers)}")
        row = dict(metadata[key])
        row[metric] = sum(values) / len(values)
        output.append(row)
    return output


def analyze_population(
    name: str, results: list[dict], layers: tuple[int, ...], replicates: int, seed: int,
) -> list[dict]:
    output = []
    metrics = {"rescue": "rescue_effect", "reversion": "reversion_effect"}
    scopes = ("pooled", *TASKS, "task_macro") if name == "FORMAL_HELDOUT_D2T" else ("pooled", "data2text")
    for direction, metric in metrics.items():
        direction_rows = [row for row in results if row["patch_direction"] == direction]
        for layer in layers:
            layer_rows = [row for row in direction_rows if int(row["layer"]) == layer]
            for scope in scopes:
                selected = layer_rows if scope in ("pooled", "task_macro") else [row for row in layer_rows if row["task"] == scope]
                summary = macro_summary(selected, metric, replicates, seed + layer) if scope == "task_macro" else summarize(selected, metric, replicates, seed + layer)
                output.append({"population": name, "summary": "layer", "layer": layer, "region": "", "direction": direction, "scope": scope, **summary})
        region_layers = FORMAL_REGION if layers == FORMAL_LAYERS else COARSE_REGION
        if all(layer in layers for layer in region_layers):
            region_rows = branch_layer_average(direction_rows, direction, metric, region_layers)
            for scope in scopes:
                selected = region_rows if scope in ("pooled", "task_macro") else [row for row in region_rows if row["task"] == scope]
                summary = macro_summary(selected, metric, replicates, seed + 628) if scope == "task_macro" else summarize(selected, metric, replicates, seed + 628)
                region_name = "6-28_full" if region_layers == FORMAL_REGION else "6-28_coarse_samples"
                output.append({"population": name, "summary": "region", "layer": "", "region": region_name, "direction": direction, "scope": scope, **summary})
    return output


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1) + 1
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def pearson(a: list[float], b: list[float]) -> float | None:
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else None


def profile_correlation(a: list[float], b: list[float]) -> float | None:
    return pearson(ranks(a), ranks(b))


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def positive_bands(values: list[float]) -> list[tuple[int, int]]:
    bands = []
    start = None
    for layer, value in enumerate(values + [0.0]):
        if value > 0 and start is None:
            start = layer
        elif value <= 0 and start is not None:
            bands.append((start, layer - 1))
            start = None
    return bands


def state_contrast(state_rows: list[dict], manifest: list[dict], layer: int, task: str | None) -> float:
    role = {(row["trajectory_id"], int(row["target_turn"])): row["sample_role"] for row in manifest}
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in state_rows:
        key = (row["trajectory_id"], int(row["target_turn"]))
        if key not in role or int(row["layer"]) != layer:
            continue
        if int(row["downstream_anchor_ordinal"]) != int(row["downstream_anchor_count"]) - 1:
            continue
        if task and row["task"] != task:
            continue
        pairs[row["trajectory_id"]][role[key]] = float(row["cosine_distance"])
    differences = [values["positive"] - values["matched_zero"] for values in pairs.values()]
    return sum(differences) / len(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, action="append", required=True)
    parser.add_argument("--prospective-manifest", type=Path, required=True)
    parser.add_argument("--prospective-result", type=Path, action="append", required=True)
    parser.add_argument("--prospective-full-result", type=Path, action="append", default=[])
    parser.add_argument("--formal-state-anchors", type=Path, required=True)
    parser.add_argument("--prospective-state-anchor", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    formal_manifest = read_csv(args.formal_manifest)
    prospective_manifest = read_csv(args.prospective_manifest)
    formal_results = [row for path in args.formal_result for row in read_jsonl(path)]
    prospective_results = [row for path in (*args.prospective_result, *args.prospective_full_result) for row in read_jsonl(path)]
    prospective_layers = FORMAL_LAYERS if args.prospective_full_result else COARSE_LAYERS
    integrity = {
        "formal_full_layer": validate(formal_manifest, formal_results, FORMAL_LAYERS),
        "prospective_full_layer" if args.prospective_full_result else "prospective_coarse": validate(
            prospective_manifest, prospective_results, prospective_layers
        ),
    }
    (args.output_dir / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    if any(value["status"] != "PASS" for value in integrity.values()):
        raise RuntimeError(f"Restoration integrity failure: {integrity}")

    formal = analyze_population("FORMAL_HELDOUT_D2T", formal_results, FORMAL_LAYERS, args.bootstrap, args.seed)
    prospective = analyze_population("PROSPECTIVE_D2T", prospective_results, prospective_layers, args.bootstrap, args.seed + 1000)
    write_csv(args.output_dir / "formal_full_layer_results.csv", formal)
    prospective_name = "prospective_full_layer_results.csv" if args.prospective_full_result else "prospective_coarse_results.csv"
    write_csv(args.output_dir / prospective_name, prospective)

    synthesis = []
    for population, rows, layers in (
        ("FORMAL_HELDOUT_D2T", formal, COARSE_LAYERS),
        ("PROSPECTIVE_D2T", prospective, prospective_layers),
    ):
        for layer in layers:
            for direction in ("rescue", "reversion"):
                row = next(item for item in rows if item["summary"] == "layer" and item["scope"] == "data2text" and item["direction"] == direction and int(item["layer"]) == layer)
                synthesis.append({"population": population, "summary": "layer", "layer": layer, "direction": direction, **{key: row[key] for key in ("trajectory_pairs", "positive_mean", "zero_mean", "difference", "ci_low", "ci_high")}})
        for direction in ("rescue", "reversion"):
            source_results = formal_results if population == "FORMAL_HELDOUT_D2T" else prospective_results
            region_rows = branch_layer_average(source_results, direction, f"{direction}_effect", COARSE_REGION)
            region_summary = summarize([row for row in region_rows if row["task"] == "data2text"], f"{direction}_effect", args.bootstrap, args.seed + 2000)
            synthesis.append({"population": population, "summary": "region_6_28_coarse_samples", "layer": "", "direction": direction, **{key: region_summary[key] for key in ("trajectory_pairs", "positive_mean", "zero_mean", "difference", "ci_low", "ci_high")}})
            if population == "FORMAL_HELDOUT_D2T":
                full_region_rows = branch_layer_average(source_results, direction, f"{direction}_effect", FORMAL_REGION)
                full_region_summary = summarize([row for row in full_region_rows if row["task"] == "data2text"], f"{direction}_effect", args.bootstrap, args.seed + 3000)
                synthesis.append({"population": population, "summary": "region_6_28_full", "layer": "", "direction": direction, **{key: full_region_summary[key] for key in ("trajectory_pairs", "positive_mean", "zero_mean", "difference", "ci_low", "ci_high")}})
    correlations = []
    comparison_layers = prospective_layers
    for direction in ("rescue", "reversion"):
        formal_profile = [next(row for row in formal if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == direction and int(row["layer"]) == layer)["difference"] for layer in comparison_layers]
        prospective_profile = [next(row for row in prospective if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == direction and int(row["layer"]) == layer)["difference"] for layer in comparison_layers]
        correlation = profile_correlation(formal_profile, prospective_profile)
        correlations.append({"comparison": "formal_vs_prospective", "scope": "data2text", "direction": direction, "layers": len(comparison_layers), "spearman": correlation})

    for row in synthesis:
        row["trajectory_n"] = row["trajectory_pairs"]
        row["pair_n"] = row["trajectory_pairs"]
        row["profile_spearman"] = ""
    for direction in ("rescue", "reversion"):
        correlation = next(row["spearman"] for row in correlations if row["comparison"] == "formal_vs_prospective" and row["direction"] == direction)
        synthesis.append({
            "population": "FORMAL_VS_PROSPECTIVE",
            "summary": f"profile_spearman_{len(comparison_layers)}_layers",
            "layer": "",
            "direction": direction,
            "trajectory_pairs": "",
            "positive_mean": "",
            "zero_mean": "",
            "difference": "",
            "ci_low": "",
            "ci_high": "",
            "trajectory_n": "33/43",
            "pair_n": "33/43",
            "profile_spearman": correlation,
        })
    write_csv(args.output_dir / "STATE_RESTORATION_REPLICATION_SYNTHESIS.csv", synthesis)

    formal_state = read_csv(args.formal_state_anchors)
    prospective_state = [row for path in args.prospective_state_anchor for row in read_csv(path)]
    for population, manifest, state_rows, restoration_rows, layers in (
        ("FORMAL_HELDOUT_D2T", formal_manifest, formal_state, formal, FORMAL_LAYERS),
        ("PROSPECTIVE_D2T", prospective_manifest, prospective_state, prospective, prospective_layers),
    ):
        if not state_rows:
            continue
        state_profile = [state_contrast(state_rows, manifest, layer, "data2text") for layer in layers]
        for direction in ("rescue", "reversion"):
            restoration_profile = [next(row for row in restoration_rows if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == direction and int(row["layer"]) == layer)["difference"] for layer in layers]
            correlations.append({"comparison": "state_propagation_vs_restoration", "scope": population, "direction": direction, "layers": len(layers), "spearman": profile_correlation(state_profile, restoration_profile)})
    write_csv(args.output_dir / "profile_correlations.csv", correlations)

    if args.prospective_full_result:
        full_replication = []
        for layer in FORMAL_LAYERS:
            output = {"summary": "layer", "layer_or_region": layer}
            for population_name, population_rows in (("formal", formal), ("prospective", prospective)):
                for direction in ("rescue", "reversion"):
                    row = next(
                        item for item in population_rows
                        if item["summary"] == "layer" and item["scope"] == "data2text"
                        and item["direction"] == direction and int(item["layer"]) == layer
                    )
                    output[f"{population_name}_{direction}_difference"] = row["difference"]
                    output[f"{population_name}_{direction}_ci_low"] = row["ci_low"]
                    output[f"{population_name}_{direction}_ci_high"] = row["ci_high"]
            full_replication.append(output)
        for region_name, region_layers in PREDEFINED_REGIONS.items():
            output = {"summary": "region", "layer_or_region": region_name}
            for population_name, raw_results in (("formal", formal_results), ("prospective", prospective_results)):
                for direction in ("rescue", "reversion"):
                    metric = f"{direction}_effect"
                    region_rows = branch_layer_average(raw_results, direction, metric, region_layers)
                    region_summary = summarize(
                        [row for row in region_rows if row["task"] == "data2text"],
                        metric,
                        args.bootstrap,
                        args.seed + sum(region_layers) + (0 if population_name == "formal" else 1000),
                    )
                    output[f"{population_name}_{direction}_difference"] = region_summary["difference"]
                    output[f"{population_name}_{direction}_ci_low"] = region_summary["ci_low"]
                    output[f"{population_name}_{direction}_ci_high"] = region_summary["ci_high"]
            full_replication.append(output)
        write_csv(args.output_dir / "STATE_RESTORATION_FULL_LAYER_REPLICATION.csv", full_replication)

        depth_profile = {}
        for direction in ("rescue", "reversion"):
            formal_values = [
                float(next(row for row in formal if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == direction and int(row["layer"]) == layer)["difference"])
                for layer in FORMAL_LAYERS
            ]
            prospective_values = [
                float(next(row for row in prospective if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == direction and int(row["layer"]) == layer)["difference"])
                for layer in FORMAL_LAYERS
            ]
            formal_bands = positive_bands(formal_values)
            prospective_bands = positive_bands(prospective_values)
            overlaps = [
                (max(a0, b0), min(a1, b1))
                for a0, a1 in formal_bands for b0, b1 in prospective_bands
                if max(a0, b0) <= min(a1, b1)
            ]
            strongest = max(range(40), key=lambda layer: prospective_values[layer])
            longest = max(prospective_bands, key=lambda band: band[1] - band[0] + 1) if prospective_bands else None
            depth_profile[direction] = {
                "spearman_40_layers": profile_correlation(formal_values, prospective_values),
                "sign_concordance_all": sum(sign(a) == sign(b) for a, b in zip(formal_values, prospective_values)),
                "sign_concordance_regions": {
                    name: sum(sign(formal_values[layer]) == sign(prospective_values[layer]) for layer in layers)
                    for name, layers in PREDEFINED_REGIONS.items()
                },
                "region_sizes": {name: len(layers) for name, layers in PREDEFINED_REGIONS.items()},
                "formal_positive_bands": formal_bands,
                "prospective_positive_bands": prospective_bands,
                "positive_band_overlaps": overlaps,
                "prospective_strongest_layer": strongest,
                "prospective_strongest_difference": prospective_values[strongest],
                "prospective_longest_positive_band": longest,
            }
        (args.output_dir / "depth_profile_replication.json").write_text(
            json.dumps(depth_profile, indent=2) + "\n", encoding="utf-8"
        )

    formal_reversion = [next(row for row in formal if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == "reversion" and int(row["layer"]) == layer)["difference"] for layer in (6, 11, 17, 22, 28)]
    prospective_reversion = [next(row for row in prospective if row["summary"] == "layer" and row["scope"] == "data2text" and row["direction"] == "reversion" and int(row["layer"]) == layer)["difference"] for layer in (6, 11, 17, 22, 28)]
    summary = {
        "integrity": integrity,
        "formal_positive_layers_6_11_17_22_28": sum(value > 0 for value in formal_reversion),
        "prospective_sign_concordance": sum((a > 0) == (b > 0) for a, b in zip(formal_reversion, prospective_reversion)),
        "profile_correlations": correlations,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

