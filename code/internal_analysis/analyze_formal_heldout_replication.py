"""Validate and analyze the complete FORMAL_HELDOUT_118 probe population."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["trajectory_id", "target_turn"]
TASKS = ["actions", "code", "data2text"]
QUERY_REGIONS = ["first_token", "first_8", "all_generated"]
LAYER_GROUPS = ["all_layers", "early", "mid_early", "mid_late", "late"]
REPS = 5000
SEED = 20260820


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def group_contrast(frame: pd.DataFrame, metric: str, seed: int = SEED) -> dict:
    data = frame[frame.local_effect_class.isin(["positive", "zero"])].copy()
    positive = data.loc[data.local_effect_class == "positive", metric]
    zero = data.loc[data.local_effect_class == "zero", metric]
    clusters = data.trajectory_id.drop_duplicates().tolist()
    by_cluster = []
    for trajectory_id in clusters:
        subset = data[data.trajectory_id == trajectory_id]
        p = subset.loc[subset.local_effect_class == "positive", metric]
        z = subset.loc[subset.local_effect_class == "zero", metric]
        by_cluster.append((p.sum(), len(p), z.sum(), len(z)))
    stats = np.asarray(by_cluster, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = stats[rng.integers(0, len(stats), size=(REPS, len(stats)))].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = sampled[:, 0] / sampled[:, 1] - sampled[:, 2] / sampled[:, 3]
    low, high = quantiles(draws)
    return {
        "positive_n": int(len(positive)), "zero_n": int(len(zero)),
        "trajectory_n": int(data.trajectory_id.nunique()),
        "positive_mean": float(positive.mean()), "positive_median": float(positive.median()),
        "zero_mean": float(zero.mean()), "zero_median": float(zero.median()),
        "difference": float(positive.mean() - zero.mean()), "ci_low": low, "ci_high": high,
    }


def task_macro_contrast(frame: pd.DataFrame, metric: str, seed: int = SEED) -> dict:
    task_points = {task: group_contrast(frame[frame.task == task], metric, seed)["difference"] for task in TASKS}
    task_clusters: dict[str, tuple[list[str], np.ndarray]] = {}
    for task in TASKS:
        data = frame[(frame.task == task) & frame.local_effect_class.isin(["positive", "zero"])]
        clusters = data.trajectory_id.drop_duplicates().tolist()
        rows = []
        for trajectory_id in clusters:
            subset = data[data.trajectory_id == trajectory_id]
            p = subset.loc[subset.local_effect_class == "positive", metric]
            z = subset.loc[subset.local_effect_class == "zero", metric]
            rows.append((p.sum(), len(p), z.sum(), len(z)))
        task_clusters[task] = (clusters, np.asarray(rows, dtype=float))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(REPS):
        contrasts = []
        for task in TASKS:
            clusters, stats = task_clusters[task]
            sampled = stats[rng.integers(0, len(clusters), size=len(clusters))].sum(axis=0)
            if sampled[1] and sampled[3]:
                contrasts.append(sampled[0] / sampled[1] - sampled[2] / sampled[3])
        if len(contrasts) == len(TASKS):
            draws.append(float(np.mean(contrasts)))
    low, high = quantiles(np.asarray(draws))
    return {
        "difference": float(np.mean(list(task_points.values()))), "ci_low": low, "ci_high": high,
        **{f"{task}_difference": value for task, value in task_points.items()},
    }


def all_levels(frame: pd.DataFrame, metric: str) -> list[dict]:
    rows = [{"level": "pooled", "task": "all", "metric": metric, **group_contrast(frame, metric)}]
    rows.extend(
        {"level": "task_specific", "task": task, "metric": metric, **group_contrast(frame[frame.task == task], metric)}
        for task in TASKS
    )
    rows.append({"level": "task_macro", "task": "equal_task", "metric": metric, **task_macro_contrast(frame, metric)})
    return rows


def ols_positive_coefficient(frame: pd.DataFrame, seed: int = SEED) -> dict:
    data = frame[frame.local_effect_class.isin(["positive", "zero"])].copy()
    data["positive_indicator"] = (data.local_effect_class == "positive").astype(float)
    columns = ["delta_assistant_tokens", "positive_indicator"]

    x = np.column_stack([np.ones(len(data)), *[data[column].to_numpy(float) for column in columns]])
    y = data.delta_assistant_history_attention.to_numpy(float)
    point = float(np.linalg.lstsq(x, y, rcond=None)[0][2])
    clusters = data.trajectory_id.drop_duplicates().tolist()
    sufficient = []
    for trajectory_id in clusters:
        mask = data.trajectory_id.eq(trajectory_id).to_numpy()
        cluster_x, cluster_y = x[mask], y[mask]
        sufficient.append((cluster_x.T @ cluster_x, cluster_x.T @ cluster_y))
    xtx = np.asarray([item[0] for item in sufficient])
    xty = np.asarray([item[1] for item in sufficient])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(REPS):
        indices = rng.integers(0, len(clusters), size=len(clusters))
        try:
            draws.append(float(np.linalg.solve(xtx[indices].sum(axis=0), xty[indices].sum(axis=0))[2]))
        except np.linalg.LinAlgError:
            continue
    low, high = quantiles(np.asarray(draws))
    return {"n": int(len(data)), "trajectory_n": len(clusters), "coefficient": point, "ci_low": low, "ci_high": high}


def paired_analysis(frame: pd.DataFrame, metric: str, subset_name: str) -> dict:
    values = frame[metric]
    clusters = frame.trajectory_id.drop_duplicates().tolist()
    stats = np.asarray([
        (frame.loc[frame.trajectory_id == trajectory_id, metric].sum(),
         len(frame.loc[frame.trajectory_id == trajectory_id]))
        for trajectory_id in clusters
    ], dtype=float)
    rng = np.random.default_rng(SEED)
    sampled = stats[rng.integers(0, len(stats), size=(REPS, len(stats)))].sum(axis=1)
    low, high = quantiles(sampled[:, 0] / sampled[:, 1])
    return {
        "subset": subset_name, "metric": metric, "pairs": int(len(frame)), "trajectory_n": len(clusters),
        "mean_difference": float(values.mean()), "median_difference": float(values.median()),
        "ci_low": low, "ci_high": high,
    }


def build_pairs(branches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mixed = []
    for trajectory_id, group in branches.groupby("trajectory_id"):
        positive = group[group.local_effect_class == "positive"]
        zero = group[group.local_effect_class == "zero"]
        for _, p in positive.iterrows():
            for _, z in zero.iterrows():
                mixed.append({
                    "trajectory_id": trajectory_id,
                    "positive_target_turn": int(p.target_turn), "zero_target_turn": int(z.target_turn),
                    "relative_position_difference": abs(float(p.relative_position) - float(z.relative_position)),
                    "distance_to_final_difference": abs(float(p.distance_to_final) - float(z.distance_to_final)),
                })
    all_pairs = pd.DataFrame(mixed)
    nearest = (all_pairs.sort_values(["trajectory_id", "positive_target_turn", "relative_position_difference", "zero_target_turn"])
               .drop_duplicates(["trajectory_id", "positive_target_turn"]))
    return {
        "all": all_pairs,
        "nearest_position": nearest,
        "distance_to_final_le_1": all_pairs[all_pairs.distance_to_final_difference <= 1],
        "relative_position_le_0_15": all_pairs[all_pairs.relative_position_difference <= 0.15],
        "exact_relative_position": all_pairs[all_pairs.relative_position_difference <= 1e-12],
    }


def join_pair_metrics(pairs: pd.DataFrame, metrics: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    p = metrics.rename(columns={"target_turn": "positive_target_turn", **{m: f"{m}_positive" for m in metric_names}})
    z = metrics.rename(columns={"target_turn": "zero_target_turn", **{m: f"{m}_zero" for m in metric_names}})
    result = pairs.merge(p[["trajectory_id", "positive_target_turn", *[f"{m}_positive" for m in metric_names]]],
                         on=["trajectory_id", "positive_target_turn"], validate="many_to_one")
    result = result.merge(z[["trajectory_id", "zero_target_turn", *[f"{m}_zero" for m in metric_names]]],
                          on=["trajectory_id", "zero_target_turn"], validate="many_to_one")
    for metric in metric_names:
        result[f"{metric}_difference"] = result[f"{metric}_positive"] - result[f"{metric}_zero"]
    return result


def load_shards(root: Path) -> list[Path]:
    return sorted(root.glob("L40_g*/probes/heldout_shard_*"))


def validate_and_combine(root: Path, expected_manifest: Path, output: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    shards = load_shards(root)
    if len(shards) != 8:
        raise RuntimeError(f"Expected 8 held-out shard directories, found {len(shards)}")
    expected = pd.read_csv(expected_manifest)
    expected_keys = set(map(tuple, expected[KEY].astype({"target_turn": int}).itertuples(index=False, name=None)))
    combined: dict[str, pd.DataFrame] = {}
    validation: dict[str, object] = {"expected_branches": len(expected_keys), "shards": len(shards)}

    specs = {
        "s1b": ("S1B/s1b_branch_summary.csv", 3, "query_region", set(QUERY_REGIONS)),
        "s1": ("S1/attention_case_summary.csv", 3, "query_region", set(QUERY_REGIONS)),
        "state": ("STATE_PROP/state_propagation_branch_summary.csv", 5, "layer_group", set(LAYER_GROUPS)),
    }
    for name, (relative, multiplier, group_column, groups) in specs.items():
        frame = pd.concat([pd.read_csv(path / relative) for path in shards], ignore_index=True)
        frame.target_turn = frame.target_turn.astype(int)
        keys = set(map(tuple, frame[KEY].drop_duplicates().itertuples(index=False, name=None)))
        complete_groups = frame.groupby(KEY)[group_column].agg(set).eq(groups).all()
        numeric = frame.select_dtypes(include=[np.number])
        valid = len(frame) == len(expected_keys) * multiplier and keys == expected_keys and bool(complete_groups) and np.isfinite(numeric).all().all()
        validation[name] = {"rows": len(frame), "branches": len(keys), "complete_groups": bool(complete_groups), "valid": bool(valid)}
        combined[name] = frame

    s3 = pd.DataFrame([row for path in shards for row in read_jsonl(path / "S3/s3_knockout_results.jsonl")])
    s3.target_turn = s3.target_turn.astype(int)
    s3_keys = set(map(tuple, s3[KEY].drop_duplicates().itertuples(index=False, name=None)))
    s3_groups = s3.groupby(KEY).condition.agg(set).eq({"baseline", "direct_only", "full_downstream"}).all()
    invariance = [json.loads((path / "S3/baseline_invariance_check.json").read_text(encoding="utf-8")) for path in shards]
    inv_valid = all(
        row.get("exact_output_match", row.get("manual_vs_standard_exact_output_match", False))
        and row.get("score_match", row.get("manual_vs_standard_score_match", False)) for row in invariance
    )
    validation["s3"] = {
        "rows": len(s3), "branches": len(s3_keys), "complete_conditions": bool(s3_groups),
        "baseline_invariance_all_shards": inv_valid,
        "valid": bool(len(s3) == len(expected_keys) * 3 and s3_keys == expected_keys and s3_groups and inv_valid),
    }
    combined["s3"] = s3

    anchors = pd.concat([pd.read_csv(path / "STATE_PROP/state_propagation_anchor_level.csv") for path in shards], ignore_index=True)
    anchors.target_turn = anchors.target_turn.astype(int)
    validation["state"]["anchor_rows"] = len(anchors)
    validation["state"]["anchor_values_finite"] = bool(np.isfinite(anchors.cosine_distance).all())
    validation["state"]["valid"] = bool(validation["state"]["valid"] and validation["state"]["anchor_values_finite"])
    combined["state_anchors"] = anchors

    spans = pd.concat([pd.DataFrame(read_jsonl(path / "S1B/token_spans.jsonl")) for path in shards], ignore_index=True)
    spans.target_turn = spans.target_turn.astype(int)
    combined["s1b_spans"] = spans
    validation["s1b"]["span_rows"] = len(spans)
    validation["s1b"]["valid"] = bool(validation["s1b"]["valid"] and len(spans) == len(expected_keys))

    probes = {
        "s1b": [row for path in shards for row in read_jsonl(path / "S1B/probe_results.jsonl")],
        "s1": [row for path in shards for row in read_jsonl(path / "S1/probe_results.jsonl")],
    }
    validation["s1b"]["invariance"] = bool(all(
        row.get("original_exact_native_output_match") and row.get("surgery_exact_native_output_match")
        and row.get("final_user_content_exact_match") for row in probes["s1b"]
    ))
    validation["s1b"]["valid"] = bool(validation["s1b"]["valid"] and validation["s1b"]["invariance"])
    validation["s1"]["generation_invariance"] = bool(all(row.get("exact_reference_output_match") for row in probes["s1"]))
    validation["s1"]["trajectory_probe_rows"] = len(probes["s1"])
    validation["s1"]["valid"] = bool(validation["s1"]["valid"] and validation["s1"]["generation_invariance"] and len(probes["s1"]) == 118)

    output.mkdir(parents=True, exist_ok=True)
    for name, frame in combined.items():
        if name != "s3":
            frame.to_csv(output / f"combined_{name}.csv", index=False)
    s3.to_json(output / "combined_s3.jsonl", orient="records", lines=True)
    (output / "probe_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation, combined


def analyze_s1b(branch: pd.DataFrame, spans: pd.DataFrame, output: Path) -> dict:
    span_rows = []
    for row in spans.to_dict("records"):
        original = row["original_segment_counts"]
        surgery = row["surgery_segment_counts"]
        original_tokens = original["target_assistant_turn"] + original["other_assistant_history"]
        surgery_tokens = surgery["target_assistant_turn"] + surgery["other_assistant_history"]
        span_rows.append({"trajectory_id": row["trajectory_id"], "target_turn": int(row["target_turn"]),
                          "assistant_tokens_original": original_tokens, "assistant_tokens_surgery": surgery_tokens,
                          "delta_assistant_tokens": surgery_tokens - original_tokens})
    merged = branch.merge(pd.DataFrame(span_rows), on=KEY, validate="many_to_one")
    metrics = ["delta_assistant_history_attention", "delta_assistant_history_attention_normalized"]
    rows = []
    for metric in metrics:
        rows.extend(all_levels(merged[merged.query_region == "all_generated"], metric))
    result = pd.DataFrame(rows)
    result.to_csv(output / "heldout_s1b_primary.csv", index=False)
    regression = {"pooled": ols_positive_coefficient(merged[merged.query_region == "all_generated"])}
    for task in TASKS:
        subset = merged[(merged.query_region == "all_generated") & (merged.task == task)]
        regression[task] = ols_positive_coefficient(subset) if subset.local_effect_class.nunique() >= 2 else None
    (output / "heldout_s1b_token_adjustment.json").write_text(json.dumps(regression, indent=2), encoding="utf-8")
    return {"primary": result.to_dict("records"), "token_adjustment": regression}


def analyze_s1(frame: pd.DataFrame, output: Path) -> dict:
    subset = frame[frame.query_region == "all_generated"].rename(columns={"effect_class": "local_effect_class"})
    rows = all_levels(subset, "mean_target_attention_normalized")
    pd.DataFrame(rows).to_csv(output / "heldout_s1_static.csv", index=False)
    return {"primary": rows}


def analyze_state(frame: pd.DataFrame, anchors: pd.DataFrame, branches: pd.DataFrame, output: Path) -> dict:
    overall = frame[frame.layer_group == "all_layers"].copy()
    metrics = ["final_user_anchor_divergence", "propagation_auc"]
    primary = [row for metric in metrics for row in all_levels(overall, metric)]
    pd.DataFrame(primary).to_csv(output / "heldout_state_primary.csv", index=False)
    layer_rows = []
    for layer_group in LAYER_GROUPS[1:]:
        subset = frame[frame.layer_group == layer_group]
        for metric in metrics:
            layer_rows.extend({"layer_group": layer_group, **row} for row in all_levels(subset, metric))
    pd.DataFrame(layer_rows).to_csv(output / "heldout_state_layer_bands.csv", index=False)

    anchor_branch = (anchors.groupby(
        KEY + ["task", "local_effect_class", "local_delta", "downstream_anchor_ordinal", "downstream_anchor_count"],
        as_index=False,
    ).cosine_distance.mean())
    anchor_branch["normalized_position"] = (anchor_branch.downstream_anchor_ordinal + 1) / anchor_branch.downstream_anchor_count
    anchor_branch["position_bin"] = pd.cut(
        anchor_branch.normalized_position, bins=[0, .25, .5, .75, 1],
        labels=["(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"], include_lowest=True,
    )
    branch_bin = (anchor_branch.groupby(KEY + ["task", "local_effect_class", "position_bin"], observed=True, as_index=False)
                  .cosine_distance.mean())
    profile_rows = []
    for position_bin in ["(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"]:
        subset = branch_bin[branch_bin.position_bin.astype(str) == position_bin]
        profile_rows.append({"position_bin": position_bin, **group_contrast(subset, "cosine_distance")})
    pd.DataFrame(profile_rows).to_csv(output / "heldout_state_normalized_profile.csv", index=False)
    branch_meta = branches[KEY + ["local_effect_class", "relative_position", "distance_to_final"]].drop_duplicates(KEY)
    pairs = build_pairs(branch_meta)
    pair_rows = []
    metric_frame = overall[KEY + metrics]
    for subset_name, pair_frame in pairs.items():
        if pair_frame.empty:
            continue
        joined = join_pair_metrics(pair_frame, metric_frame, metrics)
        for metric in metrics:
            pair_rows.append(paired_analysis(joined, f"{metric}_difference", subset_name))
    pd.DataFrame(pair_rows).to_csv(output / "heldout_state_within_trajectory.csv", index=False)
    pair_counts = {name: {"pairs": len(value), "trajectories": value.trajectory_id.nunique() if len(value) else 0} for name, value in pairs.items()}
    return {"primary": primary, "layer_bands": layer_rows, "normalized_profile": profile_rows,
            "within": pair_rows, "pair_counts": pair_counts}


def analyze_s3(frame: pd.DataFrame, branches: pd.DataFrame, output: Path) -> dict:
    metadata = frame.drop_duplicates(KEY)[KEY + ["task", "local_effect_class", "local_delta"]]
    scores = frame.pivot(index=KEY, columns="condition", values="evaluator_score").reset_index()
    wide = metadata.merge(scores, on=KEY, validate="one_to_one")
    wide["delta_direct"] = wide.direct_only - wide.baseline
    wide["delta_full"] = wide.full_downstream - wide.baseline
    wide["abs_delta_direct"] = wide.delta_direct.abs()
    wide["abs_delta_full"] = wide.delta_full.abs()
    wide["propagation_advantage"] = wide.abs_delta_full - wide.abs_delta_direct
    wide["direct_changed"] = wide.abs_delta_direct > 1e-12
    wide["full_changed"] = wide.abs_delta_full > 1e-12
    metrics = ["delta_direct", "abs_delta_direct", "delta_full", "abs_delta_full", "propagation_advantage"]
    primary = [row for metric in metrics for row in all_levels(wide, metric)]
    pd.DataFrame(primary).to_csv(output / "heldout_s3_primary.csv", index=False)
    descriptive = []
    for task in TASKS:
        for effect in ["positive", "zero", "negative"]:
            subset = wide[(wide.task == task) & (wide.local_effect_class == effect)]
            descriptive.append({
                "task": task, "local_effect_class": effect, "n": len(subset),
                "delta_direct_mean": subset.delta_direct.mean(), "delta_direct_median": subset.delta_direct.median(),
                "abs_delta_direct_mean": subset.abs_delta_direct.mean(), "direct_change_rate": subset.direct_changed.mean(),
                "delta_full_mean": subset.delta_full.mean(), "delta_full_median": subset.delta_full.median(),
                "abs_delta_full_mean": subset.abs_delta_full.mean(), "full_change_rate": subset.full_changed.mean(),
                "propagation_advantage_mean": subset.propagation_advantage.mean(),
            })
    pd.DataFrame(descriptive).to_csv(output / "heldout_s3_task_descriptives.csv", index=False)
    directional = []
    for task in ["all", *TASKS]:
        subset = wide if task == "all" else wide[wide.task == task]
        for metric in ["delta_direct", "delta_full"]:
            nonzero = subset[subset.local_effect_class != "zero"]
            directional.append({
                "task": task, "metric": metric, "n": len(subset), "nonzero_n": len(nonzero),
                "spearman_local_delta": subset.local_delta.rank().corr(subset[metric].rank()),
                "nonzero_sign_agreement": float((np.sign(nonzero.local_delta) == np.sign(nonzero[metric])).mean()) if len(nonzero) else np.nan,
                "nonzero_zero_knockout_rate": float((nonzero[metric].abs() <= 1e-12).mean()) if len(nonzero) else np.nan,
            })
    pd.DataFrame(directional).to_csv(output / "heldout_s3_directional_alignment.csv", index=False)
    binary = []
    for task in ["actions", "code"]:
        for effect in ["positive", "zero", "negative"]:
            subset = wide[(wide.task == task) & (wide.local_effect_class == effect)]
            binary.append({"task": task, "local_effect_class": effect, "n": len(subset),
                           "direct_flip_rate": float((subset.direct_only != subset.baseline).mean()) if len(subset) else np.nan,
                           "full_flip_rate": float((subset.full_downstream != subset.baseline).mean()) if len(subset) else np.nan})
    pd.DataFrame(binary).to_csv(output / "heldout_s3_binary_flip_rates.csv", index=False)
    branch_meta = branches[KEY + ["local_effect_class", "relative_position", "distance_to_final"]].drop_duplicates(KEY)
    pairs = build_pairs(branch_meta)
    within_rows = []
    for subset_name, pair_frame in pairs.items():
        if pair_frame.empty:
            continue
        joined = join_pair_metrics(pair_frame, wide[KEY + ["abs_delta_direct", "abs_delta_full", "propagation_advantage"]],
                                   ["abs_delta_direct", "abs_delta_full", "propagation_advantage"])
        for metric in ["abs_delta_direct", "abs_delta_full", "propagation_advantage"]:
            within_rows.append(paired_analysis(joined, f"{metric}_difference", subset_name))
    pd.DataFrame(within_rows).to_csv(output / "heldout_s3_within_trajectory.csv", index=False)
    wide.to_csv(output / "heldout_s3_branch_metrics.csv", index=False)
    return {"primary": primary, "descriptive": descriptive, "directional": directional,
            "binary_flip_rates": binary, "within": within_rows}


def analyze_s2(path: Path, output: Path, expected: int = 38, label: str = "primary") -> dict:
    frame = pd.DataFrame(read_jsonl(path))
    expected_fail = 19
    expected_control = expected - expected_fail
    valid = (
        len(frame) == expected
        and len(frame[frame.s2_group == "fail_to_success"]) == expected_fail
        and len(frame[frame.s2_group == "matched_nochange"]) == expected_control
        and frame[["match_id", "s2_group", "control_rank"]].astype(str).duplicated().sum() == 0
        and set(frame.s2_group) == {"fail_to_success", "matched_nochange"}
    )
    rows = []
    for group in ["fail_to_success", "matched_nochange"]:
        subset = frame[frame.s2_group == group]
        rows.append({"group": group, "n": len(subset), "trajectory_n": subset.trajectory_id.nunique(),
                     "mean": subset.delta_margin.mean(), "median": subset.delta_margin.median()})
    contrast_frame = frame.rename(columns={"s2_group": "local_effect_class"}).copy()
    contrast_frame.local_effect_class = contrast_frame.local_effect_class.replace({"fail_to_success": "positive", "matched_nochange": "zero"})
    contrast = group_contrast(contrast_frame, "delta_margin")
    task_rows = []
    for task in ["actions", "code"]:
        subset = contrast_frame[contrast_frame.task == task]
        task_rows.append({"task": task, **group_contrast(subset, "delta_margin")})
    report = {"valid": bool(valid), "rows": len(frame), "groups": rows, "contrast": contrast, "task_specific": task_rows}
    (output / f"heldout_s2_{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--s2-results", type=Path, required=True)
    parser.add_argument("--s2-results-1to3", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation, frames = validate_and_combine(args.probe_root, args.manifest, args.output)
    complete = all(validation[name]["valid"] for name in ["s1b", "s1", "state", "s3"])
    analyses: dict[str, object] = {}
    if validation["s1b"]["valid"]:
        analyses["s1b"] = analyze_s1b(frames["s1b"], frames["s1b_spans"], args.output)
    if validation["s1"]["valid"]:
        analyses["s1"] = analyze_s1(frames["s1"], args.output)
    manifest = pd.read_csv(args.manifest)
    if validation["state"]["valid"]:
        analyses["state"] = analyze_state(frames["state"], frames["state_anchors"], manifest, args.output)
    if validation["s3"]["valid"]:
        analyses["s3"] = analyze_s3(frames["s3"], manifest, args.output)
    analyses["s2"] = analyze_s2(args.s2_results, args.output)
    if args.s2_results_1to3:
        analyses["s2_1to3"] = analyze_s2(args.s2_results_1to3, args.output, expected=76, label="secondary_1to3")
    report = {"formal_heldout_all_probes_complete": complete, "validation": validation, "analyses": analyses,
              "settings": {"bootstrap_reps": REPS, "seed": SEED, "cluster": "trajectory_id"}}
    (args.output / "formal_heldout_replication_summary.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "complete": complete,
                      "probe_valid": {name: validation[name]["valid"] for name in ["s1b", "s1", "state", "s3"]},
                      "s2_valid": analyses["s2"]["valid"],
                      "s2_1to3_valid": analyses.get("s2_1to3", {}).get("valid")}, indent=2))


if __name__ == "__main__":
    main()

