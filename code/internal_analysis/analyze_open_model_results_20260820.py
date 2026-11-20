"""Validate and analyze the completed S3, State Propagation, and S1b twins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPS = 5000
SEED = 20260820
KEY = ["trajectory_id", "target_turn"]
QUERY_REGIONS = ["first_token", "first_8", "all_generated"]
LAYER_BANDS = ["early", "mid_early", "mid_late", "late"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(series: pd.Series) -> bool:
    return bool(np.isfinite(pd.to_numeric(series, errors="coerce")).all())


def cluster_bootstrap_groups(
    frame: pd.DataFrame,
    value: str,
    group: str = "local_effect_class",
    positive: str = "positive",
    zero: str = "zero",
    reps: int = REPS,
    seed: int = SEED,
) -> dict:
    data = frame[frame[group].isin([positive, zero])].copy()
    clusters = data["trajectory_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    cluster_stats = []
    for cluster in clusters:
        subset = data[data["trajectory_id"] == cluster]
        p_values = subset.loc[subset[group] == positive, value]
        z_values = subset.loc[subset[group] == zero, value]
        cluster_stats.append((p_values.sum(), len(p_values), z_values.sum(), len(z_values)))
    cluster_stats = np.asarray(cluster_stats, dtype=float)
    sampled_indices = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    sampled_stats = cluster_stats[sampled_indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_draws = sampled_stats[:, 0] / sampled_stats[:, 1]
        z_draws = sampled_stats[:, 2] / sampled_stats[:, 3]
    draws = np.column_stack((p_draws, z_draws, p_draws - z_draws))
    p_values = data.loc[data[group] == positive, value]
    z_values = data.loc[data[group] == zero, value]
    return {
        "positive_n": int(len(p_values)),
        "zero_n": int(len(z_values)),
        "positive_mean": float(p_values.mean()),
        "positive_median": float(p_values.median()),
        "zero_mean": float(z_values.mean()),
        "zero_median": float(z_values.median()),
        "difference": float(p_values.mean() - z_values.mean()),
        "positive_mean_ci_low": float(np.nanquantile(draws[:, 0], 0.025)),
        "positive_mean_ci_high": float(np.nanquantile(draws[:, 0], 0.975)),
        "zero_mean_ci_low": float(np.nanquantile(draws[:, 1], 0.025)),
        "zero_mean_ci_high": float(np.nanquantile(draws[:, 1], 0.975)),
        "difference_ci_low": float(np.nanquantile(draws[:, 2], 0.025)),
        "difference_ci_high": float(np.nanquantile(draws[:, 2], 0.975)),
        "bootstrap_reps": reps,
        "bootstrap_clusters": len(clusters),
    }


def paired_cluster_bootstrap(frame: pd.DataFrame, value: str, reps: int = REPS, seed: int = SEED) -> dict:
    clusters = frame["trajectory_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    cluster_stats = np.asarray(
        [
            (
                frame.loc[frame["trajectory_id"] == cluster, value].sum(),
                len(frame.loc[frame["trajectory_id"] == cluster, value]),
            )
            for cluster in clusters
        ],
        dtype=float,
    )
    sampled_indices = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    sampled_stats = cluster_stats[sampled_indices].sum(axis=1)
    draws = sampled_stats[:, 0] / sampled_stats[:, 1]
    values = frame[value]
    return {
        "pairs": int(len(values)),
        "trajectories": len(clusters),
        "mean_paired_difference": float(values.mean()),
        "median_paired_difference": float(values.median()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_reps": reps,
    }


def spearman(frame: pd.DataFrame, x: str, y: str) -> dict:
    clean = frame[[x, y]].dropna()
    rho = clean[x].rank(method="average").corr(clean[y].rank(method="average"), method="pearson")
    return {"n": int(len(clean)), "rho": float(rho)}


def validate_s3(directory: Path, expected_branches: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(read_jsonl(directory / "s3_knockout_results.jsonl"))
    spans = pd.DataFrame(read_jsonl(directory / "token_spans.jsonl"))
    invariance = json.loads((directory / "baseline_invariance_check.json").read_text(encoding="utf-8"))
    expected_rows = expected_branches * 3
    keys = results[KEY].drop_duplicates()
    condition_counts = results.groupby(KEY)["condition"].agg(lambda values: set(values))
    valid_conditions = condition_counts.apply(lambda values: values == {"baseline", "direct_only", "full_downstream"}).all()
    report = {
        "rows": int(len(results)),
        "expected_rows": expected_rows,
        "branches": int(len(keys)),
        "span_rows": int(len(spans)),
        "conditions_complete": bool(valid_conditions),
        "scores_finite": finite(results["evaluator_score"]),
        "target_spans_nonempty": bool((spans["target_token_count"] > 0).all()),
        "target_span_counts_consistent": bool(
            ((spans["target_token_end_exclusive"] - spans["target_token_start"]) == spans["target_token_count"]).all()
        ),
        "baseline_invariance": invariance,
        "run_config": json.loads((directory / "run_config.json").read_text(encoding="utf-8")),
    }
    report["valid"] = bool(
        report["rows"] == expected_rows
        and report["branches"] == expected_branches
        and report["span_rows"] == expected_branches
        and report["conditions_complete"]
        and report["scores_finite"]
        and report["target_spans_nonempty"]
        and report["target_span_counts_consistent"]
        and invariance.get("exact_output_match", invariance.get("manual_vs_standard_exact_output_match"))
        and invariance.get("score_match", invariance.get("manual_vs_standard_score_match"))
    )
    return report, results, spans


def validate_state(directory: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    anchors = pd.read_csv(directory / "state_propagation_anchor_level.csv")
    summary = pd.read_csv(directory / "state_propagation_branch_summary.csv")
    key_counts = summary[KEY].drop_duplicates()
    labels = summary.drop_duplicates(KEY)["local_effect_class"].value_counts().to_dict()
    expected_groups = summary.groupby(KEY)["layer_group"].agg(lambda values: set(values))
    expected_layers = anchors.groupby(KEY + ["downstream_user_turn"])["layer"].agg(lambda values: set(values))
    report = {
        "anchor_rows": int(len(anchors)),
        "summary_rows": int(len(summary)),
        "branches": int(len(key_counts)),
        "trajectories": int(summary["trajectory_id"].nunique()),
        "labels": {str(key): int(value) for key, value in labels.items()},
        "all_layer_groups_present": bool(expected_groups.apply(lambda values: values == {"all_layers", *LAYER_BANDS}).all()),
        "all_40_layers_present": bool(expected_layers.apply(lambda values: values == set(range(40))).all()),
        "cosine_values_finite": finite(anchors["cosine_distance"]),
        "cosine_values_in_range": bool(anchors["cosine_distance"].between(-1e-6, 2 + 1e-6).all()),
        "fixed_user_hash_nonempty": bool(anchors["user_content_sha256"].str.len().eq(64).all()),
        "run_config": json.loads((directory / "run_config.json").read_text(encoding="utf-8")),
    }
    report["valid"] = bool(
        report["branches"] == 131
        and report["trajectories"] == 24
        and report["labels"] == {"zero": 93, "positive": 26, "negative": 12}
        and report["summary_rows"] == 655
        and report["all_layer_groups_present"]
        and report["all_40_layers_present"]
        and report["cosine_values_finite"]
        and report["cosine_values_in_range"]
        and report["fixed_user_hash_nonempty"]
    )
    return report, anchors, summary


def validate_s1b(directory: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branch = pd.read_csv(directory / "s1b_branch_summary.csv")
    bands = pd.read_csv(directory / "s1b_layer_band_summary.csv")
    probes = pd.DataFrame(read_jsonl(directory / "probe_results.jsonl"))
    spans = pd.DataFrame(read_jsonl(directory / "token_spans.jsonl"))
    invariance = json.loads((directory / "first_pair_invariance_check.json").read_text(encoding="utf-8"))
    keys = branch[KEY].drop_duplicates()
    labels = branch.drop_duplicates(KEY)["local_effect_class"].value_counts().to_dict()
    query_sets = branch.groupby(KEY)["query_region"].agg(lambda values: set(values))
    band_sets = bands.groupby(KEY + ["query_region"])["layer_group"].agg(lambda values: set(values))
    report = {
        "branch_rows": int(len(branch)),
        "band_rows": int(len(bands)),
        "branches": int(len(keys)),
        "trajectories": int(branch["trajectory_id"].nunique()),
        "labels": {str(key): int(value) for key, value in labels.items()},
        "probe_rows": int(len(probes)),
        "span_rows": int(len(spans)),
        "query_regions_complete": bool(query_sets.apply(lambda values: values == set(QUERY_REGIONS)).all()),
        "layer_bands_complete": bool(band_sets.apply(lambda values: values == set(LAYER_BANDS)).all()),
        "original_exact_matches": int(probes["original_exact_native_output_match"].sum()),
        "surgery_exact_matches": int(probes["surgery_exact_native_output_match"].sum()),
        "final_user_exact_matches": int(probes["final_user_content_exact_match"].sum()),
        "metrics_finite": bool(all(finite(branch[column]) for column in branch if column.startswith(("original_", "surgery_", "delta_")))),
        "first_pair_invariance": invariance,
        "run_config": json.loads((directory / "run_config.json").read_text(encoding="utf-8")),
    }
    report["valid"] = bool(
        report["branch_rows"] == 393
        and report["band_rows"] == 1572
        and report["branches"] == 131
        and report["trajectories"] == 24
        and report["labels"] == {"zero": 93, "positive": 26, "negative": 12}
        and report["probe_rows"] == 131
        and report["span_rows"] == 131
        and report["query_regions_complete"]
        and report["layer_bands_complete"]
        and report["original_exact_matches"] == 131
        and report["surgery_exact_matches"] == 131
        and report["final_user_exact_matches"] == 131
        and report["metrics_finite"]
    )
    return report, branch, bands, probes


def hardware_compare(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], numeric: list[str], text: list[str]) -> dict:
    merged = left.merge(right, on=keys, suffixes=("_l40", "_gpujl"), validate="one_to_one")
    output = {"left_rows": int(len(left)), "right_rows": int(len(right)), "matched_rows": int(len(merged))}
    for column in numeric:
        output[f"{column}_max_abs_diff"] = float(
            np.max(np.abs(merged[f"{column}_l40"] - merged[f"{column}_gpujl"]))
        )
    for column in text:
        output[f"{column}_exact_matches"] = int((merged[f"{column}_l40"] == merged[f"{column}_gpujl"]).sum())
    output["complete_key_match"] = len(merged) == len(left) == len(right)
    return output


def state_analysis(summary: pd.DataFrame, anchors: pd.DataFrame, output: Path) -> dict:
    rows = []
    for group in ["all_layers", *LAYER_BANDS]:
        subset = summary[summary["layer_group"] == group]
        for metric in ["final_user_anchor_divergence", "mean_downstream_anchor_divergence", "propagation_auc"]:
            rows.append({"layer_group": group, "metric": metric, **cluster_bootstrap_groups(subset, metric)})
    primary = pd.DataFrame(rows)
    primary.to_csv(output / "state_propagation_primary.csv", index=False)

    overall = summary[summary["layer_group"] == "all_layers"].copy()
    correlations = {
        "all_branches": spearman(overall.assign(abs_local_delta=overall["local_delta"].abs()), "abs_local_delta", "propagation_auc"),
        "nonzero_branches": spearman(
            overall[overall["local_effect_class"] != "zero"].assign(abs_local_delta=lambda x: x["local_delta"].abs()),
            "abs_local_delta", "propagation_auc",
        ),
    }

    anchor_branch = (
        anchors.groupby(KEY + ["task", "local_effect_class", "local_delta", "downstream_anchor_ordinal", "downstream_anchor_count"], as_index=False)
        ["cosine_distance"].mean()
    )
    anchor_branch["normalized_position"] = (anchor_branch["downstream_anchor_ordinal"] + 1) / anchor_branch["downstream_anchor_count"]
    anchor_branch["position_bin"] = pd.cut(
        anchor_branch["normalized_position"], bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"], include_lowest=True,
    )
    branch_bin = (
        anchor_branch.groupby(
            KEY + ["task", "local_effect_class", "local_delta", "position_bin"],
            observed=True,
            as_index=False,
        )["cosine_distance"].mean()
    )
    profile_rows = []
    for position_bin in ["(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"]:
        subset = branch_bin[branch_bin["position_bin"].astype(str) == position_bin]
        profile_rows.append({"position_bin": position_bin, **cluster_bootstrap_groups(subset, "cosine_distance")})
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(output / "state_propagation_normalized_profile.csv", index=False)
    return {"correlations": correlations, "primary_rows": len(primary), "profile_rows": len(profile)}


def within_analysis(metric_frame: pd.DataFrame, controls_dir: Path, metrics: list[str], prefix: str, output: Path) -> pd.DataFrame:
    rows = []
    for set_name, filename in [
        ("all_40_combinations", "all_within_trajectory_pairs.csv"),
        ("nearest_22_pairs", "preferred_nearest_position_pairs.csv"),
    ]:
        pairs = pd.read_csv(controls_dir / filename)
        positive = metric_frame.rename(columns={"target_turn": "positive_target_turn"})
        zero = metric_frame.rename(columns={"target_turn": "zero_target_turn"})
        keep_positive = ["trajectory_id", "positive_target_turn", *metrics]
        keep_zero = ["trajectory_id", "zero_target_turn", *metrics]
        joined = pairs.merge(positive[keep_positive], on=["trajectory_id", "positive_target_turn"], validate="many_to_one")
        joined = joined.merge(zero[keep_zero], on=["trajectory_id", "zero_target_turn"], suffixes=("_positive", "_zero"), validate="many_to_one")
        for metric in metrics:
            difference = f"{metric}_paired_difference"
            joined[difference] = joined[f"{metric}_positive"] - joined[f"{metric}_zero"]
            result = paired_cluster_bootstrap(joined, difference)
            result.update(
                {
                    "analysis": prefix,
                    "pair_set": set_name,
                    "metric": metric,
                    "relative_position_mean_abs_difference": float(pairs["absolute_relative_position_difference"].mean()),
                    "relative_position_median_abs_difference": float(pairs["absolute_relative_position_difference"].median()),
                    "relative_position_max_abs_difference": float(pairs["absolute_relative_position_difference"].max()),
                    "exact_relative_position_matches": int((pairs["absolute_relative_position_difference"] == 0).sum()),
                }
            )
            rows.append(result)
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output / f"{prefix}_within_trajectory.csv", index=False)
    return result_frame


def s1b_analysis(branch: pd.DataFrame, bands: pd.DataFrame, output: Path) -> dict:
    metrics = [
        "delta_final_user_attention",
        "delta_assistant_history_attention",
        "delta_assistant_history_attention_normalized",
        "delta_previous_user_attention",
    ]
    primary_rows = []
    for query in QUERY_REGIONS:
        subset = branch[branch["query_region"] == query]
        for metric in metrics:
            primary_rows.append({"query_region": query, "metric": metric, **cluster_bootstrap_groups(subset, metric)})
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(output / "s1b_primary.csv", index=False)
    layer_rows = []
    for query in QUERY_REGIONS:
        for band in LAYER_BANDS:
            subset = bands[(bands["query_region"] == query) & (bands["layer_group"] == band)]
            for metric in metrics[:3]:
                layer_rows.append(
                    {"query_region": query, "layer_group": band, "metric": metric, **cluster_bootstrap_groups(subset, metric)}
                )
    layer = pd.DataFrame(layer_rows)
    layer.to_csv(output / "s1b_layer_bands.csv", index=False)
    return {"primary_rows": len(primary), "layer_rows": len(layer)}


def s3_wide(results: pd.DataFrame) -> pd.DataFrame:
    metadata = results.drop_duplicates(KEY)[KEY + ["task", "local_effect_class", "local_delta", "formal_effect_class"]]
    scores = results.pivot(index=KEY, columns="condition", values="evaluator_score").reset_index()
    wide = metadata.merge(scores, on=KEY, validate="one_to_one")
    wide["delta_direct"] = wide["direct_only"] - wide["baseline"]
    wide["delta_full"] = wide["full_downstream"] - wide["baseline"]
    wide["abs_delta_direct"] = wide["delta_direct"].abs()
    wide["abs_delta_full"] = wide["delta_full"].abs()
    wide["propagation_advantage"] = wide["abs_delta_full"] - wide["abs_delta_direct"]
    wide["direct_changed"] = wide["abs_delta_direct"] > 1e-12
    wide["full_changed"] = wide["abs_delta_full"] > 1e-12
    wide["binary_task"] = wide["task"].isin(["actions", "code"])
    wide["direct_binary_flip"] = wide["binary_task"] & (wide["direct_only"] != wide["baseline"])
    wide["full_binary_flip"] = wide["binary_task"] & (wide["full_downstream"] != wide["baseline"])
    return wide


def s3_analysis(results: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = s3_wide(results)
    rows = []
    for metric in ["delta_direct", "abs_delta_direct", "delta_full", "abs_delta_full", "propagation_advantage"]:
        rows.append({"metric": metric, **cluster_bootstrap_groups(wide, metric)})
    primary = pd.DataFrame(rows)
    primary.to_csv(output / "s3_primary.csv", index=False)
    rates = []
    for effect_class in ["positive", "zero", "negative"]:
        subset = wide[wide["local_effect_class"] == effect_class]
        binary = subset[subset["binary_task"]]
        rates.append(
            {
                "local_effect_class": effect_class,
                "n": len(subset),
                "direct_score_change_fraction": float(subset["direct_changed"].mean()),
                "full_score_change_fraction": float(subset["full_changed"].mean()),
                "binary_n": len(binary),
                "direct_binary_flip_fraction": float(binary["direct_binary_flip"].mean()) if len(binary) else None,
                "full_binary_flip_fraction": float(binary["full_binary_flip"].mean()) if len(binary) else None,
            }
        )
    rate_frame = pd.DataFrame(rates)
    rate_frame.to_csv(output / "s3_change_and_flip_rates.csv", index=False)
    return wide, rate_frame


def point_difference(frame: pd.DataFrame, metric: str) -> float:
    positive = frame.loc[frame["local_effect_class"] == "positive", metric].mean()
    zero = frame.loc[frame["local_effect_class"] == "zero", metric].mean()
    return float(positive - zero)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = {}
    pilot = {}
    for hardware in ["L40", "gpujl"]:
        validation[f"s3_pilot_{hardware}"], pilot[hardware], _ = validate_s3(args.raw_root / "s3_pilot" / hardware, 16)
    state = {}
    for hardware in ["L40", "gpujl"]:
        validation[f"state_{hardware}"], state[f"anchors_{hardware}"], state[f"summary_{hardware}"] = validate_state(
            args.raw_root / "state_prop" / hardware
        )
    full = {}
    for hardware in ["L40", "gpujl"]:
        validation[f"s3_full_{hardware}"], full[hardware], _ = validate_s3(args.raw_root / "s3_full" / hardware, 131)
    s1b = {}
    for hardware in ["L40", "gpujl"]:
        validation[f"s1b_{hardware}"], s1b[f"branch_{hardware}"], s1b[f"bands_{hardware}"], s1b[f"probes_{hardware}"] = validate_s1b(
            args.raw_root / "s1b" / hardware
        )

    hardware = {
        "s3_pilot": hardware_compare(
            pilot["L40"], pilot["gpujl"], KEY + ["condition"], ["evaluator_score", "delta_vs_baseline"], ["generated_output"]
        ),
        "state_propagation": hardware_compare(
            state["summary_L40"], state["summary_gpujl"], KEY + ["layer_group"],
            ["final_user_anchor_divergence", "mean_downstream_anchor_divergence", "propagation_auc"], [],
        ),
        "s3_full": hardware_compare(
            full["L40"], full["gpujl"], KEY + ["condition"], ["evaluator_score", "delta_vs_baseline"], ["generated_output"]
        ),
        "s1b": hardware_compare(
            s1b["branch_L40"], s1b["branch_gpujl"], KEY + ["query_region", "layer_group"],
            ["delta_final_user_attention", "delta_assistant_history_attention", "delta_assistant_history_attention_normalized"], [],
        ),
    }

    state_result = state_analysis(state["summary_L40"], state["anchors_L40"], args.output_dir)
    state_overall = state["summary_L40"][state["summary_L40"]["layer_group"] == "all_layers"]
    state_within = within_analysis(
        state_overall, args.controls_dir,
        ["final_user_anchor_divergence", "propagation_auc"], "state_propagation", args.output_dir,
    )
    s1b_result = s1b_analysis(s1b["branch_L40"], s1b["bands_L40"], args.output_dir)
    s3_wide_frame, s3_rates = s3_analysis(full["L40"], args.output_dir)
    s3_within = within_analysis(
        s3_wide_frame, args.controls_dir,
        ["abs_delta_direct", "abs_delta_full", "propagation_advantage"], "s3", args.output_dir,
    )

    negative_rows = []
    state_negative = state_overall[state_overall["local_effect_class"] == "negative"]
    for metric in ["final_user_anchor_divergence", "mean_downstream_anchor_divergence", "propagation_auc"]:
        negative_rows.append(
            {"experiment": "state_propagation", "query_region": "all", "metric": metric,
             "n": len(state_negative), "mean": state_negative[metric].mean(), "median": state_negative[metric].median()}
        )
    for query in QUERY_REGIONS:
        negative = s1b["branch_L40"][(s1b["branch_L40"]["query_region"] == query) & (s1b["branch_L40"]["local_effect_class"] == "negative")]
        for metric in ["delta_final_user_attention", "delta_assistant_history_attention", "delta_assistant_history_attention_normalized"]:
            negative_rows.append(
                {"experiment": "s1b", "query_region": query, "metric": metric,
                 "n": len(negative), "mean": negative[metric].mean(), "median": negative[metric].median()}
            )
    s3_negative = s3_wide_frame[s3_wide_frame["local_effect_class"] == "negative"]
    for metric in ["delta_direct", "abs_delta_direct", "delta_full", "abs_delta_full", "propagation_advantage"]:
        negative_rows.append(
            {"experiment": "s3_full", "query_region": "all", "metric": metric,
             "n": len(s3_negative), "mean": s3_negative[metric].mean(), "median": s3_negative[metric].median()}
        )
    pd.DataFrame(negative_rows).to_csv(args.output_dir / "negative_secondary_descriptives.csv", index=False)

    hardware_conclusion_rows = []
    for hardware_name in ["L40", "gpujl"]:
        state_overall_hardware = state[f"summary_{hardware_name}"]
        state_overall_hardware = state_overall_hardware[state_overall_hardware["layer_group"] == "all_layers"]
        for metric in ["final_user_anchor_divergence", "propagation_auc"]:
            hardware_conclusion_rows.append(
                {"experiment": "state_propagation", "hardware": hardware_name, "metric": metric,
                 **cluster_bootstrap_groups(state_overall_hardware, metric)}
            )
        for query in QUERY_REGIONS:
            subset = s1b[f"branch_{hardware_name}"]
            subset = subset[subset["query_region"] == query]
            for metric in ["delta_final_user_attention", "delta_assistant_history_attention", "delta_assistant_history_attention_normalized"]:
                hardware_conclusion_rows.append(
                    {"experiment": "s1b", "hardware": hardware_name, "metric": f"{query}:{metric}",
                     **cluster_bootstrap_groups(subset, metric)}
                )
        wide_hardware = s3_wide(full[hardware_name])
        for metric in ["delta_direct", "abs_delta_full", "propagation_advantage"]:
            hardware_conclusion_rows.append(
                {"experiment": "s3_full", "hardware": hardware_name, "metric": metric,
                 **cluster_bootstrap_groups(wide_hardware, metric)}
            )
    pd.DataFrame(hardware_conclusion_rows).to_csv(
        args.output_dir / "hardware_conclusion_comparison.csv", index=False
    )

    report = {
        "validation": validation,
        "hardware_reproducibility": hardware,
        "state_analysis": state_result,
        "state_within_rows": len(state_within),
        "s1b_analysis": s1b_result,
        "s3_rate_rows": len(s3_rates),
        "s3_within_rows": len(s3_within),
        "analysis_settings": {
            "bootstrap_reps": REPS,
            "bootstrap_cluster": "trajectory_id",
            "seed": SEED,
            "canonical_hardware": "L40",
            "normalized_profile_bins": ["(0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"],
        },
    }
    (args.output_dir / "open_model_analysis_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "valid_outputs": {key: value["valid"] for key, value in validation.items()}}, indent=2))


if __name__ == "__main__":
    main()

