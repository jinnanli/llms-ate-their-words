"""Freeze final paper-facing analyses for the completed open-model track."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPS = 5000
SEED = 20260820
KEY = ["trajectory_id", "target_turn"]
TASK_LABELS = {"actions": "Actions", "code": "Code", "data2text": "Data-to-Text"}
QUERY_REGIONS = ["first_token", "first_8", "all_generated"]


def average_ranks(values: pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])
    ranks = np.empty(len(array), dtype=float)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
    return ranks


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cluster_row_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    return [indices.to_numpy() for _, indices in frame.groupby("trajectory_id", sort=False).groups.items()]


def cluster_mean_ci(frame: pd.DataFrame, value: str, seed_offset: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + seed_offset)
    stats = frame.groupby("trajectory_id", sort=False)[value].agg(["sum", "count"]).to_numpy(dtype=float)
    sampled = rng.integers(0, len(stats), size=(REPS, len(stats)))
    totals = stats[sampled].sum(axis=1)
    draws = totals[:, 0] / totals[:, 1]
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def cluster_group_difference(
    frame: pd.DataFrame,
    value: str,
    group: str = "local_effect_class",
    left: str = "positive",
    right: str = "zero",
    seed_offset: int = 0,
) -> dict:
    data = frame[frame[group].isin([left, right])].copy()
    left_values = data.loc[data[group] == left, value]
    right_values = data.loc[data[group] == right, value]
    clusters = data["trajectory_id"].drop_duplicates().tolist()
    stats = []
    for cluster in clusters:
        subset = data[data["trajectory_id"] == cluster]
        left_subset = subset.loc[subset[group] == left, value]
        right_subset = subset.loc[subset[group] == right, value]
        stats.append((left_subset.sum(), len(left_subset), right_subset.sum(), len(right_subset)))
    stats = np.asarray(stats, dtype=float)
    rng = np.random.default_rng(SEED + seed_offset)
    sampled = rng.integers(0, len(stats), size=(REPS, len(stats)))
    totals = stats[sampled].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = totals[:, 0] / totals[:, 1] - totals[:, 2] / totals[:, 3]
    return {
        "positive_n": int(len(left_values)),
        "zero_n": int(len(right_values)),
        "positive_mean": float(left_values.mean()),
        "positive_median": float(left_values.median()),
        "zero_mean": float(right_values.mean()),
        "zero_median": float(right_values.median()),
        "difference": float(left_values.mean() - right_values.mean()),
        "ci_low": float(np.nanquantile(draws, 0.025)),
        "ci_high": float(np.nanquantile(draws, 0.975)),
    }


def spearman_value(frame: pd.DataFrame, x: str, y: str) -> float:
    clean = frame[[x, y]].dropna()
    if clean[x].nunique() < 2 or clean[y].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(average_ranks(clean[x]), average_ranks(clean[y]))[0, 1])


def cluster_spearman(frame: pd.DataFrame, x: str, y: str, seed_offset: int = 0) -> dict:
    data = frame[["trajectory_id", x, y]].dropna().copy()
    observed = spearman_value(data, x, y)
    if np.isnan(observed):
        return {
            "n": int(len(data)), "rho": None, "ci_low": None, "ci_high": None,
            "clusters": int(data["trajectory_id"].nunique()),
        }
    rng = np.random.default_rng(SEED + seed_offset)
    rows = cluster_row_indices(data.reset_index(drop=True))
    draws = []
    for sampled in rng.integers(0, len(rows), size=(REPS, len(rows))):
        indices = np.concatenate([rows[index] for index in sampled])
        sample = data.iloc[indices]
        draws.append(spearman_value(sample, x, y))
    draws = np.asarray(draws)
    return {
        "n": int(len(data)),
        "rho": observed,
        "ci_low": float(np.nanquantile(draws, 0.025)),
        "ci_high": float(np.nanquantile(draws, 0.975)),
        "clusters": int(data["trajectory_id"].nunique()),
    }


def ols_positive_coefficient(frame: pd.DataFrame, outcome: str) -> float:
    data = frame[frame["local_effect_class"].isin(["positive", "zero"])].copy()
    x = np.column_stack(
        [
            np.ones(len(data)),
            data["delta_assistant_tokens"].to_numpy(dtype=float),
            (data["local_effect_class"] == "positive").to_numpy(dtype=float),
        ]
    )
    y = data[outcome].to_numpy(dtype=float)
    return float(np.linalg.lstsq(x, y, rcond=None)[0][2])


def cluster_regression(frame: pd.DataFrame, outcome: str, seed_offset: int = 0) -> dict:
    data = frame[frame["local_effect_class"].isin(["positive", "zero"])].copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED + seed_offset)
    rows = cluster_row_indices(data)
    draws = []
    for sampled in rng.integers(0, len(rows), size=(REPS, len(rows))):
        indices = np.concatenate([rows[index] for index in sampled])
        draws.append(ols_positive_coefficient(data.iloc[indices], outcome))
    draws = np.asarray(draws)
    return {
        "n": int(len(data)),
        "clusters": int(data["trajectory_id"].nunique()),
        "positive_indicator_coefficient": ols_positive_coefficient(data, outcome),
        "ci_low": float(np.nanquantile(draws, 0.025)),
        "ci_high": float(np.nanquantile(draws, 0.975)),
        "covariate": "delta_assistant_tokens",
    }


def s3_frame(path: Path) -> pd.DataFrame:
    rows = pd.DataFrame(read_jsonl(path))
    base = rows[rows["condition"] == "baseline"][KEY + ["task", "local_effect_class", "local_delta", "evaluator_score"]]
    base = base.rename(columns={"evaluator_score": "baseline_score"})
    direct = rows[rows["condition"] == "direct_only"][KEY + ["evaluator_score"]].rename(
        columns={"evaluator_score": "direct_score"}
    )
    full = rows[rows["condition"] == "full_downstream"][KEY + ["evaluator_score"]].rename(
        columns={"evaluator_score": "full_score"}
    )
    data = base.merge(direct, on=KEY, validate="one_to_one").merge(full, on=KEY, validate="one_to_one")
    data["delta_direct"] = data["direct_score"] - data["baseline_score"]
    data["delta_full"] = data["full_score"] - data["baseline_score"]
    data["propagation_advantage"] = data["delta_full"].abs() - data["delta_direct"].abs()
    return data


def s3_analysis(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    result: dict = {
        "correlations": {}, "task_correlations": {}, "sign_agreement": {},
        "class_pattern": {}, "tasks": {}, "leave_one_out": {},
    }
    subsets = {"all": data, "local_nonzero": data[data["local_effect_class"] != "zero"]}
    for name, subset in subsets.items():
        result["correlations"][name] = {
            "direct": cluster_spearman(subset, "local_delta", "delta_direct", seed_offset=10),
            "full": cluster_spearman(subset, "local_delta", "delta_full", seed_offset=11),
            "direct_mean_absolute_deviation": float((subset["delta_direct"] - subset["local_delta"]).abs().mean()),
            "full_mean_absolute_deviation": float((subset["delta_full"] - subset["local_delta"]).abs().mean()),
        }

    nonzero = subsets["local_nonzero"].copy()
    for condition in ["direct", "full"]:
        effect = nonzero[f"delta_{condition}"]
        local_sign = np.sign(nonzero["local_delta"])
        effect_sign = np.sign(effect)
        positive = nonzero["local_effect_class"] == "positive"
        negative = nonzero["local_effect_class"] == "negative"
        result["sign_agreement"][condition] = {
            "n": int(len(nonzero)),
            "agreement_n": int((local_sign == effect_sign).sum()),
            "agreement_rate": float((local_sign == effect_sign).mean()),
            "zero_knockout_n": int((effect_sign == 0).sum()),
            "positive_surgery_n": int(positive.sum()),
            "positive_surgery_effect_positive_n": int((effect[positive] > 0).sum()),
            "positive_surgery_effect_positive_fraction": float((effect[positive] > 0).mean()),
            "negative_surgery_n": int(negative.sum()),
            "negative_surgery_effect_negative_n": int((effect[negative] < 0).sum()),
            "negative_surgery_effect_negative_fraction": float((effect[negative] < 0).mean()),
        }

    for class_name in ["positive", "zero", "negative"]:
        subset = data[data["local_effect_class"] == class_name]
        ci_low, ci_high = cluster_mean_ci(subset, "delta_direct", seed_offset=20)
        result["class_pattern"][class_name] = {
            "n": int(len(subset)),
            "mean": float(subset["delta_direct"].mean()),
            "median": float(subset["delta_direct"].median()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "positive_fraction": float((subset["delta_direct"] > 0).mean()),
            "zero_fraction": float((subset["delta_direct"] == 0).mean()),
            "negative_fraction": float((subset["delta_direct"] < 0).mean()),
        }

    task_rows = []
    for task in ["actions", "code", "data2text"]:
        subset = data[data["task"] == task]
        comparison = cluster_group_difference(subset, "delta_direct", seed_offset=30)
        result["tasks"][task] = comparison
        result["task_correlations"][task] = {
            "direct": cluster_spearman(subset, "local_delta", "delta_direct", seed_offset=32),
            "full": cluster_spearman(subset, "local_delta", "delta_full", seed_offset=33),
        }
        for class_name in ["positive", "zero", "negative"]:
            group = subset[subset["local_effect_class"] == class_name]
            if group.empty:
                continue
            mean_ci = cluster_mean_ci(group, "delta_direct", seed_offset=31)
            binary = task in {"actions", "code"}
            task_rows.append(
                {
                    "task": TASK_LABELS[task],
                    "local_class": class_name,
                    "n": len(group),
                    "mean_delta_direct": group["delta_direct"].mean(),
                    "median_delta_direct": group["delta_direct"].median(),
                    "mean_ci_low": mean_ci[0],
                    "mean_ci_high": mean_ci[1],
                    "direct_score_change_rate": (group["delta_direct"] != 0).mean(),
                    "full_score_change_rate": (group["delta_full"] != 0).mean(),
                    "direct_binary_outcome_flip_rate": (
                        (group["direct_score"] != group["baseline_score"]).mean() if binary else np.nan
                    ),
                    "full_binary_outcome_flip_rate": (
                        (group["full_score"] != group["baseline_score"]).mean() if binary else np.nan
                    ),
                }
            )

    combinations = {
        "Actions+Code": ["actions", "code"],
        "Actions+Data-to-Text": ["actions", "data2text"],
        "Code+Data-to-Text": ["code", "data2text"],
    }
    for name, tasks in combinations.items():
        result["leave_one_out"][name] = cluster_group_difference(
            data[data["task"].isin(tasks)], "delta_direct", seed_offset=40
        )
    return result, pd.DataFrame(task_rows)


def s1b_analysis(branch_path: Path, span_path: Path) -> tuple[dict, pd.DataFrame]:
    branch = pd.read_csv(branch_path)
    spans = pd.DataFrame(read_jsonl(span_path))
    span_rows = []
    for row in spans.to_dict("records"):
        original = row["original_segment_counts"]
        surgery = row["surgery_segment_counts"]
        original_assistant = original["target_assistant_turn"] + original["other_assistant_history"]
        surgery_assistant = surgery["target_assistant_turn"] + surgery["other_assistant_history"]
        span_rows.append(
            {
                "trajectory_id": row["trajectory_id"],
                "target_turn": row["target_turn"],
                "assistant_tokens_original": original_assistant,
                "assistant_tokens_surgery": surgery_assistant,
                "delta_assistant_tokens": surgery_assistant - original_assistant,
                "original_prompt_tokens": row["original_prompt_tokens"],
                "surgery_prompt_tokens": row["surgery_prompt_tokens"],
            }
        )
    tokens = pd.DataFrame(span_rows)
    merged = branch.merge(tokens, on=KEY, validate="many_to_one")
    result: dict = {"token_counts": {}, "query_regions": {}, "denominator_decision": {}}

    one_per_branch = merged[merged["query_region"] == "all_generated"].copy()
    for class_name in ["positive", "zero", "negative"]:
        subset = one_per_branch[one_per_branch["local_effect_class"] == class_name]
        result["token_counts"][class_name] = {
            "n": int(len(subset)),
            "original_mean": float(subset["assistant_tokens_original"].mean()),
            "original_median": float(subset["assistant_tokens_original"].median()),
            "surgery_mean": float(subset["assistant_tokens_surgery"].mean()),
            "surgery_median": float(subset["assistant_tokens_surgery"].median()),
            "delta_mean": float(subset["delta_assistant_tokens"].mean()),
            "delta_median": float(subset["delta_assistant_tokens"].median()),
        }
    result["token_count_positive_vs_zero"] = cluster_group_difference(
        one_per_branch, "delta_assistant_tokens", seed_offset=50
    )

    for index, query_region in enumerate(QUERY_REGIONS):
        subset = merged[merged["query_region"] == query_region].copy()
        result["query_regions"][query_region] = {
            "token_attention_spearman": cluster_spearman(
                subset, "delta_assistant_tokens", "delta_assistant_history_attention", seed_offset=60 + index
            ),
            "unadjusted_total_attention": cluster_group_difference(
                subset, "delta_assistant_history_attention", seed_offset=70 + index
            ),
            "per_token_attention": cluster_group_difference(
                subset, "delta_assistant_history_attention_normalized", seed_offset=80 + index
            ),
            "token_adjusted_regression": cluster_regression(
                subset, "delta_assistant_history_attention", seed_offset=90 + index
            ),
        }

    result["denominator_decision"] = {
        "enrichment_computed": False,
        "reason": (
            "Eligible key count changes during autoregressive generation, so a single prompt-token denominator does not "
            "exactly represent token availability across first_token, first_8, and all_generated queries."
        ),
        "fallback": "Branch-level OLS with delta_assistant_tokens and local-positive indicator; trajectory-cluster bootstrap.",
    }
    return result, one_per_branch


def add_number(
    rows: list[dict], experiment: str, analysis: str, population: str, metric: str, group_a: str,
    group_b: str, estimate: float, ci_low: float | None, ci_high: float | None, n: str,
    priority: str, notes: str,
) -> None:
    rows.append(
        {
            "experiment": experiment,
            "analysis": analysis,
            "population": population,
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
            "paper_priority": priority,
            "notes": notes,
        }
    )


def number_lookup(s3: dict, s1b: dict) -> pd.DataFrame:
    rows: list[dict] = []
    overall = s3["tasks"]["data2text"]  # Added below with explicit overall result.
    del overall
    # S3 paper-facing statistics.
    all_s3 = cluster_group_difference(S3_DATA, "delta_direct", seed_offset=100)
    add_number(rows, "S3", "full cohort", "131 branches/24 trajectories", "signed Delta_direct",
               "local positive", "local zero", all_s3["difference"], 0.00007373719205514685, 0.0758038324386708,
               "26 vs 93", "SPACE_DEPENDENT", "Frozen trajectory-cluster bootstrap CI.")
    for scope in ["all", "local_nonzero"]:
        for condition in ["direct", "full"]:
            value = s3["correlations"][scope][condition]
            add_number(rows, "S3", f"directional alignment ({scope})", scope, "Spearman local_delta vs knockout delta",
                       "local surgery", condition, value["rho"], value["ci_low"], value["ci_high"], str(value["n"]),
                       "SPACE_DEPENDENT", "Cluster-bootstrap CI.")
    for name, value in s3["leave_one_out"].items():
        add_number(rows, "S3", "leave-one-task-out", name, "signed Delta_direct difference",
                   "local positive", "local zero", value["difference"], value["ci_low"], value["ci_high"],
                   f"{value['positive_n']} vs {value['zero_n']}", "SPACE_DEPENDENT", "Trajectory-cluster bootstrap.")
    for task, value in s3["tasks"].items():
        add_number(rows, "S3", "task breakdown", TASK_LABELS[task], "signed Delta_direct difference",
                   "local positive", "local zero", value["difference"], value["ci_low"], value["ci_high"],
                   f"{value['positive_n']} vs {value['zero_n']}", "SPACE_DEPENDENT", "Trajectory-cluster bootstrap.")
    for condition, value in s3["sign_agreement"].items():
        add_number(rows, "S3", "directional sign agreement", "38 local nonzero branches",
                   "sign agreement rate", "local surgery", condition, value["agreement_rate"], None, None,
                   str(value["n"]), "SPACE_DEPENDENT", f"Zero knockout effects: {value['zero_knockout_n']}.")

    # S1b paper-facing statistics.
    for region in QUERY_REGIONS:
        values = s1b["query_regions"][region]
        for analysis, key, priority in [
            ("unadjusted total attention", "unadjusted_total_attention", "CORE_CANDIDATE"),
            ("per-token attention", "per_token_attention", "SPACE_DEPENDENT"),
            ("token-count-adjusted regression", "token_adjusted_regression", "CORE_CANDIDATE"),
        ]:
            value = values[key]
            estimate_key = "positive_indicator_coefficient" if key == "token_adjusted_regression" else "difference"
            low_key = "ci_low"
            high_key = "ci_high"
            add_number(rows, "S1b", f"{analysis} ({region})", "131 branches/24 trajectories",
                       "assistant-history attention shift", "local positive", "local zero", value[estimate_key],
                       value[low_key], value[high_key], "26 vs 93", priority,
                       "Adjusted model includes assistant-token-count shift." if key == "token_adjusted_regression" else "")
        correlation = values["token_attention_spearman"]
        add_number(rows, "S1b", f"token-count diagnostic ({region})", "131 branches/24 trajectories",
                   "Spearman Delta tokens vs Delta total attention", "all branches", "", correlation["rho"],
                   correlation["ci_low"], correlation["ci_high"], str(correlation["n"]), "SPACE_DEPENDENT",
                   "Trajectory-cluster bootstrap.")
    token = s1b["token_count_positive_vs_zero"]
    add_number(rows, "S1b", "token-count audit", "131 branches/24 trajectories", "Delta assistant tokens",
               "local positive", "local zero", token["difference"], token["ci_low"], token["ci_high"],
               "26 vs 93", "SPACE_DEPENDENT", "Surgery minus original assistant content tokens.")

    # Previously frozen diagnostic results.
    add_number(rows, "S1", "static attention", "131 local branches", "normalized target-turn attention",
               "local positive", "local zero", -5.6253e-05, -0.00035941, 0.00027036, "26 vs 93",
               "SPACE_DEPENDENT", "Frozen prior analysis.")
    add_number(rows, "S2", "contrastive margin", "native fail-to-success vs matched zero",
               "between-group Delta M", "fail-to-success", "matched zero", -0.05095, -1.10985, 1.05996,
               "5 vs 5", "SPACE_DEPENDENT", "Frozen native S2 analysis; 10 scored rows.")
    add_number(rows, "State Propagation", "population AUC", "131 branches/24 trajectories", "propagation AUC",
               "local positive", "local zero", -0.00526045, -0.01055873, 0.00008137, "26 vs 93",
               "SPACE_DEPENDENT", "Direction conflicts with within-trajectory analysis.")
    return pd.DataFrame(rows)


def cross_track_keys(state_path: Path, s1_path: Path) -> pd.DataFrame:
    state = pd.read_csv(state_path)
    s1 = pd.read_csv(s1_path)
    metadata = s1[KEY + ["relative_position", "n_total_turns"]].drop_duplicates(KEY)
    frame = state.merge(metadata, on=KEY, validate="one_to_one")
    frame["formal_delta"] = frame["formal_surgery_score"] - frame["formal_original_score"]

    primary = frame[
        frame["task"].isin(["actions", "code"])
        & frame["local_original_score"].eq(0.0)
        & frame["local_surgery_score"].eq(1.0)
    ].sort_values(["task", "trajectory_id", "target_turn"])
    zeros = frame[frame["local_delta"].abs() <= 1e-12]
    controls = []
    used: set[tuple[str, int]] = set()
    for row in primary.itertuples(index=False):
        candidates = zeros[
            zeros["task"].eq(row.task)
            & ~zeros.apply(lambda item: (item["trajectory_id"], int(item["target_turn"])) in used, axis=1)
        ].copy()
        candidates["position_distance"] = (candidates["relative_position"] - row.relative_position).abs()
        candidates["length_distance"] = (candidates["n_total_turns"] - row.n_total_turns).abs()
        control = candidates.sort_values(
            ["position_distance", "length_distance", "trajectory_id", "target_turn"]
        ).iloc[0]
        key = (control["trajectory_id"], int(control["target_turn"]))
        used.add(key)
        controls.append(key)
    s2_keys = set(zip(primary["trajectory_id"], primary["target_turn"].astype(int))) | set(controls)
    s1_keys = set(zip(s1["trajectory_id"], s1["target_turn"].astype(int)))

    output = frame[KEY + ["task", "formal_effect_class", "formal_delta", "local_effect_class", "local_delta"]].copy()
    keys = list(zip(output["trajectory_id"], output["target_turn"].astype(int)))
    output["has_s1"] = [key in s1_keys for key in keys]
    output["has_s1b"] = True
    output["has_s2"] = [key in s2_keys for key in keys]
    output["has_state_propagation"] = True
    output["has_s3"] = True
    output["native_cohort"] = True
    return output.sort_values(KEY).reset_index(drop=True)


def figure_data() -> pd.DataFrame:
    columns = [
        "panel", "probe", "question_short", "primary_metric", "estimate", "ci_low", "ci_high",
        "result_direction", "support_status", "task_generality", "recommended_display_priority", "notes",
    ]
    rows = [
        ["A", "Cohort", "Model revision", "revision", "40c069824f4251a91eefaf281ebe4c544efd3e18", "", "", "metadata", "DIAGNOSTIC_ONLY", "Actions/Code/Data-to-Text", "CORE_CANDIDATE", "Qwen/Qwen3-14B"],
        ["A", "Cohort", "Native trajectories", "count", 24, "", "", "metadata", "DIAGNOSTIC_ONLY", "three tasks", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Native branches", "count", 131, "", "", "metadata", "DIAGNOSTIC_ONLY", "three tasks", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Actions branches", "count", 35, "", "", "metadata", "DIAGNOSTIC_ONLY", "Actions", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Code branches", "count", 53, "", "", "metadata", "DIAGNOSTIC_ONLY", "Code", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Data-to-Text branches", "count", 43, "", "", "metadata", "DIAGNOSTIC_ONLY", "Data-to-Text", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Local positive branches", "count", 26, "", "", "metadata", "DIAGNOSTIC_ONLY", "three tasks", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Local zero branches", "count", 93, "", "", "metadata", "DIAGNOSTIC_ONLY", "three tasks", "CORE_CANDIDATE", ""],
        ["A", "Cohort", "Local negative branches", "count", 12, "", "", "metadata", "DIAGNOSTIC_ONLY", "Data-to-Text", "CORE_CANDIDATE", ""],
        ["A", "Calibration", "Formal/local effect-class agreement", "agreement rate", 91 / 131, "", "", "moderate", "DIAGNOSTIC_ONLY", "native cohort", "CORE_CANDIDATE", "91/131 = 69.5%"],
        ["A", "Calibration", "Binary reversal agreement", "agreement rate", 80 / 88, "", "", "high", "DIAGNOSTIC_ONLY", "Actions/Code", "CORE_CANDIDATE", "80/88 = 90.9%"],
        ["B", "S1", "Static target-turn attention", "normalized TTA difference", -5.6253e-05, -0.00035941, 0.00027036, "null", "NULL", "native cohort", "SPACE_DEPENDENT", "local positive minus zero"],
        ["B", "S1b", "Aggregate assistant-history reallocation", "all-generated total-attention shift difference", -0.005192172302073991, -0.010612504936911245, -0.0007334236788174918, "more negative for positive", "SUPPORTED_WITH_CAVEAT", "native cohort", "CORE_CANDIDATE", "local positive minus zero"],
        ["B", "S1b", "Token-adjusted aggregate reallocation", "local-positive regression coefficient", -0.00521753747080894, -0.009984990993204965, -0.0009723043254420312, "more negative for positive", "SUPPORTED_WITH_CAVEAT", "native cohort", "CORE_CANDIDATE", "controls delta assistant tokens"],
        ["B", "S1b", "Per-token assistant-history reallocation", "per-token shift difference", -5.6439769434578585e-05, -0.00016090401835055625, 0.00002104927611543053, "null", "NULL", "native cohort", "CORE_CANDIDATE", "required caveat"],
        ["B", "S1b", "Final-user attention reallocation", "all-generated shift difference", -0.0036070600724754306, -0.009164714682332984, 0.001730836528726143, "null", "NULL", "native cohort", "SPACE_DEPENDENT", "secondary"],
        ["C", "S1", "Does static target attention mark influence?", "normalized TTA difference", -5.6253e-05, -0.00035941, 0.00027036, "null", "NULL", "native cohort", "SPACE_DEPENDENT", "diagnostic boundary"],
        ["C", "S1b", "Does surgery change aggregate history allocation?", "token-adjusted coefficient", -0.00521753747080894, -0.009984990993204965, -0.0009723043254420312, "larger reduction", "SUPPORTED_WITH_CAVEAT", "not a cross-task mechanism claim", "CORE_CANDIDATE", "aggregate, not per-token"],
        ["C", "S2", "Is margin movement influence-specific?", "between-group Delta M", -0.050948371333671805, -1.10984646767717, 1.0599568863226496, "null/non-specific", "DIAGNOSTIC_ONLY", "Actions/Code", "SPACE_DEPENDENT", "5 vs 5 native branches"],
        ["C", "State Propagation", "Is persistent divergence influence-specific?", "population propagation-AUC difference", -0.00526044961266731, -0.010558729703233684, 0.00008136843303764873, "population negative; paired positive", "DIAGNOSTIC_ONLY", "native cohort", "SPACE_DEPENDENT", "direction conflict"],
        ["C", "S3 direct-only", "Does direct target access reproduce influence?", "signed score-effect difference", 0.025617968130115944, 0.00007373719205514685, 0.0758038324386708, "positive but sparse", "DIAGNOSTIC_ONLY", "Data-to-Text-driven", "SPACE_DEPENDENT", "not cross-task"],
        ["C", "S3 full-downstream", "Is downstream propagation selective?", "propagation-advantage difference", 0.07608658339966433, -0.012326109123744117, 0.2536574323154997, "positive point estimate; null CI", "DIAGNOSTIC_ONLY", "not established", "SPACE_DEPENDENT", "better directional alignment only"],
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    root = Path("mechanistic_analysis_results/retrieval_20260820_v1")
    output = root / "analysis" / "open_model_final_freeze"
    output.mkdir(parents=True, exist_ok=True)

    global S3_DATA
    S3_DATA = s3_frame(root / "raw/s3_full/L40/s3_knockout_results.jsonl")
    s3, task_breakdown = s3_analysis(S3_DATA)
    s1b, token_audit = s1b_analysis(
        root / "raw/s1b/L40/s1b_branch_summary.csv",
        root / "raw/s1b/L40/token_spans.jsonl",
    )

    frozen = {
        "analysis_version": "open_model_final_freeze_20260820_v1",
        "canonical_hardware": "L40",
        "bootstrap": {"repetitions": REPS, "seed": SEED, "cluster": "trajectory_id"},
        "population": {"trajectories": 24, "branches": 131, "positive": 26, "zero": 93, "negative": 12},
        "s3": s3,
        "s1b": s1b,
        "s1b_final_classification": "TOKEN_ADJUSTED_AGGREGATE_REALLOCATION_SUPPORT",
    }
    (output / "OPEN_MODEL_FINAL_ANALYSIS.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    task_breakdown.to_csv(output / "OPEN_MODEL_TASK_BREAKDOWN.csv", index=False)
    token_audit.to_csv(output / "OPEN_MODEL_S1B_TOKEN_AUDIT.csv", index=False)
    number_lookup(s3, s1b).to_csv(output / "OPEN_MODEL_NUMBER_LOOKUP.csv", index=False)
    cross_track_keys(
        Path("mechanistic_precompute_inputs/state_propagation_manifest.csv"),
        Path("mechanistic_precompute_inputs/s1_qwen_manifest.csv"),
    ).to_csv(output / "OPEN_MODEL_CROSS_TRACK_KEYS.csv", index=False)
    figure_data().to_csv(output / "OPEN_MODEL_FIGURE_DATA.csv", index=False)
    print(json.dumps(frozen, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

