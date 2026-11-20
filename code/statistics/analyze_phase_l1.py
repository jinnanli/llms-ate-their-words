"""Phase L1: offline structural analysis of frozen Turn Surgery branches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "rq3" / "turn_surgery_branches_compact.csv"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "phase_l1"
PRACTICAL_EPSILON = 0.01
BINARY_TASKS = {"actions", "code", "database", "math"}
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260819
MODEL_ORDER = [
    "gpt-5.6-luna",
    "grok-4.5",
    "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-14B",
]
TASK_ORDER = ["actions", "code", "data2text", "database", "math", "summary"]


def quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
    }


def classify(delta: pd.Series, task: pd.Series, epsilon: float) -> pd.Series:
    effective_epsilon = np.where(task.isin(BINARY_TASKS), 0.0, epsilon)
    return pd.Series(
        np.where(delta > effective_epsilon, "positive", np.where(delta < -effective_epsilon, "negative", "zero")),
        index=delta.index,
    )


def topology(n_positive: int, n_negative: int) -> str:
    if n_positive == 0 and n_negative == 0:
        return "insensitive"
    if n_positive > 0 and n_negative == 0:
        return "positive_only"
    if n_positive == 0 and n_negative > 0:
        return "negative_only"
    return "mixed"


def fit_logistic(frame: pd.DataFrame, outcome: str, adjusted: bool) -> dict:
    formula = f"{outcome} ~ relative_position"
    if adjusted:
        formula += " + C(model) + C(task)"
    fit = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": frame["trajectory_id"]}
    )
    beta = float(fit.params["relative_position"])
    se = float(fit.bse["relative_position"])
    ci_low, ci_high = fit.conf_int().loc["relative_position"].tolist()
    return {
        "formula": formula,
        "coefficient": beta,
        "cluster_se": se,
        "odds_ratio_per_full_position_range": math.exp(beta),
        "odds_ratio_95ci": [math.exp(float(ci_low)), math.exp(float(ci_high))],
        "p_value": float(fit.pvalues["relative_position"]),
        "odds_ratio_per_0_1_position": math.exp(beta * 0.1),
        "odds_ratio_per_0_1_position_95ci": [
            math.exp(float(ci_low) * 0.1), math.exp(float(ci_high) * 0.1)
        ],
        "clusters": int(frame["trajectory_id"].nunique()),
        "branches": int(len(frame)),
    }


def clustered_spearman_ci(frame: pd.DataFrame, x: str, y: str) -> dict:
    observed = stats.spearmanr(frame[x], frame[y])
    grouped = {key: value[[x, y]].to_numpy() for key, value in frame.groupby("trajectory_id")}
    trajectory_ids = np.array(list(grouped), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choice(trajectory_ids, size=len(trajectory_ids), replace=True)
        values = np.concatenate([grouped[key] for key in sampled], axis=0)
        rho = stats.spearmanr(values[:, 0], values[:, 1]).statistic
        if np.isfinite(rho):
            draws.append(float(rho))
    return {
        "rho": float(observed.statistic),
        "naive_p_value": float(observed.pvalue),
        "trajectory_cluster_bootstrap_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_replicates": len(draws),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def bin_summary(frame: pd.DataFrame, summary_type: str, bin_column: str, order: list[str]) -> list[dict]:
    rows: list[dict] = []
    for bin_label in order:
        cell = frame[frame[bin_column].astype(str) == bin_label]
        if cell.empty:
            continue
        rows.append(
            {
                "summary_type": summary_type,
                "bin": bin_label,
                "n_branches": int(len(cell)),
                "n_trajectories": int(cell["trajectory_id"].nunique()),
                "mean_relative_position": float(cell["relative_position"].mean()),
                "mean_distance_to_final": float(cell["distance_to_final"].mean()),
                "positive_rate": float((cell["effect_class"] == "positive").mean()),
                "zero_rate": float((cell["effect_class"] == "zero").mean()),
                "negative_rate": float((cell["effect_class"] == "negative").mean()),
                "robust_positive_rate": float((cell["effect_class_robust"] == "positive").mean()),
                "robust_zero_rate": float((cell["effect_class_robust"] == "zero").mean()),
                "robust_negative_rate": float((cell["effect_class_robust"] == "negative").mean()),
                "mean_delta": float(cell["delta"].mean()),
                "mean_abs_delta": float(cell["abs_delta"].mean()),
            }
        )
    return rows


def make_trajectory_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for trajectory_id, group in frame.groupby("trajectory_id", sort=True):
        first = group.iloc[0]
        counts = group["effect_class"].value_counts()
        robust_counts = group["effect_class_robust"].value_counts()
        n_valid = len(group)
        n_planned = int(first["n_intermediate_turns"])
        abs_sum = float(group["abs_delta"].sum())
        max_positive = group.loc[group["delta"] > 0, "delta"].max()
        n_positive = int(counts.get("positive", 0))
        n_negative = int(counts.get("negative", 0))
        robust_positive = int(robust_counts.get("positive", 0))
        robust_negative = int(robust_counts.get("negative", 0))
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "model": first["model"],
                "task": first["task"],
                "n_planned_targets": n_planned,
                "n_valid_targets": n_valid,
                "n_missing_targets": n_planned - n_valid,
                "n_positive": n_positive,
                "n_zero": int(counts.get("zero", 0)),
                "n_negative": n_negative,
                "positive_fraction": n_positive / n_valid,
                "negative_fraction": n_negative / n_valid,
                "mean_delta": float(group["delta"].mean()),
                "max_positive_delta": "" if pd.isna(max_positive) else float(max_positive),
                "max_abs_delta": float(group["abs_delta"].max()),
                "sum_abs_delta": abs_sum,
                "concentration": float(group["abs_delta"].max() / abs_sum) if abs_sum > 0 else "",
                "sign_topology": topology(n_positive, n_negative),
                "n_positive_robust": robust_positive,
                "n_zero_robust": int(robust_counts.get("zero", 0)),
                "n_negative_robust": robust_negative,
                "positive_fraction_robust": robust_positive / n_valid,
                "negative_fraction_robust": robust_negative / n_valid,
                "sign_topology_robust": topology(robust_positive, robust_negative),
            }
        )
    return pd.DataFrame(rows)


def count_distribution(summary: pd.DataFrame, column: str) -> dict:
    values = summary[column]
    counts = {
        "zero": int((values == 0).sum()),
        "one": int((values == 1).sum()),
        "two": int((values == 2).sum()),
        "three_or_more": int((values >= 3).sum()),
    }
    return {
        key: {"count": value, "proportion": value / len(summary)} for key, value in counts.items()
    }


def topology_distribution(summary: pd.DataFrame, column: str) -> dict:
    counts = summary[column].value_counts()
    return {
        label: {"count": int(counts.get(label, 0)), "proportion": float((summary[column] == label).mean())}
        for label in ["insensitive", "positive_only", "negative_only", "mixed"]
    }


def missingness_bounds(summary: pd.DataFrame) -> dict:
    incomplete = summary[summary["n_missing_targets"] > 0]
    observed_any = int((summary["n_positive"] > 0).sum())
    any_upper_add = int((incomplete["n_positive"] == 0).sum())
    observed_multiple = int((summary["n_positive"] >= 2).sum())
    can_reach_multiple = incomplete[
        (incomplete["n_positive"] < 2)
        & (incomplete["n_positive"] + incomplete["n_missing_targets"] >= 2)
    ]

    observed_topology = topology_distribution(summary, "sign_topology")
    complete = summary[summary["n_missing_targets"] == 0]
    complete_counts = complete["sign_topology"].value_counts()
    incomplete_counts = incomplete["sign_topology"].value_counts()
    insensitive_two_missing = int(
        ((incomplete["sign_topology"] == "insensitive") & (incomplete["n_missing_targets"] >= 2)).sum()
    )
    topology_bounds = {
        "insensitive": {
            "lower": int(complete_counts.get("insensitive", 0)),
            "observed": observed_topology["insensitive"]["count"],
            "upper": observed_topology["insensitive"]["count"],
        },
        "positive_only": {
            "lower": int(complete_counts.get("positive_only", 0)),
            "observed": observed_topology["positive_only"]["count"],
            "upper": observed_topology["positive_only"]["count"] + int(incomplete_counts.get("insensitive", 0)),
        },
        "negative_only": {
            "lower": int(complete_counts.get("negative_only", 0)),
            "observed": observed_topology["negative_only"]["count"],
            "upper": observed_topology["negative_only"]["count"] + int(incomplete_counts.get("insensitive", 0)),
        },
        "mixed": {
            "lower": observed_topology["mixed"]["count"],
            "observed": observed_topology["mixed"]["count"],
            "upper": (
                observed_topology["mixed"]["count"]
                + int(incomplete_counts.get("positive_only", 0))
                + int(incomplete_counts.get("negative_only", 0))
                + insensitive_two_missing
            ),
        },
    }
    return {
        "incomplete_trajectories": int(len(incomplete)),
        "missing_branches": int(incomplete["n_missing_targets"].sum()),
        "any_positive_observed_lower": observed_any,
        "any_positive_worst_case_upper": observed_any + any_upper_add,
        "multiple_positive_observed_lower": observed_multiple,
        "multiple_positive_worst_case_upper": observed_multiple + int(len(can_reach_multiple)),
        "topology_count_bounds": topology_bounds,
    }


def model_task_summary(frame: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            branches = frame[(frame["model"] == model) & (frame["task"] == task)]
            tr = trajectories[(trajectories["model"] == model) & (trajectories["task"] == task)]
            concentrations = pd.to_numeric(tr["concentration"], errors="coerce").dropna()
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "n_trajectories": len(tr),
                    "n_valid_branches": len(branches),
                    "n_planned_branches": int(tr["n_planned_targets"].sum()),
                    "n_missing_branches": int(tr["n_missing_targets"].sum()),
                    "positive_rate": float((branches["effect_class"] == "positive").mean()),
                    "negative_rate": float((branches["effect_class"] == "negative").mean()),
                    "robust_positive_rate": float((branches["effect_class_robust"] == "positive").mean()),
                    "robust_negative_rate": float((branches["effect_class_robust"] == "negative").mean()),
                    "any_positive_trajectory_rate": float((tr["n_positive"] > 0).mean()),
                    "robust_any_positive_trajectory_rate": float((tr["n_positive_robust"] > 0).mean()),
                    "mixed_trajectory_rate": float((tr["sign_topology"] == "mixed").mean()),
                    "robust_mixed_trajectory_rate": float((tr["sign_topology_robust"] == "mixed").mean()),
                    "median_concentration": "" if concentrations.empty else float(concentrations.median()),
                    "n_concentration_defined": len(concentrations),
                    "small_cell": "true" if len(tr) < 8 else "false",
                }
            )
    return pd.DataFrame(rows)


def make_figures(position: pd.DataFrame, trajectories: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    deciles = position[position["summary_type"] == "position_decile"].copy()
    x = np.arange(len(deciles))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
    axes[0].plot(x, deciles["positive_rate"], marker="o", label="Positive")
    axes[0].plot(x, deciles["negative_rate"], marker="s", label="Negative")
    axes[0].set_xticks(x, deciles["bin"], rotation=45, ha="right")
    axes[0].set_ylabel("Valid-branch probability")
    axes[0].set_title("Outcome class by normalized position")
    axes[0].legend(frameon=False)
    axes[1].plot(x, deciles["mean_delta"], marker="o", label="Mean delta")
    axes[1].plot(x, deciles["mean_abs_delta"], marker="s", label="Mean |delta|")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, deciles["bin"], rotation=45, ha="right")
    axes[1].set_title("Effect magnitude by normalized position")
    axes[1].legend(frameon=False)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"exploratory_position_profile.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    count_dist = count_distribution(trajectories, "n_positive")
    sign_dist = topology_distribution(trajectories, "sign_topology")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35), constrained_layout=True)
    labels = ["0", "1", "2", "3+"]
    vals = [count_dist[k]["proportion"] for k in ["zero", "one", "two", "three_or_more"]]
    axes[0].bar(labels, vals, color="#52789c")
    axes[0].set_xlabel("Positive target turns per trajectory")
    axes[0].set_ylabel("Trajectory proportion")
    topology_labels = ["Insensitive", "Positive-only", "Negative-only", "Mixed"]
    topology_vals = [sign_dist[k]["proportion"] for k in ["insensitive", "positive_only", "negative_only", "mixed"]]
    axes[1].bar(topology_labels, topology_vals, color=["#999999", "#4c956c", "#b56576", "#8064a2"])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Sign topology")
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"exploratory_trajectory_topology.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    concentration = pd.to_numeric(trajectories["concentration"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(3.7, 2.45), constrained_layout=True)
    ax.hist(concentration, bins=np.linspace(0, 1, 21), color="#52789c", edgecolor="white")
    ax.axvline(concentration.median(), color="#b24c45", linestyle="--", label=f"Median={concentration.median():.2f}")
    ax.set_xlabel("Top-1 absolute-effect concentration")
    ax.set_ylabel("Trajectories")
    ax.legend(frameon=False)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"exploratory_influence_concentration.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {
        "trajectory_id", "model", "task", "target_turn", "n_intermediate_turns",
        "relative_position", "distance_to_final", "delta", "effect_class",
    }
    if required - set(frame.columns):
        raise RuntimeError(f"Missing canonical fields: {sorted(required - set(frame.columns))}")
    if len(frame) != 1440 or frame["trajectory_id"].nunique() != 237:
        raise RuntimeError("Frozen branch/trajectory population mismatch")
    if frame.duplicated(["trajectory_id", "target_turn"]).any():
        raise RuntimeError("Duplicate branch primary key")
    if not np.allclose(frame["delta"], frame["surgery_score"] - frame["original_score"], atol=1e-12, rtol=0):
        raise RuntimeError("Stored delta differs from surgery minus SHARDED score")
    exact = classify(frame["delta"], frame["task"], 0.0)
    if not (exact == frame["effect_class"]).all():
        raise RuntimeError("Frozen effect_class does not match exact-sign contract")
    frame["effect_class_robust"] = classify(frame["delta"], frame["task"], PRACTICAL_EPSILON)
    frame["positive"] = (frame["effect_class"] == "positive").astype(int)
    frame["positive_robust"] = (frame["effect_class_robust"] == "positive").astype(int)
    frame["abs_delta"] = frame["delta"].abs()

    # Position bins: equal-width normalized deciles plus reader-facing thirds.
    decile_index = np.minimum(np.floor(frame["relative_position"] * 10).astype(int), 9)
    frame["position_decile"] = [f"{10*i:02d}–{10*(i+1):02d}%" for i in decile_index]
    frame["position_third"] = pd.cut(
        frame["relative_position"], bins=[-np.inf, 1/3, 2/3, np.inf],
        labels=["early", "middle", "late"], right=False,
    ).astype(str)
    distance_labels = ["1", "2", "3", "4", "5", "6–8", "9–12", "13+"]
    frame["distance_bin"] = pd.cut(
        frame["distance_to_final"], bins=[0, 1, 2, 3, 4, 5, 8, 12, np.inf],
        labels=distance_labels, right=True,
    ).astype(str)

    position_rows = []
    position_rows.extend(bin_summary(
        frame, "position_decile", "position_decile", [f"{10*i:02d}–{10*(i+1):02d}%" for i in range(10)]
    ))
    position_rows.extend(bin_summary(frame, "position_third", "position_third", ["early", "middle", "late"]))
    position_rows.extend(bin_summary(frame, "distance_bin", "distance_bin", distance_labels))
    position_summary = pd.DataFrame(position_rows)

    trajectories = make_trajectory_summary(frame)
    if len(trajectories) != 237:
        raise RuntimeError("Trajectory summary count mismatch")
    if int(trajectories["n_missing_targets"].sum()) != 32:
        raise RuntimeError("Planned-minus-valid branch count is not 32")
    compact = model_task_summary(frame, trajectories)

    concentration = pd.to_numeric(trajectories["concentration"], errors="coerce").dropna()
    exact_count_distribution = count_distribution(trajectories, "n_positive")
    robust_count_distribution = count_distribution(trajectories, "n_positive_robust")
    exact_topology = topology_distribution(trajectories, "sign_topology")
    robust_topology = topology_distribution(trajectories, "sign_topology_robust")
    any_positive = int((trajectories["n_positive"] > 0).sum())
    multiple_positive = int((trajectories["n_positive"] >= 2).sum())
    robust_any_positive = int((trajectories["n_positive_robust"] > 0).sum())
    robust_multiple_positive = int((trajectories["n_positive_robust"] >= 2).sum())

    trajectory_thirds = (
        frame.groupby(["trajectory_id", "position_third"], observed=True)
        .agg(
            positive_rate=("positive", "mean"),
            robust_positive_rate=("positive_robust", "mean"),
            mean_delta=("delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
        )
        .reset_index()
        .groupby("position_third", observed=True)[
            ["positive_rate", "robust_positive_rate", "mean_delta", "mean_abs_delta"]
        ]
        .mean()
    )

    # A population-level long-range check, not case selection: upper-quartile
    # distance is >= the empirical 75th percentile (6 downstream generations).
    distance_q75 = int(frame["distance_to_final"].quantile(0.75))
    long_range = frame[frame["distance_to_final"] >= distance_q75]
    long_positive = long_range[long_range["effect_class"] == "positive"]

    results = {
        "source": str(args.input.relative_to(ROOT)),
        "population": {
            "trajectories": int(frame["trajectory_id"].nunique()),
            "valid_branches": int(len(frame)),
            "planned_branches": int(trajectories["n_planned_targets"].sum()),
            "missing_branches": int(trajectories["n_missing_targets"].sum()),
            "incomplete_trajectories": int((trajectories["n_missing_targets"] > 0).sum()),
        },
        "classification": {
            "primary": "frozen exact sign of delta",
            "robustness": f"epsilon={PRACTICAL_EPSILON} for data2text/summary only; binary tasks exact",
        },
        "position": {
            "spearman_relative_position_delta": clustered_spearman_ci(frame, "relative_position", "delta"),
            "spearman_relative_position_abs_delta": clustered_spearman_ci(frame, "relative_position", "abs_delta"),
            "logistic_positive_unadjusted": fit_logistic(frame, "positive", adjusted=False),
            "logistic_positive_model_task_adjusted": fit_logistic(frame, "positive", adjusted=True),
            "logistic_positive_robust_model_task_adjusted": fit_logistic(frame, "positive_robust", adjusted=True),
            "thirds": {
                row["bin"]: row for row in position_rows if row["summary_type"] == "position_third"
            },
            "trajectory_equal_thirds": {
                str(index): {key: float(value) for key, value in row.items()}
                for index, row in trajectory_thirds.to_dict("index").items()
            },
        },
        "recency": {
            "spearman_distance_delta": clustered_spearman_ci(frame, "distance_to_final", "delta"),
            "spearman_distance_abs_delta": clustered_spearman_ci(frame, "distance_to_final", "abs_delta"),
            "upper_quartile_distance_threshold": distance_q75,
            "upper_quartile_or_farther_valid_branches": int(len(long_range)),
            "upper_quartile_or_farther_positive_branches": int(len(long_positive)),
            "upper_quartile_or_farther_positive_trajectories": int(long_positive["trajectory_id"].nunique()),
            "upper_quartile_or_farther_positive_trajectory_proportion": float(
                long_positive["trajectory_id"].nunique() / len(trajectories)
            ),
        },
        "positive_turn_topology": {
            "primary": exact_count_distribution,
            "robustness_epsilon_0_01": robust_count_distribution,
            "any_positive": {"count": any_positive, "proportion": any_positive / len(trajectories)},
            "multiple_positive": {"count": multiple_positive, "proportion": multiple_positive / len(trajectories)},
            "multiple_conditional_on_any_positive": {
                "count": multiple_positive,
                "denominator": any_positive,
                "proportion": multiple_positive / any_positive,
            },
            "robust_any_positive": {
                "count": robust_any_positive,
                "proportion": robust_any_positive / len(trajectories),
            },
            "robust_multiple_positive": {
                "count": robust_multiple_positive,
                "proportion": robust_multiple_positive / len(trajectories),
            },
            "robust_multiple_conditional_on_any_positive": {
                "count": robust_multiple_positive,
                "denominator": robust_any_positive,
                "proportion": robust_multiple_positive / robust_any_positive,
            },
        },
        "concentration": {
            "defined_trajectories": int(len(concentration)),
            **quantiles(concentration),
            "fraction_at_least_0_5": float((concentration >= 0.5).mean()),
            "count_at_least_0_5": int((concentration >= 0.5).sum()),
            "fraction_at_least_0_75": float((concentration >= 0.75).mean()),
            "count_at_least_0_75": int((concentration >= 0.75).sum()),
            "fraction_equal_1": float(np.isclose(concentration, 1.0).mean()),
            "count_equal_1": int(np.isclose(concentration, 1.0).sum()),
        },
        "sign_topology": {
            "primary": exact_topology,
            "robustness_epsilon_0_01": robust_topology,
            "continuous_tasks_primary": topology_distribution(
                trajectories[trajectories["task"].isin(["data2text", "summary"])], "sign_topology"
            ),
            "continuous_tasks_robustness_epsilon_0_01": topology_distribution(
                trajectories[trajectories["task"].isin(["data2text", "summary"])], "sign_topology_robust"
            ),
        },
        "missingness_sensitivity": missingness_bounds(trajectories),
        "heterogeneity": {
            "model_task_table": "model_task_l1_summary.csv",
            "small_cells": compact.loc[compact["small_cell"] == "true", ["model", "task", "n_trajectories"]].to_dict("records"),
        },
        "mean_effect_invariance_check": {
            "branch_weighted_mean": float(frame["delta"].mean()),
            "trajectory_equal_mean": float(trajectories["mean_delta"].mean()),
        },
    }

    trajectories.to_csv(args.output_dir / "trajectory_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    position_summary.to_csv(args.output_dir / "position_summary.csv", index=False, encoding="utf-8-sig")
    compact.to_csv(args.output_dir / "model_task_l1_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "phase_l1_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(position_summary, trajectories, args.output_dir)
    print(json.dumps({
        "population": results["population"],
        "positive_turn_topology": results["positive_turn_topology"],
        "concentration": results["concentration"],
        "sign_topology": results["sign_topology"],
    }, indent=2))


if __name__ == "__main__":
    main()
