"""Phase L2: offline cross-RQ analysis over frozen behavioral results."""
from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_EVIDENCE = ROOT / "external_raw" / "P5_scaling_history"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "phase_l2"
RQ1_INPUT = ROOT / "data" / "rq1" / "rq1_cell_summary.csv"
RQ2_INPUT = ROOT / "data" / "rq2" / "rq2_paired_trajectory.csv"
RQ3_INPUT = ROOT / "data" / "rq3" / "turn_surgery_branches_compact.csv"
L1_TRAJECTORY_INPUT = ROOT / "data" / "rq3" / "trajectory_sensitivity_summary.csv"
L1_CELL_INPUT = ROOT / "results" / "rq3" / "model_task_l1_summary.csv"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260820
MODEL_ORDER = [
    "gpt-5.6-luna", "grok-4.5", "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-14B",
]
TASK_ORDER = ["actions", "code", "data2text", "database", "math", "summary"]
TASK_COLORS = {
    "actions": "#4c78a8", "code": "#f58518", "data2text": "#54a24b",
    "database": "#e45756", "math": "#b279a2", "summary": "#72b7b2",
}
MODEL_MARKERS = {
    "gpt-5.6-luna": "o", "grok-4.5": "s", "gpt-4o": "^",
    "deepseek-ai/DeepSeek-V3.2": "D", "Qwen/Qwen3-14B": "P",
}


def finite_pair(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()


def weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    mx = np.sum(weights * x)
    my = np.sum(weights * y)
    covariance = np.sum(weights * (x - mx) * (y - my))
    denominator = math.sqrt(
        np.sum(weights * (x - mx) ** 2) * np.sum(weights * (y - my) ** 2)
    )
    return float(covariance / denominator) if denominator else float("nan")


def correlation_bundle(
    frame: pd.DataFrame, x: str, y: str, *, weights: str | None = None,
    seed_offset: int = 0,
) -> dict:
    columns = [x, y] + ([weights] if weights else [])
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    xv = data[x].to_numpy(float)
    yv = data[y].to_numpy(float)
    if len(data) < 3 or np.unique(xv).size < 2 or np.unique(yv).size < 2:
        return {"n": int(len(data)), "status": "INSUFFICIENT_VARIATION"}
    pearson = float(stats.pearsonr(xv, yv).statistic)
    spearman = float(stats.spearmanr(xv, yv).statistic)
    weighted = weighted_pearson(xv, yv, data[weights].to_numpy(float)) if weights else None
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    # Vectorized row-bootstrap.  This preserves the frozen 10,000-replicate
    # convention while avoiding thousands of Python-level scipy calls.
    idx = rng.integers(0, len(data), size=(BOOTSTRAP_REPS, len(data)))
    bx, by = xv[idx], yv[idx]

    def row_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        denominator = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (ac * bc).sum(axis=1) / denominator

    pearson_draws = row_correlation(bx, by)
    spearman_draws = row_correlation(
        stats.rankdata(bx, axis=1), stats.rankdata(by, axis=1)
    )
    valid = np.isfinite(pearson_draws) & np.isfinite(spearman_draws)
    pearson_draws = pearson_draws[valid]
    spearman_draws = spearman_draws[valid]
    weighted_draws = np.array([], dtype=float)
    if weights:
        bw = data[weights].to_numpy(float)[idx]
        weight_sum = bw.sum(axis=1, keepdims=True)
        normalized = bw / weight_sum
        mx = (normalized * bx).sum(axis=1, keepdims=True)
        my = (normalized * by).sum(axis=1, keepdims=True)
        covariance = (normalized * (bx - mx) * (by - my)).sum(axis=1)
        denominator = np.sqrt(
            (normalized * (bx - mx) ** 2).sum(axis=1)
            * (normalized * (by - my) ** 2).sum(axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted_draws = covariance / denominator
        weighted_draws = weighted_draws[np.isfinite(weighted_draws)]
    result = {
        "n": int(len(data)),
        "pearson": pearson,
        "pearson_bootstrap_95ci": [
            float(np.quantile(pearson_draws, 0.025)),
            float(np.quantile(pearson_draws, 0.975)),
        ],
        "spearman": spearman,
        "spearman_bootstrap_95ci": [
            float(np.quantile(spearman_draws, 0.025)),
            float(np.quantile(spearman_draws, 0.975)),
        ],
        "bootstrap_unit": "row (cell or trajectory, according to analysis table)",
        "bootstrap_replicates": len(pearson_draws),
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
    }
    if weights:
        result.update({
            "weighted_pearson": weighted,
            "weighted_pearson_bootstrap_95ci": [
                float(np.quantile(weighted_draws, 0.025)),
                float(np.quantile(weighted_draws, 0.975)),
            ],
            "weight_column": weights,
        })
    return result


def mean_ci(values: pd.Series, seed_text: str) -> tuple[float, float, float]:
    data = values.dropna().to_numpy(float)
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(BOOTSTRAP_REPS, len(data)))
    draws = data[indices].mean(axis=1)
    return float(data.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fit_ols(frame: pd.DataFrame, formula: str, term: str) -> dict:
    fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
    ci_low, ci_high = fit.conf_int().loc[term].tolist()
    return {
        "formula": formula,
        "term": term,
        "coefficient": float(fit.params[term]),
        "hc3_95ci": [float(ci_low), float(ci_high)],
        "p_value": float(fit.pvalues[term]),
        "r_squared": float(fit.rsquared),
        "n": int(fit.nobs),
        "residual_df": float(fit.df_resid),
    }


def fit_logistic(frame: pd.DataFrame, formula: str, term: str) -> dict:
    try:
        fit = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(cov_type="HC3")
        ci_low, ci_high = fit.conf_int().loc[term].tolist()
        return {
            "formula": formula,
            "term": term,
            "coefficient": float(fit.params[term]),
            "odds_ratio": math.exp(float(fit.params[term])),
            "hc3_odds_ratio_95ci": [math.exp(float(ci_low)), math.exp(float(ci_high))],
            "p_value": float(fit.pvalues[term]),
            "n": int(fit.nobs),
            "residual_df": float(fit.df_resid),
        }
    except Exception as error:
        return {"formula": formula, "term": term, "status": "FIT_FAILED", "error": str(error)}


def top_third_overlap(frame: pd.DataFrame, x: str, y: str) -> dict:
    n_top = math.ceil(len(frame) / 3)
    stable = frame.copy()
    stable["_key"] = stable["model"] + "|" + stable["task"]
    top_x = set(stable.sort_values([x, "_key"], ascending=[False, True]).head(n_top)["_key"])
    top_y = set(stable.sort_values([y, "_key"], ascending=[False, True]).head(n_top)["_key"])
    intersection = top_x & top_y
    return {
        "top_n": n_top,
        "overlap_count": len(intersection),
        "overlap_fraction_of_top_third": len(intersection) / n_top,
        "jaccard": len(intersection) / len(top_x | top_y),
        "overlap_cells": sorted(intersection),
    }


def add_ranks(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[f"{column}_rank_desc"] = result[column].rank(method="average", ascending=False)
    return result


def task_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("task")[column].rank(method="average", pct=True)


def make_figures(cell12: pd.DataFrame, cell23: pd.DataFrame, trajectories: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})

    fig, ax = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
    for _, row in cell12.iterrows():
        ax.scatter(
            row["degradation_full_sharded"], row["rq2_mean_short_delta"],
            color=TASK_COLORS[row["task"]], marker=MODEL_MARKERS[row["model"]], s=35,
            edgecolor="black", linewidth=0.35,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("FULL − SHARDED degradation")
    ax.set_ylabel("SHORT_ACK − SHARDED")
    ax.set_title("RQ1 degradation vs whole-history sensitivity")
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"exploratory_rq1_rq2_cells.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for _, row in cell23.iterrows():
        kwargs = dict(color=TASK_COLORS[row["task"]], marker=MODEL_MARKERS[row["model"]], s=35, edgecolor="black", linewidth=0.35)
        axes[0].scatter(row["rq2_mean_short_delta"], row["positive_turn_fraction"], **kwargs)
        axes[1].scatter(row["rq2_mean_short_delta"], row["any_positive_trajectory_rate"], **kwargs)
    axes[0].set_ylabel("Positive valid-branch fraction")
    axes[1].set_ylabel("Any-positive trajectory fraction")
    for ax in axes:
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Cell SHORT_ACK − SHARDED")
    axes[0].set_title("Turn-level prevalence")
    axes[1].set_title("Trajectory-level prevalence")
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"exploratory_rq2_rq3_cells.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    matched = trajectories.dropna(subset=["short_ack_delta"])
    fig, ax = plt.subplots(figsize=(4.4, 3.1), constrained_layout=True)
    for task in TASK_ORDER:
        subset = matched[matched["task"] == task]
        ax.scatter(
            subset["short_ack_delta"], subset["positive_fraction"],
            color=TASK_COLORS[task], label=task, alpha=0.68, s=22,
        )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Trajectory SHORT_ACK − SHARDED")
    ax.set_ylabel("Positive target fraction")
    ax.set_title("Same-trajectory whole/local sensitivity")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"exploratory_rq2_rq3_trajectories.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rq1 = pd.read_csv(RQ1_INPUT, encoding="utf-8-sig")
    rq2 = pd.read_csv(RQ2_INPUT, encoding="utf-8-sig")
    rq3 = pd.read_csv(RQ3_INPUT, encoding="utf-8-sig")
    l1 = pd.read_csv(L1_TRAJECTORY_INPUT, encoding="utf-8-sig")
    model_task_l1 = pd.read_csv(L1_CELL_INPUT, encoding="utf-8-sig")

    if len(rq1) != 30 or len(rq2) != 2973 or len(rq3) != 1440 or len(l1) != 237:
        raise RuntimeError("Canonical L0/L1 population mismatch")

    # RQ1 uncertainty comes from the frozen P5 paired gap analysis.
    p5 = json.loads((ROOT / "P5_FINAL_STATISTICAL_RESULTS.json").read_text(encoding="utf-8"))
    gap_rows = pd.DataFrame([
        row for row in p5["mode_gaps"] if row["contrast"] == "FULL-SHARDED"
    ]).rename(columns={
        "mean": "degradation_crosscheck", "ci_low": "degradation_ci_low",
        "ci_high": "degradation_ci_high", "n": "rq1_paired_n",
    })
    if len(gap_rows) != 30:
        raise RuntimeError("Frozen RQ1 gap table does not contain 30 cells")

    rq2_cell_rows: list[dict] = []
    for (model, task), group in rq2.groupby(["model", "task"], sort=False):
        estimate, ci_low, ci_high = mean_ci(group["short_ack_delta"], f"rq2|{model}|{task}")
        rq2_cell_rows.append({
            "model": model, "task": task, "rq2_mean_short_delta": estimate,
            "rq2_ci_low": ci_low, "rq2_ci_high": ci_high, "paired_n": len(group),
        })
    rq2_cells = pd.DataFrame(rq2_cell_rows)

    cell12 = (
        rq1[["model", "task", "degradation_full_sharded", "paired_complete_n"]]
        .merge(gap_rows[["model", "task", "degradation_crosscheck", "degradation_ci_low", "degradation_ci_high", "rq1_paired_n"]], on=["model", "task"], validate="one_to_one")
        .merge(rq2_cells, on=["model", "task"], validate="one_to_one")
    )
    if not np.allclose(cell12["degradation_full_sharded"], cell12["degradation_crosscheck"], atol=1e-12, rtol=0):
        raise RuntimeError("RQ1 L0 degradation does not match frozen gap results")
    cell12 = cell12.drop(columns="degradation_crosscheck")
    cell12 = add_ranks(cell12, ["degradation_full_sharded", "rq2_mean_short_delta"])
    cell12["absolute_rank_gap"] = (
        cell12["degradation_full_sharded_rank_desc"] - cell12["rq2_mean_short_delta_rank_desc"]
    ).abs()

    # RQ3 cell summaries are trajectory-equal where a trajectory-level quantity exists.
    branch_cell = (
        rq3.assign(positive=(rq3["effect_class"] == "positive").astype(int))
        .groupby(["model", "task"])
        .agg(rq3_branch_n=("trajectory_id", "size"), positive_turn_fraction=("positive", "mean"))
        .reset_index()
    )
    trajectory_cell = (
        l1.groupby(["model", "task"])
        .agg(
            rq3_trajectory_n=("trajectory_id", "size"),
            rq3_planned_branch_n=("n_planned_targets", "sum"),
            rq3_missing_branch_n=("n_missing_targets", "sum"),
            any_positive_trajectory_rate=("n_positive", lambda x: float((x > 0).mean())),
            mean_positive_fraction_per_trajectory=("positive_fraction", "mean"),
            mean_n_positive=("n_positive", "mean"),
            mean_local_effect=("mean_delta", "mean"),
        )
        .reset_index()
    )
    robust_cell = model_task_l1[[
        "model", "task", "robust_positive_rate", "robust_any_positive_trajectory_rate"
    ]].rename(columns={
        "robust_positive_rate": "robust_positive_turn_fraction",
        "robust_any_positive_trajectory_rate": "robust_any_positive_trajectory_rate",
    })
    cell23 = (
        rq2_cells.merge(branch_cell, on=["model", "task"], validate="one_to_one")
        .merge(trajectory_cell, on=["model", "task"], validate="one_to_one")
        .merge(robust_cell, on=["model", "task"], validate="one_to_one")
    )
    cell23 = add_ranks(cell23, [
        "rq2_mean_short_delta", "positive_turn_fraction", "any_positive_trajectory_rate"
    ])
    cell23["rq2_any_positive_absolute_rank_gap"] = (
        cell23["rq2_mean_short_delta_rank_desc"] - cell23["any_positive_trajectory_rate_rank_desc"]
    ).abs()

    # Same-trajectory joins. FULL and SHARDED are recovered only from the frozen P5 source manifest.
    source_path = args.evidence_root / "manifests" / "p5_source_trajectories.csv"
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    source_selected = source[source["trajectory_id"].isin(l1["trajectory_id"])][[
        "trajectory_id", "FULL_score", "SHARDED_score"
    ]]
    if len(source_selected) != 237 or source_selected["trajectory_id"].nunique() != 237:
        raise RuntimeError("Cannot recover all 237 selected FULL/SHARDED trajectories")
    rq2_selected = rq2[rq2["trajectory_id"].isin(l1["trajectory_id"])][[
        "trajectory_id", "short_ack_score", "short_ack_delta"
    ]]
    trajectory = (
        l1.merge(source_selected, on="trajectory_id", validate="one_to_one")
        .merge(rq2_selected, on="trajectory_id", how="left", validate="one_to_one")
    )
    trajectory["degradation"] = trajectory["FULL_score"] - trajectory["SHARDED_score"]
    trajectory["any_positive"] = (trajectory["n_positive"] > 0).astype(int)
    trajectory["any_positive_robust"] = (trajectory["n_positive_robust"] > 0).astype(int)
    # Every branch in one trajectory has the same formal original score.
    original = rq3.groupby("trajectory_id")["original_score"].agg(["min", "max"]).reset_index()
    if not np.allclose(original["min"], original["max"], atol=1e-12, rtol=0):
        raise RuntimeError("RQ3 original score varies within a trajectory")
    original_map = original.set_index("trajectory_id")["min"]
    if not np.allclose(
        trajectory.set_index("trajectory_id")["SHARDED_score"].sort_index(),
        original_map.sort_index(), atol=1e-12, rtol=0,
    ):
        raise RuntimeError("Selected RQ1 SHARDED score does not match RQ3 original score")

    output_columns = [
        "trajectory_id", "model", "task", "FULL_score", "SHARDED_score", "degradation",
        "short_ack_score", "short_ack_delta", "any_positive", "n_positive",
        "positive_fraction", "max_positive_delta", "mean_delta", "concentration", "sign_topology",
    ]
    cross_trajectory = trajectory[output_columns].rename(columns={
        "FULL_score": "full_score", "SHARDED_score": "sharded_score",
    })

    # Within-task rank robustness for heterogeneous evaluator scales.
    matched = trajectory.dropna(subset=["short_ack_delta"]).copy()
    matched["short_ack_task_rank"] = task_rank(matched, "short_ack_delta")
    matched["positive_fraction_task_rank"] = task_rank(matched, "positive_fraction")
    matched["mean_delta_task_rank"] = task_rank(matched, "mean_delta")
    matched["degradation_task_rank"] = task_rank(matched, "degradation")
    matched["any_positive_task_rank"] = task_rank(matched, "any_positive")
    matched["cell"] = matched["model"] + "|" + matched["task"]
    continuous = trajectory[trajectory["task"].isin(["data2text", "summary"])].copy()
    continuous["degradation_task_rank"] = task_rank(continuous, "degradation")
    continuous["positive_fraction_task_rank"] = task_rank(continuous, "positive_fraction")

    rq1_rq2 = correlation_bundle(
        cell12, "degradation_full_sharded", "rq2_mean_short_delta",
        weights="paired_n", seed_offset=1,
    )
    rq1_rq2_regressions = {
        "unadjusted": fit_ols(cell12, "rq2_mean_short_delta ~ degradation_full_sharded", "degradation_full_sharded"),
        "model_adjusted": fit_ols(cell12, "rq2_mean_short_delta ~ degradation_full_sharded + C(model)", "degradation_full_sharded"),
        "task_adjusted": fit_ols(cell12, "rq2_mean_short_delta ~ degradation_full_sharded + C(task)", "degradation_full_sharded"),
        "model_task_adjusted": fit_ols(cell12, "rq2_mean_short_delta ~ degradation_full_sharded + C(model) + C(task)", "degradation_full_sharded"),
    }

    cell23_stats = {
        "positive_turn_fraction": correlation_bundle(
            cell23, "rq2_mean_short_delta", "positive_turn_fraction", seed_offset=2
        ),
        "any_positive_trajectory_rate": correlation_bundle(
            cell23, "rq2_mean_short_delta", "any_positive_trajectory_rate", seed_offset=3
        ),
        "mean_local_effect_trajectory_equal": correlation_bundle(
            cell23, "rq2_mean_short_delta", "mean_local_effect", seed_offset=4
        ),
        "robust_positive_turn_fraction": correlation_bundle(
            cell23, "rq2_mean_short_delta", "robust_positive_turn_fraction", seed_offset=5
        ),
        "robust_any_positive_trajectory_rate": correlation_bundle(
            cell23, "rq2_mean_short_delta", "robust_any_positive_trajectory_rate", seed_offset=6
        ),
        "adjusted_regressions": {
            "positive_turn_fraction_task_adjusted": fit_ols(
                cell23, "positive_turn_fraction ~ rq2_mean_short_delta + C(task)", "rq2_mean_short_delta"
            ),
            "positive_turn_fraction_model_task_adjusted": fit_ols(
                cell23, "positive_turn_fraction ~ rq2_mean_short_delta + C(model) + C(task)", "rq2_mean_short_delta"
            ),
            "any_positive_task_adjusted": fit_ols(
                cell23, "any_positive_trajectory_rate ~ rq2_mean_short_delta + C(task)", "rq2_mean_short_delta"
            ),
            "any_positive_model_task_adjusted": fit_ols(
                cell23, "any_positive_trajectory_rate ~ rq2_mean_short_delta + C(model) + C(task)", "rq2_mean_short_delta"
            ),
        },
    }

    trajectory_h_stats = {
        "short_delta_vs_positive_fraction": correlation_bundle(
            matched, "short_ack_delta", "positive_fraction", seed_offset=7
        ),
        "short_delta_vs_any_positive": correlation_bundle(
            matched, "short_ack_delta", "any_positive", seed_offset=8
        ),
        "short_delta_vs_mean_local_effect": correlation_bundle(
            matched, "short_ack_delta", "mean_delta", seed_offset=9
        ),
        "short_delta_vs_n_positive": correlation_bundle(
            matched, "short_ack_delta", "n_positive", seed_offset=16
        ),
        "short_delta_vs_max_positive_delta": correlation_bundle(
            matched, "short_ack_delta", "max_positive_delta", seed_offset=17
        ),
        "within_task_rank_short_vs_positive_fraction": correlation_bundle(
            matched, "short_ack_task_rank", "positive_fraction_task_rank", seed_offset=10
        ),
        "within_task_rank_short_vs_mean_local_effect": correlation_bundle(
            matched, "short_ack_task_rank", "mean_delta_task_rank", seed_offset=18
        ),
        "positive_fraction_task_model_adjusted": fit_ols(
            matched, "positive_fraction ~ short_ack_delta + C(model) + C(task)", "short_ack_delta"
        ),
        "any_positive_task_model_adjusted": fit_logistic(
            matched, "any_positive ~ short_ack_delta + C(model) + C(task)", "short_ack_delta"
        ),
        "positive_fraction_cell_adjusted": fit_ols(
            matched, "positive_fraction ~ short_ack_delta + C(cell)", "short_ack_delta"
        ),
        "any_positive_cell_adjusted": fit_logistic(
            matched, "any_positive ~ short_ack_delta + C(cell)", "short_ack_delta"
        ),
    }

    trajectory_d_stats = {
        "degradation_vs_positive_fraction": correlation_bundle(
            trajectory, "degradation", "positive_fraction", seed_offset=11
        ),
        "degradation_vs_any_positive": correlation_bundle(
            trajectory, "degradation", "any_positive", seed_offset=12
        ),
        "degradation_vs_mean_local_effect": correlation_bundle(
            trajectory, "degradation", "mean_delta", seed_offset=13
        ),
        "degradation_vs_n_positive": correlation_bundle(
            trajectory, "degradation", "n_positive", seed_offset=19
        ),
        "degradation_vs_max_positive_delta": correlation_bundle(
            trajectory, "degradation", "max_positive_delta", seed_offset=20
        ),
        "continuous_within_task_rank_degradation_vs_positive_fraction": correlation_bundle(
            continuous, "degradation_task_rank", "positive_fraction_task_rank", seed_offset=14
        ),
        "positive_fraction_task_model_adjusted": fit_ols(
            trajectory, "positive_fraction ~ degradation + C(model) + C(task)", "degradation"
        ),
        "mean_local_effect_task_model_adjusted": fit_ols(
            trajectory, "mean_delta ~ degradation + C(model) + C(task)", "degradation"
        ),
        "max_positive_delta_task_model_adjusted": fit_ols(
            trajectory.dropna(subset=["max_positive_delta"]),
            "max_positive_delta ~ degradation + C(model) + C(task)", "degradation"
        ),
        "any_positive_task_model_adjusted": fit_logistic(
            trajectory, "any_positive ~ degradation + C(model) + C(task)", "degradation"
        ),
    }

    task_stratified = {}
    for task in TASK_ORDER:
        subset_h = matched[matched["task"] == task]
        subset_d = trajectory[trajectory["task"] == task]
        task_stratified[task] = {
            "selected_trajectories": int(len(subset_d)),
            "short_ack_matched": int(len(subset_h)),
            "degradation_unique_values": int(subset_d["degradation"].nunique()),
            "short_vs_positive_fraction": correlation_bundle(
                subset_h, "short_ack_delta", "positive_fraction", seed_offset=100 + TASK_ORDER.index(task)
            ),
            "degradation_vs_positive_fraction": correlation_bundle(
                subset_d, "degradation", "positive_fraction", seed_offset=200 + TASK_ORDER.index(task)
            ),
        }

    cross_level = {
        "rq1_rq2_top_third_overlap": top_third_overlap(
            cell12, "degradation_full_sharded", "rq2_mean_short_delta"
        ),
        "rq2_rq3_any_positive_top_third_overlap": top_third_overlap(
            cell23, "rq2_mean_short_delta", "any_positive_trajectory_rate"
        ),
        "rq2_rq3_positive_turn_fraction_top_third_overlap": top_third_overlap(
            cell23, "rq2_mean_short_delta", "positive_turn_fraction"
        ),
        "rq1_rq3_any_positive_cell_correlation": correlation_bundle(
            cell12.merge(
                cell23[["model", "task", "any_positive_trajectory_rate"]],
                on=["model", "task"], validate="one_to_one",
            ),
            "degradation_full_sharded", "any_positive_trajectory_rate", seed_offset=15,
        ),
        "largest_rq1_rq2_rank_mismatches": (
            cell12.sort_values("absolute_rank_gap", ascending=False)
            .head(5)[["model", "task", "degradation_full_sharded", "rq2_mean_short_delta", "absolute_rank_gap"]]
            .to_dict("records")
        ),
        "largest_rq2_rq3_rank_mismatches": (
            cell23.sort_values("rq2_any_positive_absolute_rank_gap", ascending=False)
            .head(5)[["model", "task", "rq2_mean_short_delta", "any_positive_trajectory_rate", "rq2_any_positive_absolute_rank_gap"]]
            .to_dict("records")
        ),
    }

    results = {
        "sources": {
            "rq1": "analysis/local_behavioral_v1/rq1_cell_summary.csv",
            "rq2": "analysis/local_behavioral_v1/rq2_paired_trajectory.csv",
            "rq3": "analysis/local_behavioral_v1/rq3_turn_surgery_branches.csv",
            "l1": "analysis/local_behavioral_v1/phase_l1/trajectory_sensitivity_summary.csv",
            "selected_full_sharded": str(source_path),
        },
        "population": {
            "model_task_cells": 30,
            "rq2_paired_trajectories": len(rq2),
            "rq3_selected_trajectories": len(l1),
            "rq3_valid_branches": len(rq3),
            "selected_with_short_ack": int(matched["trajectory_id"].nunique()),
            "selected_without_short_ack": int(len(l1) - matched["trajectory_id"].nunique()),
            "selected_with_full_sharded": int(trajectory["degradation"].notna().sum()),
        },
        "rq1_rq2_cell": {
            "correlations": rq1_rq2,
            "regressions": rq1_rq2_regressions,
        },
        "rq2_rq3_cell": cell23_stats,
        "rq2_rq3_same_trajectory": trajectory_h_stats,
        "rq1_rq3_selected_trajectory": trajectory_d_stats,
        "task_stratified_selected_trajectory": task_stratified,
        "cross_level_ordering": cross_level,
        "design_caveats": [
            "RQ3 covers a stratified selected degraded subset, not the full RQ2 population.",
            "Each RQ3 model-task cell contains at most eight trajectories; Grok Code contains five.",
            "Binary-task selection fixes trajectory degradation at 1.0, leaving within-task severity variation only in Data-to-Text and Summary.",
            "Six selected trajectories have no completed SHORT_ACK counterpart and remain missing in same-trajectory RQ2 comparisons.",
        ],
    }

    cell12.to_csv(args.output_dir / "rq1_rq2_cell_relationship.csv", index=False, encoding="utf-8-sig")
    cell23.to_csv(args.output_dir / "rq2_rq3_cell_relationship.csv", index=False, encoding="utf-8-sig")
    cross_trajectory.to_csv(args.output_dir / "cross_rq_trajectory_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "phase_l2_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(cell12, cell23, cross_trajectory, args.output_dir)
    print(json.dumps({
        "population": results["population"],
        "rq1_rq2": rq1_rq2,
        "rq2_rq3_cell_any_positive": cell23_stats["any_positive_trajectory_rate"],
        "rq2_rq3_trajectory_positive_fraction": trajectory_h_stats["short_delta_vs_positive_fraction"],
        "rq1_rq3_trajectory_positive_fraction": trajectory_d_stats["degradation_vs_positive_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
