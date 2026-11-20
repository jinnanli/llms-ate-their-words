"""Phase L3: offline trajectory-level structure of frozen SHORT_ACK results."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "rq2" / "rq2_paired_trajectory.csv"
OUTPUT = ROOT / "results" / "reproduced" / "phase_l3"
EPSILON = 0.01
TASK_ORDER = ["actions", "database", "math", "code", "data2text", "summary"]
MODEL_ORDER = [
    "gpt-5.6-luna", "grok-4.5", "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-14B",
]


def safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_summary(group: pd.DataFrame) -> dict:
    original = group["sharded_score"]
    short = group["short_ack_score"]
    n00 = int(((original == 0) & (short == 0)).sum())
    n01 = int(((original == 0) & (short == 1)).sum())
    n10 = int(((original == 1) & (short == 0)).sum())
    n11 = int(((original == 1) & (short == 1)).sum())
    n = len(group)
    failures = n00 + n01
    successes = n10 + n11
    return {
        "paired_n": n,
        "baseline_failure_n": failures,
        "baseline_success_n": successes,
        "n_00": n00,
        "n_01": n01,
        "n_10": n10,
        "n_11": n11,
        "p_00": n00 / n,
        "p_01": n01 / n,
        "p_10": n10 / n,
        "p_11": n11 / n,
        "rescue_rate_given_failure": safe_div(n01, failures),
        "harm_rate_given_success": safe_div(n10, successes),
        "retention_rate_given_success": safe_div(n11, successes),
        "gross_change_rate": (n01 + n10) / n,
        "beneficial_change_rate": n01 / n,
        "harmful_change_rate": n10 / n,
        "net_delta": (n01 - n10) / n,
        "baseline_mean": float(original.mean()),
        "short_ack_mean": float(short.mean()),
    }


def continuous_summary(group: pd.DataFrame) -> dict:
    delta = group["short_ack_delta"].astype(float)
    positive = delta > EPSILON
    negative = delta < -EPSILON
    unchanged = ~(positive | negative)
    return {
        "paired_n": len(group),
        "baseline_mean": float(group["sharded_score"].mean()),
        "short_ack_mean": float(group["short_ack_score"].mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "q10": float(delta.quantile(.10)),
        "q25": float(delta.quantile(.25)),
        "q75": float(delta.quantile(.75)),
        "q90": float(delta.quantile(.90)),
        "positive_n_eps01": int(positive.sum()),
        "nochange_n_eps01": int(unchanged.sum()),
        "negative_n_eps01": int(negative.sum()),
        "positive_rate_eps01": float(positive.mean()),
        "nochange_rate_eps01": float(unchanged.mean()),
        "negative_rate_eps01": float(negative.mean()),
        "changed_rate_eps01": float((positive | negative).mean()),
        "mean_positive_delta": float(delta[positive].mean()) if positive.any() else None,
        "mean_negative_delta": float(delta[negative].mean()) if negative.any() else None,
    }


def correlation(x: pd.Series, y: pd.Series) -> dict:
    pair = pd.concat([x, y], axis=1).dropna()
    return {
        "n": len(pair),
        "pearson": float(stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic),
        "spearman": float(stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic),
    }


def make_figures(data: pd.DataFrame, cells: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})

    binary = data[data["task_type"] == "binary"].copy()
    binary["transition"] = np.select(
        [
            (binary["sharded_score"] == 0) & (binary["short_ack_score"] == 1),
            (binary["sharded_score"] == 1) & (binary["short_ack_score"] == 0),
            (binary["sharded_score"] == 0) & (binary["short_ack_score"] == 0),
        ],
        ["Rescued (0->1)", "Harmed (1->0)", "Persistent failure (0->0)"],
        default="Persistent success (1->1)",
    )
    order = ["Rescued (0->1)", "Harmed (1->0)", "Persistent failure (0->0)", "Persistent success (1->1)"]
    comp = pd.crosstab(binary["task"], binary["transition"], normalize="index").reindex(
        index=["actions", "database", "math", "code"], columns=order, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(5.4, 2.8), constrained_layout=True)
    comp.plot.barh(stacked=True, ax=ax, color=["#4c956c", "#c65d57", "#b7b7b7", "#6483a8"], width=.72)
    ax.set_xlabel("Fraction of paired trajectories")
    ax.set_ylabel("")
    ax.set_title("Binary SHORT_ACK transition composition")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"exploratory_binary_transitions_by_task.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    continuous = data[data["task_type"] == "continuous"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.55), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, ["data2text", "summary"]):
        values = continuous.loc[continuous["task"] == task, "short_ack_delta"]
        ax.hist(values, bins=28, color="#6e88a8", edgecolor="white", linewidth=.35)
        ax.axvline(-EPSILON, color="#8e3b46", linestyle="--", linewidth=.8)
        ax.axvline(EPSILON, color="#3b755f", linestyle="--", linewidth=.8)
        ax.axvline(values.mean(), color="black", linewidth=1.0)
        ax.set_title(task)
        ax.set_xlabel("SHORT_ACK - SHARDED")
    axes[0].set_ylabel("Trajectories")
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"exploratory_continuous_delta_distributions.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
    for task in TASK_ORDER:
        subset = cells[cells["task"] == task]
        axes[0].scatter(subset["positive_rate"], subset["mean_delta"], label=task, s=30)
        axes[1].scatter(subset["changed_rate"], subset["mean_delta"], label=task, s=30)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_ylabel("Cell mean SHORT_ACK effect")
    axes[0].set_xlabel("Positive fraction")
    axes[1].set_xlabel("Practical changed fraction")
    axes[0].set_title("Breadth of beneficial change")
    axes[1].set_title("Gross churn")
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"exploratory_cell_breadth_vs_effect.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT, encoding="utf-8-sig")
    if len(data) != 2973 or data["trajectory_id"].nunique() != 2973:
        raise RuntimeError("Frozen RQ2 paired population mismatch")
    if not data["short_ack_paired_valid"].astype(bool).all():
        raise RuntimeError("Invalid row found in paired RQ2 interface")
    if not np.allclose(
        data["short_ack_delta"], data["short_ack_score"] - data["sharded_score"], atol=1e-12, rtol=0
    ):
        raise RuntimeError("SHORT_ACK delta mismatch")
    expected_types = {
        "actions": "binary", "code": "binary", "database": "binary", "math": "binary",
        "data2text": "continuous", "summary": "continuous",
    }
    if data.groupby("task")["task_type"].first().to_dict() != expected_types:
        raise RuntimeError("Task/evaluator type mapping mismatch")

    binary = data[data["task_type"] == "binary"].copy()
    continuous = data[data["task_type"] == "continuous"].copy()
    if not set(binary["sharded_score"].unique()).issubset({0.0, 1.0}) or not set(binary["short_ack_score"].unique()).issubset({0.0, 1.0}):
        raise RuntimeError("Binary evaluator task contains non-binary scores")

    binary_rows = []
    for (model, task), group in binary.groupby(["model", "task"], sort=False):
        binary_rows.append({"model": model, "task": task, **binary_summary(group)})
    binary_cells = pd.DataFrame(binary_rows).sort_values(["task", "model"])
    binary_cells.to_csv(OUTPUT / "rq2_binary_transitions.csv", index=False, encoding="utf-8-sig")

    continuous_rows = []
    for (model, task), group in continuous.groupby(["model", "task"], sort=False):
        continuous_rows.append({"model": model, "task": task, **continuous_summary(group)})
    continuous_cells = pd.DataFrame(continuous_rows).sort_values(["task", "model"])
    continuous_cells.to_csv(OUTPUT / "rq2_continuous_distribution.csv", index=False, encoding="utf-8-sig")

    cell_rows = []
    for (model, task), group in data.groupby(["model", "task"], sort=False):
        delta = group["short_ack_delta"].astype(float)
        is_binary = group["task_type"].iloc[0] == "binary"
        epsilon = 0.0 if is_binary else EPSILON
        positive = delta > epsilon
        negative = delta < -epsilon
        unchanged = ~(positive | negative)
        base = {
            "model": model,
            "task": task,
            "task_type": group["task_type"].iloc[0],
            "paired_n": len(group),
            "baseline_mean": float(group["sharded_score"].mean()),
            "short_ack_mean": float(group["short_ack_score"].mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "positive_rate": float(positive.mean()),
            "unchanged_rate": float(unchanged.mean()),
            "negative_rate": float(negative.mean()),
            "changed_rate": float((positive | negative).mean()),
            "mean_positive_delta": float(delta[positive].mean()) if positive.any() else None,
            "mean_negative_delta": float(delta[negative].mean()) if negative.any() else None,
            "signed_effect_sum": float(delta.sum()),
            "positive_effect_mass": float(delta.clip(lower=0).sum()),
            "negative_effect_mass": float(delta.clip(upper=0).sum()),
            "classification_epsilon": epsilon,
        }
        if is_binary:
            bs = binary_summary(group)
            base.update({
                "rescue_rate_given_failure": bs["rescue_rate_given_failure"],
                "harm_rate_given_success": bs["harm_rate_given_success"],
                "gross_change_rate": bs["gross_change_rate"],
            })
        else:
            base.update({
                "rescue_rate_given_failure": None,
                "harm_rate_given_success": None,
                "gross_change_rate": base["changed_rate"],
            })
        cell_rows.append(base)
    cells = pd.DataFrame(cell_rows).sort_values(["task", "model"])
    cells.to_csv(OUTPUT / "rq2_cell_effect_structure.csv", index=False, encoding="utf-8-sig")

    binary_overall = binary_summary(binary)
    binary_by_task = {task: binary_summary(group) for task, group in binary.groupby("task")}
    continuous_by_task = {task: continuous_summary(group) for task, group in continuous.groupby("task")}

    overall_positive = int(binary_overall["n_01"] + sum(v["positive_n_eps01"] for v in continuous_by_task.values()))
    overall_negative = int(binary_overall["n_10"] + sum(v["negative_n_eps01"] for v in continuous_by_task.values()))
    overall_unchanged = len(data) - overall_positive - overall_negative
    overall_practical_structure = {
        "positive_n": overall_positive,
        "positive_rate": overall_positive / len(data),
        "unchanged_n": overall_unchanged,
        "unchanged_rate": overall_unchanged / len(data),
        "negative_n": overall_negative,
        "negative_rate": overall_negative / len(data),
        "changed_n": overall_positive + overall_negative,
        "changed_rate": (overall_positive + overall_negative) / len(data),
        "note": "binary tasks use exact transitions; continuous tasks use epsilon=.01",
    }

    breadth = {
        "mean_delta_vs_positive_rate": correlation(cells["mean_delta"], cells["positive_rate"]),
        "mean_delta_vs_changed_rate": correlation(cells["mean_delta"], cells["changed_rate"]),
        "mean_delta_vs_conditional_positive_magnitude": correlation(cells["mean_delta"], cells["mean_positive_delta"]),
    }
    continuous_cell = cells[cells["task_type"] == "continuous"]
    breadth["continuous_cells_only"] = {
        "mean_delta_vs_positive_rate": correlation(continuous_cell["mean_delta"], continuous_cell["positive_rate"]),
        "mean_delta_vs_changed_rate": correlation(continuous_cell["mean_delta"], continuous_cell["changed_rate"]),
        "mean_delta_vs_conditional_positive_magnitude": correlation(
            continuous_cell["mean_delta"], continuous_cell["mean_positive_delta"]
        ),
    }

    positive_total = float(data["short_ack_delta"].clip(lower=0).sum())
    negative_total = float(data["short_ack_delta"].clip(upper=0).sum())
    cells_by_positive = cells.sort_values("positive_effect_mass", ascending=False)
    task_contribution = (
        data.assign(positive_mass=data["short_ack_delta"].clip(lower=0))
        .groupby("task")
        .agg(n=("trajectory_id", "size"), net=("short_ack_delta", "sum"), positive_mass=("positive_mass", "sum"))
        .reset_index()
    )
    model_contribution = (
        data.assign(positive_mass=data["short_ack_delta"].clip(lower=0))
        .groupby("model")
        .agg(n=("trajectory_id", "size"), net=("short_ack_delta", "sum"), positive_mass=("positive_mass", "sum"))
        .reset_index()
    )

    def top_tenth_share(frame: pd.DataFrame) -> dict:
        k = int(math.ceil(.10 * len(frame)))
        positive = frame["short_ack_delta"].clip(lower=0).sort_values(ascending=False)
        return {
            "population_n": len(frame), "top_n": k,
            "positive_mass": float(positive.sum()),
            "top_tenth_share_of_positive_mass": safe_div(float(positive.head(k).sum()), float(positive.sum())),
        }

    concentration = {
        "overall_net_effect_sum": float(data["short_ack_delta"].sum()),
        "overall_positive_effect_mass": positive_total,
        "overall_negative_effect_mass": negative_total,
        "top_3_cells_share_positive_mass": float(cells_by_positive.head(3)["positive_effect_mass"].sum() / positive_total),
        "top_3_cells": cells_by_positive.head(3)[
            ["model", "task", "paired_n", "mean_delta", "positive_effect_mass"]
        ].to_dict("records"),
        "task_contribution": task_contribution.to_dict("records"),
        "model_contribution": model_contribution.to_dict("records"),
        "top_tenth_all": top_tenth_share(data),
        "top_tenth_continuous": top_tenth_share(continuous),
    }

    results = {
        "source": str(INPUT.relative_to(ROOT)),
        "analysis_population": {
            "paired_trajectories": len(data),
            "unique_trajectories": data["trajectory_id"].nunique(),
            "binary_trajectories": len(binary),
            "continuous_trajectories": len(continuous),
            "model_task_cells": len(cells),
            "conditioned_on_valid_short_ack_pair": True,
        },
        "overall_scores": {
            "trajectory_weighted_sharded_mean": float(data["sharded_score"].mean()),
            "trajectory_weighted_short_ack_mean": float(data["short_ack_score"].mean()),
            "trajectory_weighted_mean_delta": float(data["short_ack_delta"].mean()),
            "cell_equal_mean_delta": float(cells["mean_delta"].mean()),
        },
        "binary_overall": binary_overall,
        "binary_by_task": binary_by_task,
        "continuous_by_task": continuous_by_task,
        "overall_practical_outcome_structure": overall_practical_structure,
        "breadth_magnitude": breadth,
        "contribution_concentration": concentration,
        "classification": {
            "binary": "exact: positive=0->1, negative=1->0, unchanged=0->0 or 1->1",
            "continuous": "practical: positive=delta>.01, unchanged=abs(delta)<=.01, negative=delta<-.01",
            "raw_continuous_delta_retained": True,
        },
    }
    (OUTPUT / "phase_l3_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(data, cells)
    print(json.dumps({
        "population": results["analysis_population"],
        "overall_scores": results["overall_scores"],
        "binary_overall": binary_overall,
        "continuous_by_task": continuous_by_task,
    }, indent=2))


if __name__ == "__main__":
    main()
