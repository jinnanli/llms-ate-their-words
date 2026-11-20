"""Offline analysis for the frozen Phase L5 Original-Turn Restoration run."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reproduced" / "phase_l5_restoration"
RAW = OUT / "raw" / "restoration_merged.jsonl"
SEED = 20260820
BOOT = 10000
BINARY_TASKS = {"actions", "code", "database", "math"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_delta(delta: float, task: str) -> str:
    eps = 0.0 if task in BINARY_TASKS else 0.01
    if delta > eps:
        return "positive"
    if delta < -eps:
        return "negative"
    return "zero"


def direction_for_cohort(cohort: str) -> str:
    return "negative" if cohort in {"P", "P0"} else "positive"


def opportunity(row: pd.Series) -> bool:
    if row["cohort"] in {"P", "N"}:
        return True  # influential cohorts were prescreened as testable
    score = float(row["short_ack_score"])
    if row["cohort"] == "P0":
        return score == 1.0 if row["task"] in BINARY_TASKS else score > 0.01
    if row["cohort"] == "N0":
        return score < 0.99
    raise ValueError(row["cohort"])


def cluster_boot_difference(df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    clusters = df["trajectory_id"].unique()
    grouped = {key: value for key, value in df.groupby("trajectory_id", sort=False)}
    diffs, ors = [], []
    for _ in range(BOOT):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [grouped[c] for c in sampled]
        boot = pd.concat(parts, ignore_index=True)
        a = boot.loc[boot["sample_role"] == "influential", "reciprocal_binary"]
        b = boot.loc[boot["sample_role"] == "matched_nochange", "reciprocal_binary"]
        if len(a) == 0 or len(b) == 0:
            continue
        diffs.append(float(a.mean() - b.mean()))
        aa, ab = float(a.sum()), float(len(a) - a.sum())
        ba, bb = float(b.sum()), float(len(b) - b.sum())
        ors.append(((aa + 0.5) * (bb + 0.5)) / ((ab + 0.5) * (ba + 0.5)))
    a = df.loc[df["sample_role"] == "influential", "reciprocal_binary"]
    b = df.loc[df["sample_role"] == "matched_nochange", "reciprocal_binary"]
    aa, ab = float(a.sum()), float(len(a) - a.sum())
    ba, bb = float(b.sum()), float(len(b) - b.sum())
    odds = ((aa + 0.5) * (bb + 0.5)) / ((ab + 0.5) * (ba + 0.5))
    return {
        "influential_n": len(a),
        "control_n": len(b),
        "influential_reciprocal_n": int(a.sum()),
        "control_reciprocal_n": int(b.sum()),
        "influential_rate": float(a.mean()),
        "control_rate": float(b.mean()),
        "difference": float(a.mean() - b.mean()),
        "difference_ci_low": float(np.percentile(diffs, 2.5)),
        "difference_ci_high": float(np.percentile(diffs, 97.5)),
        "odds_ratio": odds,
        "odds_ratio_ci_low": float(np.percentile(ors, 2.5)),
        "odds_ratio_ci_high": float(np.percentile(ors, 97.5)),
        "bootstrap_clusters": len(clusters),
        "bootstrap_replicates": BOOT,
        "bootstrap_seed": seed,
    }


def adjusted_logistic(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["influential"] = (work["sample_role"] == "influential").astype(int)
    try:
        fit = smf.glm(
            "reciprocal_binary ~ influential + C(model) + C(task)",
            data=work,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": work["trajectory_id"]})
        beta, se = float(fit.params["influential"]), float(fit.bse["influential"])
        return {
            "adjusted_odds_ratio": math.exp(beta),
            "adjusted_or_ci_low": math.exp(beta - 1.96 * se),
            "adjusted_or_ci_high": math.exp(beta + 1.96 * se),
            "adjusted_p_value": float(fit.pvalues["influential"]),
            "adjusted_n": int(fit.nobs),
        }
    except Exception as error:
        return {"adjusted_error": f"{type(error).__name__}: {error}"}


def cluster_boot_spearman(df: pd.DataFrame, x: str, y: str, seed: int) -> dict:
    observed = float(spearmanr(df[x], df[y]).statistic)
    rng = np.random.default_rng(seed)
    clusters = df["trajectory_id"].unique()
    grouped = {key: value for key, value in df.groupby("trajectory_id", sort=False)}
    values = []
    for _ in range(BOOT):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[c] for c in sampled], ignore_index=True)
        value = spearmanr(boot[x], boot[y]).statistic
        if not np.isnan(value):
            values.append(float(value))
    return {
        "n": len(df),
        "trajectory_clusters": len(clusters),
        "spearman_rho": observed,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "bootstrap_replicates": BOOT,
        "bootstrap_seed": seed,
    }


def main() -> None:
    manifest = pd.read_csv(OUT / "restoration_manifest.csv")
    attempts = read_jsonl(RAW)
    latest = {row["branch_id"]: row for row in attempts}
    if len(manifest) != 340 or len(latest) != 340:
        raise RuntimeError("Manifest/raw result count mismatch")

    result_rows = []
    for row in manifest.to_dict("records"):
        raw = latest[row["restoration_branch_id"]]
        completed = raw.get("status") == "completed"
        base = {
            "restoration_branch_id": row["restoration_branch_id"],
            "trajectory_id": row["trajectory_id"],
            "target_turn": int(row["target_turn"]),
            "model": row["model"],
            "task": row["task"],
            "cohort": row["cohort"],
            "sample_role": row["sample_role"],
            "match_set_id": row["match_set_id"],
            "matched_control_id": row.get("matched_control_id", ""),
            "matching_method": row["matching_method"],
            "directionally_testable": row["directionally_testable"],
            "testability_reason": row["testability_reason"],
            "original_score": float(row["original_score"]),
            "surgery_score": float(row["surgery_score"]),
            "short_ack_score": float(row["short_ack_score"]),
            "restoration_score": float(raw["restoration_score"]) if completed else "",
            "surgery_delta": float(row["surgery_delta"]),
            "restoration_delta": float(raw["restoration_delta"]) if completed else "",
            "restoration_effect_class": "",
            "control_directionally_testable": "",
            "reciprocal_class": "missing" if not completed else "",
            "restoration_final_output": "",
            "evaluator_result": "",
            "status": raw.get("status", "failed"),
            "error_type": raw.get("error_type", ""),
            "error": raw.get("error", ""),
        }
        if completed:
            effect_class = classify_delta(float(raw["restoration_delta"]), row["task"])
            base["restoration_effect_class"] = effect_class
            can_respond = opportunity(pd.Series(row))
            base["control_directionally_testable"] = (
                "true" if can_respond else "false"
            ) if row["sample_role"] == "matched_nochange" else ""
            if not can_respond:
                base["reciprocal_class"] = "not_testable"
            elif effect_class == direction_for_cohort(row["cohort"]):
                base["reciprocal_class"] = "reciprocal"
            elif effect_class == "zero":
                base["reciprocal_class"] = "neutral"
            else:
                base["reciprocal_class"] = "discordant"
            base["restoration_final_output"] = raw["counterfactual_history"][-1]["content"]
            base["evaluator_result"] = json.dumps(raw["evaluator_output"], ensure_ascii=False)
        result_rows.append(base)
    fields = list(result_rows[0])
    write_csv(OUT / "restoration_results.csv", result_rows, fields)
    results = pd.DataFrame(result_rows)

    comparisons = {}
    comparison_frames = {}
    for influential, control, name, seed in [
        ("P", "P0", "positive_surgery", SEED),
        ("N", "N0", "negative_surgery", SEED + 1),
    ]:
        treatment = results[results["cohort"] == influential]
        controls = results[results["cohort"] == control]
        eligible_sets = set(treatment.loc[treatment["status"] == "completed", "match_set_id"])
        eligible_sets &= set(
            controls.loc[
                (controls["status"] == "completed")
                & (controls["reciprocal_class"] != "not_testable"),
                "match_set_id",
            ]
        )
        long = results[
            results["match_set_id"].isin(eligible_sets)
            & results["cohort"].isin([influential, control])
        ].copy()
        long["reciprocal_binary"] = (long["reciprocal_class"] == "reciprocal").astype(int)
        if len(long) != 2 * len(eligible_sets):
            raise RuntimeError(f"Incomplete eligible match sets for {name}")
        stat = cluster_boot_difference(long, seed)
        stat.update(adjusted_logistic(long))
        stat["eligible_match_sets"] = len(eligible_sets)
        stat["influential_full_composition"] = Counter(
            treatment["reciprocal_class"].tolist()
        )
        stat["control_full_composition"] = Counter(controls["reciprocal_class"].tolist())
        comparisons[name] = stat
        comparison_frames[name] = long

    # Binary and continuous positive-cohort controlled decompositions.
    p_long = comparison_frames["positive_surgery"]
    p_subgroups = {}
    for label, mask in {
        "binary": p_long["task"].isin(BINARY_TASKS),
        "continuous": ~p_long["task"].isin(BINARY_TASKS),
    }.items():
        part = p_long[mask]
        p_subgroups[label] = cluster_boot_difference(part, SEED + 10 + len(p_subgroups))

    # Task-level descriptive rates in eligible matched sets.
    task_rates = []
    for name, long in comparison_frames.items():
        for task, part in long.groupby("task"):
            for role, group in part.groupby("sample_role"):
                task_rates.append({
                    "comparison": name,
                    "task": task,
                    "sample_role": role,
                    "n": len(group),
                    "reciprocal_n": int(group["reciprocal_binary"].sum()),
                    "reciprocal_rate": float(group["reciprocal_binary"].mean()),
                })

    continuous = results[
        (results["sample_role"] == "influential")
        & results["task"].isin(["data2text", "summary"])
        & (results["status"] == "completed")
    ].copy()
    continuous["negative_restoration_delta"] = -continuous["restoration_delta"].astype(float)
    continuous["abs_surgery_delta"] = continuous["surgery_delta"].abs()
    continuous["abs_restoration_delta"] = continuous["restoration_delta"].astype(float).abs()
    magnitude = {
        "signed_reciprocal": cluster_boot_spearman(
            continuous, "surgery_delta", "negative_restoration_delta", SEED + 20
        ),
        "absolute_magnitude": cluster_boot_spearman(
            continuous, "abs_surgery_delta", "abs_restoration_delta", SEED + 21
        ),
    }

    completed = results[results["status"] == "completed"]
    summary = {
        "design": json.loads((OUT / "restoration_sampling_summary.json").read_text(encoding="utf-8")),
        "execution": {
            "planned": len(results),
            "completed": len(completed),
            "missing": int((results["status"] != "completed").sum()),
            "completed_by_cohort": completed.groupby("cohort").size().to_dict(),
            "missing_rows": results.loc[
                results["status"] != "completed",
                ["restoration_branch_id", "cohort", "model", "task", "error_type", "error"],
            ].to_dict("records"),
        },
        "comparisons": comparisons,
        "positive_subgroups": p_subgroups,
        "task_rates": task_rates,
        "continuous_magnitude": magnitude,
        "classification": {
            "binary": "exact score difference",
            "continuous": "epsilon=0.01 practical tolerance",
            "primary_uncertainty": "trajectory-clustered percentile bootstrap",
        },
    }
    (OUT / "phase_l5_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: dict(x)),
        encoding="utf-8",
    )

    # Exploratory Figure 1: controlled reciprocal response composition.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), sharey=True)
    colors = {"reciprocal": "#2a7f62", "neutral": "#bdbdbd", "discordant": "#b45a55"}
    for ax, (name, long) in zip(axes, comparison_frames.items()):
        cohorts = ["influential", "matched_nochange"]
        bottoms = np.zeros(2)
        for cls in ["reciprocal", "neutral", "discordant"]:
            values = [float((long.loc[long["sample_role"] == role, "reciprocal_class"] == cls).mean()) for role in cohorts]
            ax.bar([0, 1], values, bottom=bottoms, color=colors[cls], label=cls.capitalize())
            bottoms += np.array(values)
        ax.set_xticks([0, 1], ["Influential", "Matched inert"])
        ax.set_ylim(0, 1)
        ax.set_title("Positive-surgery targets" if name == "positive_surgery" else "Negative-surgery targets")
        ax.axhline(0.5, color="white", lw=0.4)
    axes[0].set_ylabel("Proportion of restorations")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "exploratory_restoration_response.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exploratory_restoration_response.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Exploratory Figure 2: continuous reciprocal magnitude relation.
    fig, ax = plt.subplots(figsize=(3.55, 2.7))
    for cohort, color, label in [("P", "#3b6fb6", "Positive surgery"), ("N", "#b45a55", "Negative surgery")]:
        part = continuous[continuous["cohort"] == cohort]
        ax.scatter(part["surgery_delta"], part["negative_restoration_delta"], s=20, alpha=.75, color=color, label=label)
    ax.axhline(0, color="#777777", lw=.7)
    ax.axvline(0, color="#777777", lw=.7)
    ax.set_xlabel("Turn Surgery effect")
    ax.set_ylabel("Negative restoration effect")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "exploratory_continuous_reciprocity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exploratory_continuous_reciprocity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: dict(x)))


if __name__ == "__main__":
    main()
