"""Freeze the Phase L5 Original-Turn Restoration candidate pool and sample.

This script is offline only. It joins frozen RQ2/RQ3 results, labels reciprocal
directional testability, and performs outcome-independent stratified sampling
and matching. It never calls a model or evaluator.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data"
OUT = ROOT / "results" / "reproduced" / "phase_l5_restoration"
EVIDENCE = ROOT / "external_raw" / "P5_scaling_history"
SEED = 20260819
ACK = "Acknowledged."
BINARY_TASKS = {"actions", "code", "database", "math"}
MODEL_ORDER = [
    "gpt-5.6-luna",
    "grok-4.5",
    "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-14B",
]
TASK_ORDER = ["actions", "database", "data2text", "code", "math", "summary"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(value) -> float:
    return float(value)


def key(row: dict) -> tuple[str, int]:
    return row["trajectory_id"], int(row["target_turn"])


def directionality(row: dict) -> tuple[str, str]:
    effect = row["effect_class"]
    task = row["task"]
    short = f(row["short_ack_score"])
    binary = task in BINARY_TASKS
    if effect == "zero":
        return "no", "control_not_directional"
    if effect == "positive":
        if binary:
            return ("yes", "testable") if short == 1.0 else ("no", "floor_limited")
        return ("yes", "testable") if short > 0.01 else ("no", "floor_limited_eps01")
    if effect == "negative":
        if binary:
            return ("yes", "testable") if short == 0.0 else ("no", "ceiling_limited")
        return ("yes", "testable") if short < 0.99 else ("no", "ceiling_limited_eps01")
    raise ValueError(f"Unknown effect class: {effect}")


def balanced_sample(rows: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Round-robin across model-task cells after fixed-seed within-cell shuffling."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["task"])].append(row)
    cell_order = [(m, t) for m in MODEL_ORDER for t in TASK_ORDER if (m, t) in groups]
    for cell in cell_order:
        indices = rng.permutation(len(groups[cell]))
        groups[cell] = [groups[cell][i] for i in indices]
    selected: list[dict] = []
    depth = 0
    while len(selected) < n:
        advanced = False
        for cell in cell_order:
            if depth < len(groups[cell]) and len(selected) < n:
                selected.append(groups[cell][depth])
                advanced = True
        if not advanced:
            break
        depth += 1
    if len(selected) != n:
        raise RuntimeError(f"Requested {n} rows but only selected {len(selected)}")
    return selected


def balanced_sample_by_task_quota(
    rows: list[dict], quotas: dict[str, int], rng: np.random.Generator
) -> list[dict]:
    selected = []
    for task in TASK_ORDER:
        quota = quotas.get(task, 0)
        if quota:
            selected.extend(balanced_sample([r for r in rows if r["task"] == task], quota, rng))
    if len(selected) != sum(quotas.values()):
        raise RuntimeError("Task-quota sampling count mismatch")
    return selected


def match_cost(treatment: dict, control: dict) -> tuple:
    same_trajectory = treatment["trajectory_id"] == control["trajectory_id"]
    same_model_task = (
        treatment["model"] == control["model"] and treatment["task"] == control["task"]
    )
    same_task = treatment["task"] == control["task"]
    # The ordered tier is the declared matching hierarchy. Within a tier,
    # minimize baseline-score, relative-position, and trajectory-length gaps.
    tier = 0 if same_trajectory else 1 if same_model_task else 2 if same_task else 3
    baseline_gap = abs(f(treatment["short_ack_score"]) - f(control["short_ack_score"]))
    position_gap = abs(f(treatment["relative_position"]) - f(control["relative_position"]))
    length_gap = abs(int(treatment["n_intermediate_turns"]) - int(control["n_intermediate_turns"]))
    stable_tie = hashlib.sha256(
        f"{SEED}|{treatment['branch_id']}|{control['branch_id']}".encode()
    ).hexdigest()
    return tier, baseline_gap, position_gap, length_gap, stable_tie


def match_controls(
    treatments: list[dict], controls: list[dict], used: set[tuple[str, int]]
) -> list[tuple[dict, dict, tuple]]:
    matches = []
    # Scarcer model-task controls are handled first; this ordering uses only
    # frozen design variables and cannot depend on restoration outcomes.
    capacity = defaultdict(int)
    for row in controls:
        if key(row) not in used:
            capacity[(row["model"], row["task"])] += 1
    ordered = sorted(
        treatments,
        key=lambda r: (
            capacity[(r["model"], r["task"])],
            MODEL_ORDER.index(r["model"]),
            TASK_ORDER.index(r["task"]),
            hashlib.sha256(f"treat|{SEED}|{r['branch_id']}".encode()).hexdigest(),
        ),
    )
    for treatment in ordered:
        # Controls are never allowed to cross task families. This keeps the
        # outcome geometry/evaluator fixed for every primary comparison.
        available = [
            row for row in controls
            if key(row) not in used and row["task"] == treatment["task"]
        ]
        if treatment["task"] in BINARY_TASKS:
            available = [
                row for row in available
                if f(row["short_ack_score"]) == f(treatment["short_ack_score"])
            ]
        if not available:
            raise RuntimeError(
                f"No unused same-task/baseline-state control remains for {treatment['task']}"
            )
        control = min(available, key=lambda row: match_cost(treatment, row))
        cost = match_cost(treatment, control)
        used.add(key(control))
        matches.append((treatment, control, cost))
    return matches


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rq3 = read_csv(ANALYSIS / "rq3" / "turn_surgery_branches_compact.csv")
    rq2 = read_csv(ANALYSIS / "rq2" / "rq2_paired_trajectory.csv")
    pool = read_csv(EVIDENCE / "manifests" / "p5_turn_surgery_pool.csv")
    neutral = [
        row for row in read_jsonl(EVIDENCE / "neutral_ack_v1_merged.jsonl")
        if row.get("status") == "completed"
    ]
    if len(rq3) != 1440 or len(rq2) != 2973 or len(pool) != 237 or len(neutral) != 2973:
        raise RuntimeError("Frozen source count mismatch")
    rq2_map = {row["trajectory_id"]: row for row in rq2}
    pool_map = {row["trajectory_id"]: row for row in pool}
    neutral_map = {row["trajectory_id"]: row for row in neutral}
    if len(rq2_map) != 2973 or len(neutral_map) != 2973:
        raise RuntimeError("SHORT_ACK trajectory key mismatch")

    candidates: list[dict] = []
    unjoined = []
    for branch in rq3:
        tid = branch["trajectory_id"]
        if tid not in rq2_map or tid not in neutral_map:
            unjoined.append(branch)
            continue
        short = rq2_map[tid]
        raw_short = neutral_map[tid]
        if abs(f(short["short_ack_score"]) - f(raw_short["final_score"])) > 1e-12:
            raise RuntimeError(f"SHORT_ACK score/raw mismatch: {tid}")
        if branch["model"] != short["model"] or branch["task"] != short["task"]:
            raise RuntimeError(f"RQ2/RQ3 identity mismatch: {tid}")
        testable, reason = directionality({**branch, **short})
        pool_row = pool_map[tid]
        final_output = raw_short["counterfactual_history"][-1]["content"]
        candidates.append({
            "trajectory_id": tid,
            "target_turn": int(branch["target_turn"]),
            "branch_id": branch["branch_id"],
            "source_instance_id": branch["source_instance_id"],
            "model": branch["model"],
            "task": branch["task"],
            "surgery_effect_class": branch["effect_class"],
            "surgery_delta": f(branch["delta"]),
            "original_score": f(branch["original_score"]),
            "surgery_score": f(branch["surgery_score"]),
            "short_ack_score": f(short["short_ack_score"]),
            "short_ack_final_output": final_output,
            "directionally_testable": testable,
            "testability_reason": reason,
            "relative_position": f(branch["relative_position"]),
            "n_intermediate_turns": int(branch["n_intermediate_turns"]),
            "distance_to_final": int(branch["distance_to_final"]),
            "target_assistant_text": branch["target_assistant_text"],
            "raw_sharded_path": pool_row["raw_sharded_path"],
            "raw_sharded_run_id": pool_row["raw_sharded_run_id"],
            "repeat_index": int(pool_row.get("repeat_index") or 0),
        })
    if len(candidates) != 1414 or len(unjoined) != 26:
        raise RuntimeError(f"Expected 1,414 joined and 26 unjoined, got {len(candidates)}/{len(unjoined)}")
    if len({key(row) for row in candidates}) != len(candidates):
        raise RuntimeError("Candidate branch key is not unique")

    candidate_fields = list(candidates[0])
    write_csv(OUT / "restoration_candidate_pool.csv", candidates, candidate_fields)

    rng = np.random.default_rng(SEED)
    positives = [r for r in candidates if r["surgery_effect_class"] == "positive" and r["directionally_testable"] == "yes"]
    negatives = [r for r in candidates if r["surgery_effect_class"] == "negative" and r["directionally_testable"] == "yes"]
    zeroes = [r for r in candidates if r["surgery_effect_class"] == "zero"]
    # Quotas are frozen before restoration outcomes. They preserve all six
    # tasks in P while respecting the small continuous-task no-change pools.
    # N is deliberately 50 rather than forcing 60--80 cases at the cost of
    # cross-task controls. Together the quotas consume no more than the 25
    # Data-to-Text and 54 Summary no-change branches available for both tests.
    p_quotas = {
        "actions": 20,
        "database": 20,
        "data2text": 5,
        "code": 14,
        "math": 37,
        "summary": 24,
    }
    n_quotas = {"data2text": 20, "summary": 30}
    selected_p = balanced_sample_by_task_quota(positives, p_quotas, rng)
    selected_n = balanced_sample_by_task_quota(negatives, n_quotas, rng)

    # Match the negative cohort first because its controls are restricted to
    # continuous tasks and several cells have very small no-change pools.
    used_controls: set[tuple[str, int]] = set()
    matches_n = match_controls(selected_n, zeroes, used_controls)
    matches_p = match_controls(selected_p, zeroes, used_controls)

    manifest: list[dict] = []
    validation_slots = {"P": 6, "P0": 6, "N": 6, "N0": 6}
    seen_validation = defaultdict(int)
    pair_counter = 0
    for treatment_cohort, control_cohort, matches in [
        ("P", "P0", matches_p), ("N", "N0", matches_n)
    ]:
        for treatment, control, cost in matches:
            pair_counter += 1
            match_set_id = f"L5M-{pair_counter:03d}"
            ids = {
                "treatment": hashlib.sha256(
                    f"restoration_v1|{treatment['trajectory_id']}|{treatment['target_turn']}".encode()
                ).hexdigest()[:24],
                "control": hashlib.sha256(
                    f"restoration_v1|{control['trajectory_id']}|{control['target_turn']}".encode()
                ).hexdigest()[:24],
            }
            for cohort, role, row, other_id in [
                (treatment_cohort, "influential", treatment, ids["control"]),
                (control_cohort, "matched_nochange", control, ids["treatment"]),
            ]:
                is_validation = seen_validation[cohort] < validation_slots[cohort]
                if is_validation:
                    seen_validation[cohort] += 1
                tier = cost[0]
                method = ["same_trajectory", "same_model_task", "same_task", "fallback"][tier]
                manifest.append({
                    **row,
                    "restoration_branch_id": ids[role == "matched_nochange" and "control" or "treatment"],
                    "cohort": cohort,
                    "sample_role": role,
                    "match_set_id": match_set_id,
                    "matched_control_id": ids["control"] if role == "influential" else "",
                    "matched_treatment_id": ids["treatment"] if role == "matched_nochange" else "",
                    "matching_method": method,
                    "match_baseline_abs_diff": cost[1],
                    "match_position_abs_diff": cost[2],
                    "match_length_abs_diff": cost[3],
                    "sampling_seed": SEED,
                    "neutral_phrase": ACK,
                    "generation_config_id": "p5_rq2_replay_t0_mt1000_v1",
                    "evaluator_config_id": "p5_task_specific_evaluator_v1",
                    "validation_sample": "true" if is_validation else "false",
                })
    # Sort validation rows first, then cohorts and stable identifiers. The sort
    # is execution bookkeeping only and does not alter membership or matching.
    cohort_order = {"P": 0, "P0": 1, "N": 2, "N0": 3}
    manifest.sort(key=lambda r: (r["validation_sample"] != "true", cohort_order[r["cohort"]], r["match_set_id"]))
    if len(manifest) != 340 or len({r["restoration_branch_id"] for r in manifest}) != 340:
        raise RuntimeError("Restoration manifest count/key mismatch")
    if sum(r["validation_sample"] == "true" for r in manifest) != 24:
        raise RuntimeError("Validation subset is not 24 rows")
    write_csv(OUT / "restoration_manifest.csv", manifest, list(manifest[0]))

    summary = {
        "interface_version": "phase_l5_restoration_v1",
        "sampling_seed": SEED,
        "rq3_valid_branches": len(rq3),
        "joined_branches": len(candidates),
        "unjoined_branches": len(unjoined),
        "joined_trajectories": len({r["trajectory_id"] for r in candidates}),
        "joined_cells": len({(r["model"], r["task"]) for r in candidates}),
        "joined_effect_classes": {
            cls: sum(r["surgery_effect_class"] == cls for r in candidates)
            for cls in ["positive", "zero", "negative"]
        },
        "testability": {
            reason: sum(r["testability_reason"] == reason for r in candidates)
            for reason in sorted({r["testability_reason"] for r in candidates})
        },
        "manifest": {
            "total": len(manifest),
            "cohorts": {c: sum(r["cohort"] == c for r in manifest) for c in ["P", "P0", "N", "N0"]},
            "validation_n": sum(r["validation_sample"] == "true" for r in manifest),
            "matching_methods": {
                m: sum(r["sample_role"] == "influential" and r["matching_method"] == m for r in manifest)
                for m in ["same_trajectory", "same_model_task", "same_task", "fallback"]
            },
        },
        "source_files": [
            "analysis/local_behavioral_v1/rq3_turn_surgery_branches.csv",
            "analysis/local_behavioral_v1/rq2_paired_trajectory.csv",
            str(EVIDENCE / "manifests" / "p5_turn_surgery_pool.csv"),
            str(EVIDENCE / "neutral_ack_v1_merged.jsonl"),
        ],
    }
    (OUT / "restoration_sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
