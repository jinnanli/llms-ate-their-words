"""Build the Phase L0 local behavioral-analysis interface from frozen results.

This script is offline-only. It reads existing P5/P7-R artifacts, validates
their alignment, and writes derived analysis tables without modifying any
canonical input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = ROOT / "external_raw" / "P5_scaling_history"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced" / "local_behavioral_v1"
MODELS = [
    "gpt-5.6-luna",
    "grok-4.5",
    "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2",
    "Qwen/Qwen3-14B",
]
TASKS = ["actions", "code", "data2text", "database", "math", "summary"]
MODES = ["FULL", "CONCAT", "SHARDED"]
BINARY_TASKS = {"actions", "code", "database", "math"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSONL at {path}:{number}: {error}") from error


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def number(value: str | float | int) -> float:
    return float(value)


def close(left: str | float, right: str | float) -> bool:
    return math.isclose(number(left), number(right), rel_tol=0.0, abs_tol=1e-12)


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("Cannot average an empty sequence")
    return sum(values) / len(values)


def user_sequence_id(instance_id: str, user_turns: list[str]) -> str:
    canonical = json.dumps(
        {"source_instance_id": instance_id, "user_turns": user_turns},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "useq_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def build_rq1(evidence: Path, output: Path) -> dict:
    source_path = evidence / "manifests" / "p5_source_trajectories.csv"
    missing_path = evidence / "manifests" / "p5_source_missing.csv"
    panel_path = ROOT / "P5_MODEL_PANEL_RESULTS.csv"
    p1_coverage_path = ROOT / "P1_UPSTREAM627_COVERAGE.csv"
    p7_missingness_path = ROOT / "P7R_MISSINGNESS_SUMMARY.csv"
    sampling_flow_path = ROOT / "P7R_SURGERY_SAMPLING_FLOW.csv"
    source = read_csv(source_path)
    missing = read_csv(missing_path)
    panel = read_csv(panel_path)
    p1_coverage = read_csv(p1_coverage_path)
    p7_missingness = read_csv(p7_missingness_path)
    sampling_flow = read_csv(sampling_flow_path)

    if len(source) != 3033 or len(missing) != 143 or len(panel) != 90:
        raise RuntimeError(
            f"RQ1 count mismatch: source={len(source)}, missing={len(missing)}, panel={len(panel)}"
        )
    source_keys = {(row["instruction_id"], row["tested_model"]) for row in source}
    if len(source_keys) != len(source):
        raise RuntimeError("RQ1 source trajectory key is not unique")
    missing_keys = {(row["instruction_id"], row["model"]) for row in missing}
    if len(missing_keys) != len(missing):
        raise RuntimeError("RQ1 missingness key is not unique")
    panel_map = {(row["model"], row["task"], row["mode"]): row for row in panel}
    if len(panel_map) != 90:
        raise RuntimeError("RQ1 model/task/mode panel key is not unique")

    source_cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    missing_cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source:
        source_cells[(row["tested_model"], row["task"])].append(row)
    for row in missing:
        missing_cells[(row["model"], row["task"])].append(row)

    planned_map = {
        (row["model"], row["task"]): int(row["count"])
        for row in sampling_flow if row["stage"] == "planned_instruction_model"
    }
    if len(planned_map) != 30:
        raise RuntimeError("RQ1 planned model/task metadata is incomplete")
    p1_map = {
        (row["model"], row["task"], row["mode"].upper()): row
        for row in p1_coverage
    }
    new_missing_map: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in p7_missingness:
        if row["experiment"] == "Layer_A_new_three_models":
            new_missing_map[(row["model"], row["task"], row["mode"].upper())] += int(row["count"])

    rows: list[dict] = []
    for model in MODELS:
        for task in TASKS:
            present = source_cells[(model, task)]
            absent = missing_cells[(model, task)]
            planned_n = planned_map[(model, task)]
            if not planned_n:
                raise RuntimeError(f"No planned population for {model}/{task}")
            missing_sets = [
                {item for item in row["missing"].split(",") if item} for row in absent
            ]
            if model in {"gpt-5.6-luna", "grok-4.5"}:
                available_n = {
                    mode: int(p1_map[(model, task, mode)]["completed_or_reused"])
                    for mode in MODES
                }
            else:
                available_n = {
                    mode: planned_n - new_missing_map[(model, task, mode)]
                    for mode in MODES
                }
            source_manifest_n = {
                mode: sum(row[f"{mode}_score"] != "" for row in present)
                for mode in MODES
            }
            formal = {mode: panel_map[(model, task, mode)] for mode in MODES}
            paired_ns = {int(formal[mode]["n"]) for mode in MODES}
            if len(paired_ns) != 1:
                raise RuntimeError(f"Formal paired N differs across modes: {model}/{task}")
            paired_n = paired_ns.pop()
            complete = [
                row for row in present
                if all(row[f"{mode}_score"] != "" for mode in MODES)
            ]
            if len(complete) != paired_n:
                raise RuntimeError(f"Complete-case N mismatch: {model}/{task}")
            for mode in MODES:
                direct = mean([number(row[f"{mode}_score"]) for row in complete])
                if not close(direct, formal[mode]["mean"]):
                    raise RuntimeError(f"Formal score mismatch: {model}/{task}/{mode}")
            score_full = number(formal["FULL"]["mean"])
            score_concat = number(formal["CONCAT"]["mean"])
            score_sharded = number(formal["SHARDED"]["mean"])
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "planned_n": planned_n,
                    "n_full": available_n["FULL"],
                    "n_concat": available_n["CONCAT"],
                    "n_sharded": available_n["SHARDED"],
                    "paired_complete_n": paired_n,
                    "score_full": score_full,
                    "score_concat": score_concat,
                    "score_sharded": score_sharded,
                    "degradation_full_sharded": score_full - score_sharded,
                    "missing_full": planned_n - available_n["FULL"],
                    "missing_concat": planned_n - available_n["CONCAT"],
                    "missing_sharded": planned_n - available_n["SHARDED"],
                    "missing_replay_aligned_sharded": planned_n - source_manifest_n["SHARDED"],
                    "source_manifest_n_full": source_manifest_n["FULL"],
                    "source_manifest_n_concat": source_manifest_n["CONCAT"],
                    "source_manifest_n_sharded": source_manifest_n["SHARDED"],
                    "score_aggregation": "paired_complete_case_mean",
                }
            )
    if len(rows) != 30:
        raise RuntimeError(f"RQ1 derived cell count differs: {len(rows)}")
    if sum(int(row["planned_n"]) * 3 for row in rows) != 9405:
        raise RuntimeError("RQ1 planned five-model observation count is not 9,405")
    if sum(
        int(row["n_full"]) + int(row["n_concat"]) + int(row["n_sharded"])
        for row in rows
    ) != 9289:
        raise RuntimeError("RQ1 formal available observation count is not 9,289")
    fields = list(rows[0])
    write_csv(output / "rq1_cell_summary.csv", fields, rows)
    return {
        "rows": len(rows),
        "source_trajectories": len(source),
        "complete_case_trajectories": sum(int(row["paired_complete_n"]) for row in rows),
        "planned_observations": sum(int(row["planned_n"]) * 3 for row in rows),
        "available_observations": sum(
            int(row["n_full"]) + int(row["n_concat"]) + int(row["n_sharded"])
            for row in rows
        ),
    }


def build_rq2(evidence: Path, output: Path) -> dict:
    neutral_path = ROOT / "P5_NEUTRAL_ACK_RESULTS.csv"
    length_path = ROOT / "P7R_LENGTH_MATCHED_ACK_RESULTS.csv"
    length_manifest_path = ROOT / "P7R_LENGTH_CONTROL_MANIFEST.csv"
    source_path = evidence / "manifests" / "p5_source_trajectories.csv"
    neutral = [row for row in read_csv(neutral_path) if row["status"].lower() == "completed"]
    length = read_csv(length_path)
    length_manifest = read_csv(length_manifest_path)
    source = read_csv(source_path)

    neutral_map = {row["trajectory_id"]: row for row in neutral}
    length_map = {row["trajectory_id"]: row for row in length}
    manifest_ids = {row["trajectory_id"] for row in length_manifest}
    if len(neutral) != 2973 or len(neutral_map) != 2973:
        raise RuntimeError("RQ2 SHORT_ACK population/key mismatch")
    source_ids = {row["trajectory_id"] for row in source}
    if len(source) != 3033 or len(source_ids) != 3033 or not set(neutral_map) <= source_ids:
        raise RuntimeError("RQ2 SHORT_ACK rows do not align to the 3,033 replay-ready SHARDED sources")
    if len(length) != 300 or len(length_map) != 300 or len(manifest_ids) != 300:
        raise RuntimeError("RQ2 LENGTH_MATCHED candidate/key mismatch")
    if set(length_map) != manifest_ids or not manifest_ids <= set(neutral_map):
        raise RuntimeError("RQ2 length-control manifest/result/SHORT_ACK join mismatch")

    rows: list[dict] = []
    length_completed = 0
    length_primary = 0
    for trajectory_id in sorted(neutral_map):
        short = neutral_map[trajectory_id]
        sharded = number(short["original_sharded_score"])
        short_score = number(short["neutral_ack_score"])
        short_delta = number(short["effect"])
        if not close(short_delta, short_score - sharded):
            raise RuntimeError(f"RQ2 SHORT_ACK delta mismatch: {trajectory_id}")
        matched = length_map.get(trajectory_id)
        sampled = matched is not None
        status = matched["status"] if matched else ""
        completed = bool(matched and status.lower() == "completed")
        applied = bool(completed and matched["manipulation_applied"].lower() == "true")
        if completed:
            length_completed += 1
            if not close(matched["sharded_score"], sharded):
                raise RuntimeError(f"RQ2 matched SHARDED score mismatch: {trajectory_id}")
            if not close(matched["short_ack_score"], short_score):
                raise RuntimeError(f"RQ2 matched SHORT_ACK score mismatch: {trajectory_id}")
        if applied:
            length_primary += 1
        matched_score = number(matched["length_matched_ack_score"]) if completed else ""
        matched_delta = number(matched["length_matched_ack_minus_sharded"]) if completed else ""
        if completed and not close(matched_delta, number(matched_score) - sharded):
            raise RuntimeError(f"RQ2 matched delta mismatch: {trajectory_id}")
        task = short["task"]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "source_instance_id": short["instruction_id"],
                "model": short["model"],
                "task": task,
                "sharded_score": sharded,
                "short_ack_score": short_score,
                "short_ack_delta": short_delta,
                "length_matched_score": matched_score,
                "length_matched_delta": matched_delta,
                "task_type": "binary" if task in BINARY_TASKS else "continuous",
                "short_ack_paired_valid": "true",
                "length_matched_sampled": "true" if sampled else "false",
                "length_matched_status": status,
                "length_matched_manipulation_applied": (
                    "true" if applied else "false" if completed else ""
                ),
                "length_matched_primary_valid": "true" if applied else "false",
            }
        )
    if len(rows) != 2973 or length_completed != 276 or length_primary != 264:
        raise RuntimeError(
            f"RQ2 derived count mismatch: rows={len(rows)}, completed={length_completed}, primary={length_primary}"
        )
    write_csv(output / "rq2_paired_trajectory.csv", list(rows[0]), rows)
    return {
        "rows": len(rows),
        "replay_ready_sharded": len(source_ids),
        "short_ack_missing": len(source_ids - set(neutral_map)),
        "length_candidates": len(length_map),
        "length_completed": length_completed,
        "length_primary": length_primary,
    }


def build_rq3(evidence: Path, output: Path) -> dict:
    pool_path = evidence / "manifests" / "p5_turn_surgery_pool.csv"
    planned_path = evidence / "manifests" / "p5_turn_surgery_branches.csv"
    raw_path = evidence / "turn_surgery_v1_merged.jsonl"
    formal_path = ROOT / "P5_TURN_SURGERY_RESULTS.csv"
    pool = read_csv(pool_path)
    planned = read_csv(planned_path)
    formal = [row for row in read_csv(formal_path) if row["status"].lower() == "completed"]
    raw = [row for row in read_jsonl(raw_path) if row.get("status") == "completed"]

    pool_map = {row["trajectory_id"]: row for row in pool}
    planned_keys = {(row["trajectory_id"], int(row["surgery_turn"])) for row in planned}
    formal_map = {(row["trajectory_id"], int(row["surgery_turn"])): row for row in formal}
    raw_map = {(row["trajectory_id"], int(row["surgery_turn"])): row for row in raw}
    if len(pool) != 237 or len(pool_map) != 237:
        raise RuntimeError("RQ3 selected trajectory count/key mismatch")
    if len(planned) != 1472 or len(planned_keys) != 1472:
        raise RuntimeError("RQ3 planned branch count/key mismatch")
    if len(formal) != 1440 or len(formal_map) != 1440:
        raise RuntimeError("RQ3 formal branch count/key mismatch")
    if len(raw) != 1440 or len(raw_map) != 1440 or set(raw_map) != set(formal_map):
        raise RuntimeError("RQ3 raw/formal branch alignment mismatch")
    if not set(formal_map) <= planned_keys:
        raise RuntimeError("RQ3 completed branch absent from planned manifest")

    # Recover each original SHARDED trajectory from the exact path and run ID
    # frozen in the surgery pool. Paths are consumed locally and are not
    # emitted into the derived table.
    path_cache: dict[Path, list[dict]] = {}
    originals: dict[str, dict] = {}
    for trajectory_id, pool_row in pool_map.items():
        source_path = Path(pool_row["raw_sharded_path"])
        if not source_path.exists():
            raise RuntimeError(f"Missing frozen SHARDED source: {source_path}")
        if source_path not in path_cache:
            path_cache[source_path] = list(read_jsonl(source_path))
        matches = [
            row for row in path_cache[source_path]
            if row.get("task_id") == pool_row["instruction_id"]
            and (row.get("model_alias") or row.get("assistant_model")) == pool_row["tested_model"]
            and row.get("run_id") == pool_row["raw_sharded_run_id"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely recover original trajectory: {trajectory_id}")
        originals[trajectory_id] = matches[0]

    trajectory_views: dict[str, dict] = {}
    for trajectory_id, original in originals.items():
        semantic = [
            {"role": item["role"], "content": item["content"]}
            for item in original["trace"]
            if item.get("role") in {"system", "user", "assistant"}
        ]
        users = [item["content"] for item in semantic if item["role"] == "user"]
        assistants = [item["content"] for item in semantic if item["role"] == "assistant"]
        if len(users) != len(assistants) or len(assistants) < 2:
            raise RuntimeError(f"Invalid original turn structure: {trajectory_id}")
        pool_row = pool_map[trajectory_id]
        if len(assistants) != int(pool_row["assistant_turn_count"]):
            raise RuntimeError(f"Original assistant count mismatch: {trajectory_id}")
        trajectory_views[trajectory_id] = {
            "users": users,
            "assistants": assistants,
            "user_sequence_id": user_sequence_id(pool_row["instruction_id"], users),
        }

    rows: list[dict] = []
    for key in sorted(formal_map):
        trajectory_id, target_turn = key
        formal_row = formal_map[key]
        raw_row = raw_map[key]
        pool_row = pool_map[trajectory_id]
        view = trajectory_views[trajectory_id]
        assistants = view["assistants"]
        n_total = len(assistants)
        n_intermediate = n_total - 1
        if not 0 <= target_turn < n_intermediate:
            raise RuntimeError(f"RQ3 target assistant index out of range: {key}")
        original_score = number(formal_row["original_score"])
        surgery_score = number(formal_row["counterfactual_score"])
        delta = number(formal_row["effect"])
        if not close(delta, surgery_score - original_score):
            raise RuntimeError(f"RQ3 formal delta mismatch: {key}")
        if not (
            close(raw_row["original_sharded_score"], original_score)
            and close(raw_row["final_score"], surgery_score)
            and close(raw_row["effect"], delta)
        ):
            raise RuntimeError(f"RQ3 raw/formal score mismatch: {key}")
        history = raw_row.get("counterfactual_history", [])
        if not history or history[-1].get("role") != "assistant":
            raise RuntimeError(f"RQ3 surgery final output missing: {key}")
        task = formal_row["task"]
        binary = task in BINARY_TASKS
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "source_instance_id": formal_row["instruction_id"],
                "branch_id": raw_row["branch_id"],
                "model": formal_row["model"],
                "task": task,
                "target_turn": target_turn,
                "target_turn_paper_label": f"A{target_turn + 1}",
                "n_intermediate_turns": n_intermediate,
                "n_total_turns": n_total,
                "relative_position": target_turn / n_intermediate,
                "distance_to_final": n_intermediate - target_turn,
                "original_score": original_score,
                "surgery_score": surgery_score,
                "delta": delta,
                "effect_class": "positive" if delta > 0 else "negative" if delta < 0 else "zero",
                "task_type": "binary" if binary else "continuous",
                "fail_to_success": (
                    "true" if original_score == 0 and surgery_score == 1 else "false"
                ) if binary else "",
                "original_final_output": assistants[-1],
                "surgery_final_output": history[-1]["content"],
                "target_assistant_text": assistants[target_turn],
                "user_sequence_id": view["user_sequence_id"],
                "branch_status": "completed",
                "fixed_user_sequence": "true" if raw_row.get("fixed_user_sequence") else "false",
            }
        )
    if len(rows) != 1440:
        raise RuntimeError(f"RQ3 derived branch count differs: {len(rows)}")
    if len({(row["trajectory_id"], row["target_turn"]) for row in rows}) != 1440:
        raise RuntimeError("RQ3 derived primary key is not unique")
    if len({row["trajectory_id"] for row in rows}) != 237:
        raise RuntimeError("RQ3 derived trajectory coverage differs")
    if any(row["fixed_user_sequence"] != "true" for row in rows):
        raise RuntimeError("RQ3 contains a branch without frozen-user replay")
    write_csv(output / "rq3_turn_surgery_branches.csv", list(rows[0]), rows)
    return {
        "rows": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "planned_branches": len(planned_keys),
        "missing_branches": len(planned_keys - set(formal_map)),
        "recovered_original_trajectories": len(originals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rq1": build_rq1(args.evidence_root, args.output_dir),
        "rq2": build_rq2(args.evidence_root, args.output_dir),
        "rq3": build_rq3(args.evidence_root, args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
