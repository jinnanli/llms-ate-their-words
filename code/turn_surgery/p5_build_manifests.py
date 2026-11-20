"""Build frozen P5 source, NEUTRAL_ACK, surgery, and stability manifests.

This script is offline-only.  It does not import the model client and never
changes canonical raw logs.  Selection is deterministic by SHA256 ordering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

MODELS = [
    "gpt-5.6-luna", "grok-4.5", "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-14B",
]
TASKS = ["actions", "code", "data2text", "database", "math", "summary"]
MODES = ["full", "concat", "sharded"]
SUCCESS = {"completed", "reused_offline", "completed_after_retry", "reused_from_tranche"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def conversation_index(roots: list[Path]):
    index, duplicates = {}, []
    for root in roots:
        for path in root.rglob("*.jsonl"):
            if "upstream_conversations" not in path.parts:
                continue
            for row in read_jsonl(path):
                model = row.get("model_alias") or row.get("assistant_model")
                mode = row.get("mode") or str(row.get("conv_type", "")).split("-")[0]
                key = (row.get("task_id"), model, mode)
                if model not in MODELS or mode not in MODES:
                    continue
                candidate = (row, str(path.resolve()))
                if key in index and index[key][0].get("run_id") != row.get("run_id"):
                    duplicates.append((key, index[key][1], str(path)))
                    # Prefer formal P5/P1 version and a completed trace; never silently
                    # merge two genuinely different formal responses.
                    old = index[key][0]
                    old_formal = old.get("experiment_version") in {"p1-wave1-v5", "p5-layer-a-full-v1"}
                    new_formal = row.get("experiment_version") in {"p1-wave1-v5", "p5-layer-a-full-v1"}
                    if new_formal and not old_formal:
                        index[key] = candidate
                else:
                    index[key] = candidate
    return index, duplicates


def ledger_index(paths: list[Path]):
    index = {}
    for path in paths:
        for row in read_jsonl(path):
            model = row.get("assistant_model")
            key = (row.get("instruction_id") or row.get("case_id"), model, row.get("mode"))
            if model in MODELS and row.get("status") in SUCCESS and row.get("score") is not None:
                index[key] = row
    return index


def stable_key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-root", type=Path, required=True)
    ap.add_argument("--p5-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    p1ledger = args.p1_root / "audit" / "wave1_full_parallel_v5_merged" / "coverage_ledger.jsonl"
    p5ledgers = sorted(args.p5_root.rglob("coverage_ledger.jsonl"))
    ledger = ledger_index([p1ledger, *p5ledgers])
    conv, duplicates = conversation_index([args.p1_root, args.p5_root])
    rows, missing = [], []
    for model in MODELS:
        for task in TASKS:
            ids = sorted({k[0] for k, v in ledger.items() if k[1] == model and v.get("task") == task})
            for iid in ids:
                cells = {mode: ledger.get((iid, model, mode)) for mode in MODES}
                source = conv.get((iid, model, "sharded"))
                missing_modes = [m for m, value in cells.items() if not value]
                if missing_modes or source is None:
                    missing.append({"instruction_id": iid, "model": model, "task": task, "missing": ",".join(missing_modes) + (",raw_sharded" if source is None else "")})
                # NEUTRAL_ACK is defined for every readable, scored SHARDED
                # trajectory.  FULL/CONCAT missingness only limits paired
                # decomposition and surgery eligibility; it must not discard a
                # valid frozen replay source.
                if cells["sharded"] is None or source is None:
                    continue
                trace = source[0].get("trace", [])
                assistant_turns = sum(x.get("role") == "assistant" for x in trace)
                row = {
                    "trajectory_id": stable_key(iid, model)[:24], "instruction_id": iid,
                    "task": task, "tested_model": model,
                    "FULL_score": float(cells["full"]["score"]) if cells["full"] else "",
                    "CONCAT_score": float(cells["concat"]["score"]) if cells["concat"] else "",
                    "SHARDED_score": float(cells["sharded"]["score"]),
                    "complete_case": bool(all(cells.values())),
                    "assistant_turn_count": assistant_turns,
                    "raw_sharded_path": source[1], "raw_sharded_run_id": source[0].get("run_id"),
                    "repeat_index": 0,
                }
                rows.append(row)
    if (missing or duplicates) and not args.allow_missing:
        raise SystemExit(f"Integrity check failed: {len(missing)} missing, {len(duplicates)} duplicate candidates")
    write_csv(args.output_dir / "p5_source_trajectories.csv", rows)
    write_csv(args.output_dir / "p5_source_missing.csv", missing)

    # Eligibility is frozen before surgery results.  Binary tasks require a
    # fail-after-success transition.  Continuous tasks use a prespecified
    # absolute 0.10 score loss on their native [0,1] scale.
    eligible = []
    for row in rows:
        if not row["complete_case"]:
            continue
        binary = row["task"] in {"actions", "code", "database", "math"}
        include = ((row["FULL_score"] >= 1 and row["SHARDED_score"] < 1) if binary
                   else (row["FULL_score"] - row["SHARDED_score"] >= 0.10))
        if include and int(row["assistant_turn_count"]) >= 2:
            eligible.append(row)
    pool = []
    buckets = defaultdict(list)
    for row in eligible:
        buckets[(row["tested_model"], row["task"])].append(row)
    for key, values in sorted(buckets.items()):
        values.sort(key=lambda r: stable_key("pool-v1", r["trajectory_id"]))
        pool.extend(values[:8])
    pool.sort(key=lambda r: (MODELS.index(r["tested_model"]), TASKS.index(r["task"]), r["trajectory_id"]))
    write_csv(args.output_dir / "p5_turn_surgery_pool.csv", pool)
    branches = []
    for row in pool:
        for turn in range(int(row["assistant_turn_count"]) - 1):
            branches.append({**row, "surgery_turn": turn})
    write_csv(args.output_dir / "p5_turn_surgery_branches.csv", branches)
    stability = sorted(pool, key=lambda r: stable_key("stability-v1", r["trajectory_id"]))[:60]
    write_csv(args.output_dir / "p5_turn_surgery_stability_cases.csv", stability)
    manifest = {
        "source_count": len(rows), "missing_count": len(missing),
        "duplicate_candidate_count": len(duplicates), "surgery_pool_count": len(pool),
        "surgery_branch_count": len(branches), "stability_case_count": len(stability),
        "continuous_meaningful_degradation": "FULL-SHARDED >= 0.10 on native [0,1] score",
        "binary_eligibility": "FULL == 1 and SHARDED < 1",
        "selection": "up to 8 per model-task by deterministic SHA256 ordering",
        "stability_selection": "60 cases by deterministic SHA256 ordering before effects",
    }
    (args.output_dir / "p5_manifest_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
