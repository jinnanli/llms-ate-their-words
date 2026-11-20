"""Build deterministic 1:1 and 1:3 S2 controls from held-out native behavior."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> dict[tuple[str, int], dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {(row["trajectory_id"], int(row["target_turn"])): row for row in csv.DictReader(handle)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    surgeries = read_jsonl(args.native_dir / "server_native_surgeries.jsonl")
    manifest = read_csv(args.native_dir / "formal_heldout_branch_manifest.csv")
    enriched = []
    for row in surgeries:
        key = (row["trajectory_id"], int(row["target_turn"]))
        metadata = manifest[key]
        enriched.append(
            {
                **row,
                "relative_position": float(metadata["relative_position"]),
                "n_user_turns": int(metadata["n_user_turns"]),
                "context_length": int(metadata["original_context_token_count"]),
            }
        )
    primary = sorted(
        [
            row for row in enriched
            if row["task"] in {"actions", "code"}
            and float(row["original_score_local"]) == 0.0
            and float(row["surgery_score_local"]) == 1.0
        ],
        key=lambda row: (row["task"], row["trajectory_id"], int(row["target_turn"])),
    )
    zeros = [row for row in enriched if abs(float(row["delta_local"])) <= 1e-12]

    outputs = {}
    for ratio in (1, 3):
        used = set()
        selected = []
        for match_index, row in enumerate(primary):
            candidates = [
                candidate for candidate in zeros
                if candidate["task"] == row["task"]
                and (candidate["trajectory_id"], int(candidate["target_turn"])) not in used
            ]
            candidates.sort(
                key=lambda candidate: (
                    abs(candidate["relative_position"] - row["relative_position"]),
                    abs(candidate["n_user_turns"] - row["n_user_turns"]),
                    abs(candidate["context_length"] - row["context_length"]),
                    abs(float(candidate["original_score_local"]) - float(row["original_score_local"])),
                    candidate["trajectory_id"], int(candidate["target_turn"]),
                )
            )
            chosen = candidates[:ratio]
            if len(chosen) < ratio:
                if ratio == 1:
                    raise RuntimeError(f"No 1:1 control for {row['trajectory_id']}:{row['target_turn']}")
                continue
            match_id = f"{row['trajectory_id']}:{row['target_turn']}"
            selected.append({"match_id": match_id, "s2_group": "fail_to_success", **row})
            for control_rank, control in enumerate(chosen, start=1):
                used.add((control["trajectory_id"], int(control["target_turn"])))
                selected.append(
                    {"match_id": match_id, "control_rank": control_rank, "s2_group": "matched_nochange", **control}
                )
        outputs[ratio] = selected
        write_jsonl(args.output_dir / f"s2_population_1to{ratio}.jsonl", selected)

    summary = {
        "population_version": "FORMAL_HELDOUT_118",
        "fail_to_success": len(primary),
        "fail_to_success_by_task": dict(Counter(row["task"] for row in primary)),
        "one_to_one_rows": len(outputs[1]),
        "one_to_one_controls": sum(row["s2_group"] == "matched_nochange" for row in outputs[1]),
        "one_to_three_rows": len(outputs[3]),
        "one_to_three_primary": sum(row["s2_group"] == "fail_to_success" for row in outputs[3]),
        "one_to_three_controls": sum(row["s2_group"] == "matched_nochange" for row in outputs[3]),
        "matching_hierarchy": ["same_task", "relative_position", "user_turn_count", "context_length", "baseline_score"],
    }
    (args.output_dir / "s2_population_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

