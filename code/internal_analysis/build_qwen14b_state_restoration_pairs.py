"""Build deterministic positive/zero pairs for the v3 state-restoration scan."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=46)
    parser.add_argument(
        "--population-version",
        default="FORMAL_HELDOUT_118_STATE_RESTORATION_V3",
    )
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.branches):
        grouped[row["trajectory_id"]].append(row)

    output = []
    pair_number = 0
    for trajectory_id, rows in sorted(grouped.items()):
        positives = [row for row in rows if row["local_effect_class"] == "positive"]
        zeros = [row for row in rows if row["local_effect_class"] == "zero"]
        if not positives or not zeros:
            continue
        candidates = []
        for positive in positives:
            for zero in zeros:
                candidates.append(
                    (
                        abs(float(positive["relative_position"]) - float(zero["relative_position"])),
                        int(positive["target_turn"]),
                        int(zero["target_turn"]),
                        positive,
                        zero,
                    )
                )
        _, _, _, positive, zero = min(candidates, key=lambda item: item[:3])
        pair_id = f"SR-{pair_number:03d}"
        pair_number += 1
        relative_gap = abs(float(positive["relative_position"]) - float(zero["relative_position"]))
        for role, row in (("positive", positive), ("matched_zero", zero)):
            output.append(
                {
                    "pair_id": pair_id,
                    "trajectory_id": trajectory_id,
                    "task": row["task"],
                    "sample_role": role,
                    "target_turn": int(row["target_turn"]),
                    "local_effect_class": row["local_effect_class"],
                    "local_delta": row["local_delta"],
                    "original_score": row["original_score"],
                    "surgery_score": row["surgery_score"],
                    "relative_position": row["relative_position"],
                    "distance_to_final": row["distance_to_final"],
                    "n_user_turns": row["n_user_turns"],
                    "pair_relative_position_gap": relative_gap,
                    "selection_rule": "min_abs_relative_position_then_target_turn",
                    "population_version": args.population_version,
                }
            )

    if pair_number != args.expected_pairs:
        raise RuntimeError(f"Expected {args.expected_pairs} mixed trajectories, found {pair_number}")
    if len({(row["trajectory_id"], row["target_turn"]) for row in output}) != len(output):
        raise RuntimeError("Duplicate trajectory/target key in restoration population")
    if any(len(rows) != 2 for rows in (
        [row for row in output if row["pair_id"] == pair_id]
        for pair_id in {row["pair_id"] for row in output}
    )):
        raise RuntimeError("Every restoration pair must contain exactly two branches")

    write_csv(args.output, output)
    task_counts = Counter(row["task"] for row in output if row["sample_role"] == "positive")
    print({"pairs": pair_number, "branches": len(output), "tasks": dict(sorted(task_counts.items()))})


if __name__ == "__main__":
    main()

