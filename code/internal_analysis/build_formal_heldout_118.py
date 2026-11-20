"""Validate the recovery packet and build load-balanced FORMAL_HELDOUT_118 shards."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


POPULATION_VERSION = "FORMAL_HELDOUT_118"
TASKS = ("actions", "code", "data2text")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "evaluation"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_discovery_users(path: Path) -> dict[str, dict]:
    output = {}
    for row in read_jsonl(path):
        systems = [item["content"] for item in row["generated_history"] if item["role"] == "system"]
        users = [item["content"] for item in row["generated_history"] if item["role"] == "user"]
        if len(systems) != 1:
            raise RuntimeError(f"Discovery system-message count is not one: {row['trajectory_id']}")
        output[row["trajectory_id"]] = {"task": row["task"], "system": systems[0], "users": users}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--discovery-baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    required = [
        "FORMAL_UNCAPPED_DEGRADED_QWEN.csv",
        "ACTIONS_CODE_HISTORICAL_USER_MESSAGES.jsonl",
        "USER_SIMULATOR_REPLAY_CONFIG.md",
        "RECOVERY_SOURCE_MAP.md",
        "REMOTE_REPLICATION_RECOVERY_SUMMARY.md",
    ]
    missing = [name for name in required if not (args.packet_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Recovery packet is incomplete: {missing}")

    formal_path = args.packet_dir / required[0]
    messages_path = args.packet_dir / required[1]
    formal = read_csv(formal_path)
    messages = {row["trajectory_id"]: row for row in read_jsonl(messages_path)}
    samples = {row["task_id"]: row for row in json.loads(args.dataset.read_text(encoding="utf-8"))}
    discovery = load_discovery_users(args.discovery_baselines)

    if len(formal) != 142 or len({row["trajectory_id"] for row in formal}) != 142:
        raise RuntimeError("Expected 142 unique recovered formal-degraded trajectories")
    if Counter(row["task"] for row in formal) != Counter(actions=41, code=33, data2text=68):
        raise RuntimeError("Recovered task totals do not equal 41/33/68")
    discovery_ids = {row["trajectory_id"] for row in formal if row["discovery_24"] == "YES"}
    heldout_rows = [row for row in formal if row["heldout_118"] == "YES"]
    if discovery_ids != set(discovery) or len(heldout_rows) != 118:
        raise RuntimeError("Recovery packet does not exactly partition the discovery 24 and held-out 118")
    if Counter(row["task"] for row in heldout_rows) != Counter(actions=33, code=25, data2text=60):
        raise RuntimeError("Held-out task totals do not equal 33/25/60")
    if len(messages) != 74 or set(messages) != {
        row["trajectory_id"] for row in formal if row["task"] in {"actions", "code"}
    }:
        raise RuntimeError("Actions/Code stored-message coverage is not exactly 74/74")

    from tasks.data2text.task_data2text import TaskData2Text

    d2t = TaskData2Text()
    resolved = {}
    reference_missing = []
    for row in formal:
        trajectory_id = row["trajectory_id"]
        sample = samples.get(row["instance_id"])
        if sample is None:
            reference_missing.append(trajectory_id)
            continue
        if row["task"] in {"actions", "code"}:
            source = messages[trajectory_id]
            system = source["system_message"]
            users = source["user_messages"]
            user_source = "RECOVERED_STORED_HISTORICAL_MESSAGES"
        else:
            system = d2t.generate_system_prompt(sample)
            users = [item["shard"] for item in sample["shards"]]
            user_source = "DETERMINISTIC_D2T_SHARDS"
        if not system or len(users) < 2 or any(not isinstance(user, str) for user in users):
            raise RuntimeError(f"Invalid user sequence for {trajectory_id}")
        resolved[trajectory_id] = {
            "system": system,
            "users": users,
            "user_message_source": user_source,
        }
    if reference_missing:
        raise RuntimeError(f"Missing exact evaluator references: {reference_missing}")

    exact_by_task = Counter()
    mismatches = []
    for trajectory_id, expected in discovery.items():
        actual = resolved[trajectory_id]
        exact = expected["system"] == actual["system"] and expected["users"] == actual["users"]
        if exact:
            exact_by_task[expected["task"]] += 1
        else:
            mismatches.append(trajectory_id)
    if mismatches or exact_by_task != Counter(actions=8, code=8, data2text=8):
        raise RuntimeError(f"Discovery user-sequence reconstruction failed: {mismatches}")

    population = []
    payloads = []
    for row in heldout_rows:
        trajectory_id = row["trajectory_id"]
        source = resolved[trajectory_id]
        user_count = len(source["users"])
        population.append(
            {
                "trajectory_id": trajectory_id,
                "instance_id": row["instance_id"],
                "task": row["task"],
                "formal_full_score": row["full_score"],
                "formal_sharded_score": row["sharded_score"],
                "formal_degraded": "true",
                "discovery_24": "false",
                "heldout_118": "true",
                "user_message_count": user_count,
                "eligible_target_turn_count": user_count - 1,
                "reference_id": row["instance_id"],
                "user_message_source": source["user_message_source"],
                "population_version": POPULATION_VERSION,
                "source_shard": "",
            }
        )
        payloads.append(
            {
                "trajectory_id": trajectory_id,
                "source_instance_id": row["instance_id"],
                "task": row["task"],
                "population_version": POPULATION_VERSION,
                "system_message": source["system"],
                "user_messages": source["users"],
            }
        )

    # Largest-first deterministic assignment balances the expected surgery load.
    shard_loads = [0] * args.num_shards
    shard_rows: list[list[dict]] = [[] for _ in range(args.num_shards)]
    ordered = sorted(
        population,
        key=lambda row: (-int(row["eligible_target_turn_count"]), row["task"], row["trajectory_id"]),
    )
    for row in ordered:
        shard_index = min(range(args.num_shards), key=lambda index: (shard_loads[index], index))
        row["source_shard"] = f"heldout_shard_{shard_index:02d}"
        shard_rows[shard_index].append(row)
        shard_loads[shard_index] += int(row["eligible_target_turn_count"])
    shard_by_trajectory = {row["trajectory_id"]: row["source_shard"] for row in population}
    for payload in payloads:
        payload["source_shard"] = shard_by_trajectory[payload["trajectory_id"]]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "formal_heldout_118_population.csv", sorted(population, key=lambda row: row["trajectory_id"]))
    write_jsonl(args.output_dir / "formal_heldout_118_source_payloads.jsonl", sorted(payloads, key=lambda row: row["trajectory_id"]))
    payload_by_id = {row["trajectory_id"]: row for row in payloads}
    for index, rows in enumerate(shard_rows):
        rows = sorted(rows, key=lambda row: row["trajectory_id"])
        write_csv(args.output_dir / "behavior_shards" / f"heldout_shard_{index:02d}.csv", rows)
        write_jsonl(
            args.output_dir / "behavior_shards" / f"heldout_shard_{index:02d}.jsonl",
            [payload_by_id[row["trajectory_id"]] for row in rows],
        )
    summary = {
        "population_version": POPULATION_VERSION,
        "packet_sha256": {name: sha256(args.packet_dir / name) for name in required},
        "discovery_revalidation": {"actions": "8/8", "code": "8/8", "data2text": "8/8", "total": "24/24"},
        "evaluator_reference_coverage": "142/142",
        "heldout_trajectories": len(population),
        "heldout_task_counts": dict(Counter(row["task"] for row in population)),
        "expected_branches": sum(int(row["eligible_target_turn_count"]) for row in population),
        "expected_branches_by_task": dict(
            Counter({task: sum(int(row["eligible_target_turn_count"]) for row in population if row["task"] == task) for task in TASKS})
        ),
        "shards": [
            {
                "source_shard": f"heldout_shard_{index:02d}",
                "trajectories": len(rows),
                "expected_branches": shard_loads[index],
                "tasks": dict(Counter(row["task"] for row in rows)),
            }
            for index, rows in enumerate(shard_rows)
        ],
    }
    (args.output_dir / "population_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

