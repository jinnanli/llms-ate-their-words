"""Build the 131-branch server-native state-propagation manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    surgery_path = args.native_dir / "server_native_surgeries.jsonl"
    rows = [json.loads(line) for line in surgery_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        history = row["surgery_history"]
        n_user_turns = sum(message["role"] == "user" for message in history)
        target_turn = int(row["target_turn"])
        n_downstream = n_user_turns - target_turn - 1
        if n_downstream < 1:
            raise RuntimeError(f"No downstream user anchor for {row['trajectory_id']}:{target_turn}")
        output.append(
            {
                "trajectory_id": row["trajectory_id"],
                "target_turn": target_turn,
                "task": row["task"],
                "local_effect_class": row["effect_class_local"],
                "local_delta": row["delta_local"],
                "local_original_score": row["original_score_local"],
                "local_surgery_score": row["surgery_score_local"],
                "formal_effect_class": row["formal_effect_class"],
                "formal_original_score": row["formal_original_score"],
                "formal_surgery_score": row["formal_surgery_score"],
                "n_user_turns": n_user_turns,
                "n_downstream_user_anchors": n_downstream,
                "native_surgery_file": str(surgery_path),
                "native_surgery_sha256": sha256(surgery_path),
            }
        )
    output.sort(key=lambda row: (row["trajectory_id"], row["target_turn"]))
    if len(output) != 131 or len({(row["trajectory_id"], row["target_turn"]) for row in output}) != 131:
        raise RuntimeError(f"Expected 131 unique branches, found {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"branches": len(output), "trajectories": len({row['trajectory_id'] for row in output})}))


if __name__ == "__main__":
    main()

