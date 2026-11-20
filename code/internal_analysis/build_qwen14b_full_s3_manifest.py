"""Build the full 131-branch native-history S3 manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPECTED_CLASSES = {"positive": 26, "zero": 93, "negative": 12}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.source_manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["trajectory_id"], int(row["target_turn"])) for row in rows}
    classes = Counter(row["local_effect_class"] for row in rows)
    if len(rows) != 131 or len(keys) != 131:
        raise RuntimeError(f"Expected 131 unique branches, found {len(rows)} rows and {len(keys)} keys")
    if dict(classes) != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected local effect-class composition: {dict(classes)}")

    output = []
    for row in rows:
        output.append(
            {
                **row,
                "primary_label_source": "server_native_local_hf",
                "knockout_context_source": "server_native_generated_history",
            }
        )
    output.sort(key=lambda row: (row["trajectory_id"], int(row["target_turn"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"branches": len(output), "trajectories": len({row['trajectory_id'] for row in output}), "classes": classes}))


if __name__ == "__main__":
    main()

