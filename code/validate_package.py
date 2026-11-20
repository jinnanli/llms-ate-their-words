"""Lightweight integrity checks for the preserved supplementary core."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    required = [
        "README.md", "MANIFEST.txt", "requirements.txt",
        "data/rq1/rq1_cell_summary.csv",
        "data/rq2/rq2_paired_trajectory.csv",
        "data/rq2/length_control_manifest.csv",
        "data/rq3/turn_surgery_branches_compact.csv",
        "data/rq3/trajectory_sensitivity_summary.csv",
        "data/rq4/formal_heldout_118_population.csv",
        "data/rq4/formal_heldout_branch_manifest.csv",
        "data/figures/figure2_plot_data.csv",
        "data/figures/figure3_plot_data.csv",
        "data/figures/figure4_plot_data.csv",
        "data/figures/figure5_plot_data.csv",
        "results/combined/LOCAL_BEHAVIORAL_FINAL_NUMBERS.json",
        "results/rq4/OPEN_MODEL_EXTENSION_V3_FIGURE_DATA.csv",
    ]
    for relative in required:
        path = ROOT / relative
        require(path.is_file(), f"Missing required file: {relative}")
        require(path.stat().st_size > 0, f"Empty required file: {relative}")

    expected_rows = {
        "data/rq1/rq1_cell_summary.csv": 30,
        "data/rq2/rq2_paired_trajectory.csv": 2973,
        "data/rq2/length_control_manifest.csv": 300,
        "data/rq3/turn_surgery_branches_compact.csv": 1440,
        "data/rq3/trajectory_sensitivity_summary.csv": 237,
        "data/rq4/formal_heldout_118_population.csv": 118,
        "data/rq4/formal_heldout_branch_manifest.csv": 603,
        "data/figures/figure2_plot_data.csv": 45,
        "data/figures/figure3_plot_data.csv": 13,
        "data/figures/figure4_plot_data.csv": 17,
        "data/figures/figure5_plot_data.csv": 59,
    }
    loaded = {path: rows(path) for path in expected_rows}
    for relative, expected in expected_rows.items():
        require(len(loaded[relative]) == expected,
                f"Row-count mismatch for {relative}: {len(loaded[relative])} != {expected}")

    rq2 = loaded["data/rq2/rq2_paired_trajectory.csv"]
    require(len({r["trajectory_id"] for r in rq2}) == 2973,
            "RQ2 trajectory_id is not unique")
    rq3 = loaded["data/rq3/turn_surgery_branches_compact.csv"]
    require(len({(r["trajectory_id"], r["target_turn"]) for r in rq3}) == 1440,
            "RQ3 (trajectory_id, target_turn) is not unique")
    require(len({r["trajectory_id"] for r in rq3}) == 237,
            "RQ3 trajectory count is not 237")
    rq4 = loaded["data/rq4/formal_heldout_branch_manifest.csv"]
    require(len({(r["trajectory_id"], r["target_turn"]) for r in rq4}) == 603,
            "RQ4 held-out branch key is not unique")

    fig3 = {(r["panel"], r["category"]): r
            for r in loaded["data/figures/figure3_plot_data.csv"]}
    anchors = {
        ("A", "short_minus_sharded"): .06940202330079084,
        ("A", "lm_minus_sharded"): .06793130623514171,
        ("A", "lm_minus_short"): -.0014707170656491429,
    }
    for key, expected in anchors.items():
        require(math.isclose(float(fig3[key]["estimate"]), expected, abs_tol=1e-15),
                f"Frozen Figure 3 anchor changed: {key}")
    require(fig3[("A", "short_minus_sharded")]["denominator"] == "264",
            "Figure 3 length-control denominator changed")

    for relative in [
        "results/figures/Figure1_four_rq_framework.pdf",
        "results/figures/Figure2_task_model_landscape.pdf",
        "results/figures/Figure3_history_intervention.pdf",
        "results/figures/Figure4_turn_surgery_structure.pdf",
        "results/figures/Figure5_internal_signatures.pdf",
    ]:
        require((ROOT / relative).stat().st_size > 1000, f"Invalid figure PDF: {relative}")

    json.loads((ROOT / "results/combined/LOCAL_BEHAVIORAL_FINAL_NUMBERS.json").read_text(encoding="utf-8-sig"))
    print("PACKAGE_VALIDATION: PASS")
    print("RQ1 cells=30; RQ2 pairs=2973; RQ3 trajectories=237, branches=1440; RQ4 held-out trajectories=118, branches=603")


if __name__ == "__main__":
    main()
