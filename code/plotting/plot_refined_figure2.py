"""Render the final compact Figure 2 only from its frozen plotting table.

This offline renderer changes canvas geometry and whitespace only. It does
not estimate or modify any score, confidence interval, denominator, or cell.
Figure 1 is intentionally outside this script's execution path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reproduced" / "figures"
DATA = ROOT / "data" / "figures" / "figure2_plot_data.csv"

MODELS = [
    "gpt-5.6-luna", "grok-4.5", "gpt-4o",
    "deepseek-ai/DeepSeek-V3.2", "Qwen/Qwen3-14B",
]
MODEL_LABELS = ["GPT-5.6 Luna", "Grok-4.5", "GPT-4o", "DeepSeek-V3.2", "Qwen3-14B"]
TASKS = ["actions", "database", "data2text", "code", "math", "summary"]
TASK_LABELS = ["Actions", "Database", "Data-to-Text", "Code", "Math", "Summary"]
COLORS = {"FULL": "#3569B7", "CONCAT": "#D98B2B", "SHARDED": "#B33A3A"}


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.1,
                 fontweight="semibold", pad=2)


def save(fig) -> None:
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(OUT / f"Figure2_task_model_landscape.{suffix}",
                    facecolor="white", bbox_inches="tight", pad_inches=.018, **kwargs)
    plt.close(fig)


def main() -> None:
    rows = read_csv(DATA)
    if len(rows) != 45:
        raise ValueError(f"Expected 45 frozen Figure 2 rows; found {len(rows)}")
    rows_a = {(r["model"], r["mode"]): r for r in rows if r["panel"] == "A"}
    rows_b = {(r["task"], r["model"]): r for r in rows if r["panel"] == "B"}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.4, "axes.labelsize": 6.4,
        "xtick.labelsize": 5.7, "ytick.labelsize": 5.7, "axes.linewidth": .65,
        "xtick.major.width": .55, "ytick.major.width": .55, "lines.linewidth": .9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.10, 2.18), layout="constrained")
    fig.get_layout_engine().set(w_pad=.008, h_pad=.002, wspace=.025, hspace=.002)
    grid = fig.add_gridspec(1, 2, width_ratios=[.40, .60])
    axis_a, axis_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    x = np.arange(len(MODELS))
    for mode, offset, marker in [("FULL", -.18, "o"), ("CONCAT", 0, "s"), ("SHARDED", .18, "D")]:
        mean = np.array([f(rows_a[m, mode]["task_macro_mean"]) for m in MODELS])
        low = np.array([f(rows_a[m, mode]["CI_low"]) for m in MODELS])
        high = np.array([f(rows_a[m, mode]["CI_high"]) for m in MODELS])
        axis_a.errorbar(x + offset, mean, yerr=np.vstack([mean - low, high - mean]),
                        fmt=marker, ms=3.7, color=COLORS[mode], capsize=1.8,
                        elinewidth=.8, label=mode)
    axis_a.set_xticks(x, MODEL_LABELS, rotation=20, ha="right")
    axis_a.set_ylabel("Task-macro score")
    axis_a.set_ylim(.39, .84)
    axis_a.grid(axis="y", alpha=.20)
    axis_a.legend(frameon=False, ncol=3, fontsize=5.8, handletextpad=.25,
                  columnspacing=.65, loc="lower left", borderaxespad=.2)
    panel_title(axis_a, "A", "Overall performance")

    values = np.array([[f(rows_b[t, m]["FULL_minus_SHARDED"]) for m in MODELS]
                       for t in TASKS])
    norm = TwoSlopeNorm(vmin=-.05, vcenter=0, vmax=.45)
    image = axis_b.pcolormesh(np.arange(6) - .5, np.arange(7) - .5, values,
                              cmap="RdBu_r", norm=norm, shading="flat", linewidth=0)
    for i, task in enumerate(TASKS):
        for j, model in enumerate(MODELS):
            row = rows_b[task, model]
            dagger = f(row["CI_low"]) <= 0 <= f(row["CI_high"])
            axis_b.text(j, i, f"{values[i, j]:.2f}" + ("†" if dagger else ""),
                        ha="center", va="center", fontsize=6.0,
                        color="white" if values[i, j] >= .285 else "#111111")
    axis_b.set_xticks(range(5), MODEL_LABELS, rotation=20, ha="right")
    axis_b.set_yticks(range(6), TASK_LABELS)
    axis_b.set_xlim(-.5, 4.5)
    axis_b.set_ylim(5.5, -.5)
    panel_title(axis_b, "B", "FULL − SHARDED gap by task and model")
    colorbar = fig.colorbar(image, ax=axis_b, fraction=.045, pad=.018)
    colorbar.set_label("Score gap", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=5.7)
    save(fig)


if __name__ == "__main__":
    main()
