"""Render the final layout refinements for Figures 3--5 only.

This is an offline visualization script. It reads frozen plotting tables and
the frozen 1,440-row Turn Surgery branch table. It does not recompute any
reported estimate, confidence interval, denominator, or scientific result.
Figures 1 and 2 are intentionally outside this script's execution path.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reproduced" / "figures"
DATA = ROOT / "data" / "figures"
BRANCH_SOURCE = ROOT / "data" / "rq3" / "turn_surgery_branches_compact.csv"
POSITION_SOURCE = ROOT / "data" / "rq3" / "position_summary.csv"

TASKS = ["actions", "database", "data2text", "code", "math", "summary"]
TASK_LABELS = ["Actions", "Database", "Data-to-Text", "Code", "Math", "Summary"]

COLORS = {
    "SHORT": "#4C78A8",
    "MATCHED": "#59A14F",
    "POSITIVE": "#4E9A55",
    "ZERO": "#C7C4C0",
    "NEGATIVE": "#C95B58",
    "STATE": "#7966A8",
    "FORMAL": "#4C78A8",
    "PROSPECTIVE": "#E18B37",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty plotting table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str) -> float:
    return float(value)


def panel_title(ax, letter: str, title: str, *, pad: float = 3) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=8.1,
                 fontweight="semibold", pad=pad)


def err(ax, x, estimate, low, high, *, horizontal=False, **kwargs) -> None:
    if horizontal:
        ax.errorbar(estimate, x, xerr=[[estimate - low], [high - estimate]],
                    capsize=2.0, elinewidth=0.8, capthick=0.7, **kwargs)
    else:
        ax.errorbar(x, estimate, yerr=[[estimate - low], [high - estimate]],
                    capsize=2.0, elinewidth=0.8, capthick=0.7, **kwargs)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def export_distance_scatter() -> list[dict[str, str]]:
    """Copy only frozen fields needed by Figure 4D into its plotting table."""
    source_rows = read_csv(BRANCH_SOURCE)
    if len(source_rows) != 1440:
        raise ValueError(f"Expected 1,440 frozen valid branches; found {len(source_rows)}")
    keep = [
        "trajectory_id", "branch_id", "model", "task", "target_turn",
        "distance_to_final", "delta", "effect_class",
    ]
    rows = [{key: row[key] for key in keep} for row in source_rows]
    write_csv(DATA / "figure4_distance_scatter.csv", rows)
    return rows


def figure3() -> None:
    data = read_csv(DATA / "figure3_plot_data.csv")
    fig, axes = plt.subplots(
        2, 1, figsize=(3.42, 2.30), layout="constrained",
        gridspec_kw={"height_ratios": [.92, 1.08]},
    )
    fig.get_layout_engine().set(w_pad=.006, h_pad=0, wspace=0, hspace=.035)

    # A: horizontal CIs keep estimates and annotations structurally separated.
    lookup = {r["category"]: r for r in data if r["panel"] == "A"}
    paired_y = [.72, .43]
    for y, key, color in [
        (paired_y[0], "short_minus_sharded", COLORS["SHORT"]),
        (paired_y[1], "lm_minus_sharded", COLORS["MATCHED"]),
    ]:
        row = lookup[key]
        estimate, low, high = f(row["estimate"]), f(row["ci_low"]), f(row["ci_high"])
        err(axes[0], y, estimate, low, high, horizontal=True,
            fmt="o", ms=4.6, color=color)
        axes[0].text(estimate, y + .095, f"{estimate:+.3f}",
                     ha="center", va="bottom", fontsize=6.2)
    contrast = lookup["lm_minus_short"]
    axes[0].text(
        .63, .075,
        "MATCHED − SHORT\n"
        f"{f(contrast['estimate']):+.3f} "
        f"[{f(contrast['ci_low']):+.3f}, {f(contrast['ci_high']):+.3f}]",
        transform=axes[0].transAxes, ha="center", va="bottom", fontsize=5.5,
        bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#AAAAAA", lw=.5),
    )
    axes[0].axvline(0, color="#333333", lw=.7)
    axes[0].set_xlim(-.008, .122)
    axes[0].set_ylim(.02, .96)
    axes[0].set_yticks(paired_y, ["SHORT", "MATCHED"])
    axes[0].set_xlabel("Effect vs SHARDED", labelpad=1)
    axes[0].grid(axis="x", alpha=.15)
    axes[0].text(.02, .94, "Common subset, n=264", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=5.5, color="#555555")
    panel_title(axes[0], "A", "Length control", pad=1.0)

    # B: frozen transition values, now explicitly conditional on baseline state.
    rows_b = [r for r in data if r["panel"] == "B"]
    groups = ["SHARDED failure", "SHARDED success"]
    palette = {
        "Rescued 0→1": COLORS["POSITIVE"],
        "Persistent failure 0→0": COLORS["ZERO"],
        "Harmed 1→0": COLORS["NEGATIVE"],
        "Persistent success 1→1": COLORS["ZERO"],
    }
    stratum_y = [0.0, .52]
    for yi, group in zip(stratum_y, groups):
        left = 0.0
        for row in [r for r in rows_b if r["category"] == group]:
            width = f(row["estimate"])
            axes[1].barh(yi, width, left=left, height=.34,
                         color=palette[row["subgroup"]], edgecolor="white", linewidth=.5)
            if width > .12:
                detail = (f"{100 * width:.1f}%" if row["subgroup"] == "Harmed 1→0"
                          else f"{100 * width:.1f}%\n{row['numerator']}/{row['denominator']}")
                axes[1].text(
                    left + width / 2, yi, detail, ha="center", va="center", fontsize=4.9,
                    color="white" if row["subgroup"] in {"Rescued 0→1", "Harmed 1→0"}
                    else "#222222",
                )
            left += width
    axes[1].set_yticks(
        stratum_y,
        ["SHARDED failure\n(n=595)", "SHARDED success\n(n=1329)"],
    )
    axes[1].tick_params(axis="y", labelsize=5.25)
    axes[1].invert_yaxis()
    axes[1].set_ylim(.80, -.28)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Conditional percentage within baseline state")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(100*x)}%"))
    axes[1].legend(
        handles=[
            Patch(facecolor=COLORS["POSITIVE"], edgecolor="none", label="green: rescued"),
            Patch(facecolor=COLORS["NEGATIVE"], edgecolor="none", label="red: harmed"),
            Patch(facecolor=COLORS["ZERO"], edgecolor="none", label="gray: unchanged"),
        ],
        frameon=False, fontsize=4.05, ncol=3, loc="center",
        bbox_to_anchor=(.5, .50), handlelength=.75, handletextpad=.25,
        columnspacing=.35, borderaxespad=0,
    )
    panel_title(axes[1], "B", "Binary transitions (conditional)", pad=1.0)

    save(fig, "Figure3_history_intervention")


def figure4() -> None:
    data = read_csv(DATA / "figure4_plot_data.csv")
    distance_rows = [r for r in read_csv(POSITION_SOURCE)
                     if r["summary_type"] == "distance_bin"]
    if len(distance_rows) != 8:
        raise ValueError(f"Expected 8 frozen distance bins; found {len(distance_rows)}")
    fig, axes = plt.subplots(
        1, 4, figsize=(7.2, 2.08), layout="constrained",
        gridspec_kw={"width_ratios": [.82, .94, 1.40, 1.62]},
    )
    fig.get_layout_engine().set(w_pad=.010, h_pad=.002, wspace=.032, hspace=.002)
    base = {r.get("metric"): r for r in data if r["panel"] in {"A", "B"}}

    # A: frozen effects at compact paired positions.
    paired_x = [-.15, .15]
    for x, key, color in [
        (paired_x[0], "trajectory_equal_mean_surgery_effect", COLORS["SHORT"]),
        (paired_x[1], "branch_weighted_mean_surgery_effect", "#9C755F"),
    ]:
        row = base[key]
        estimate, low, high = f(row["estimate"]), f(row["ci_low"]), f(row["ci_high"])
        err(axes[0], x, estimate, low, high, fmt="o", ms=4.2, color=color)
        axes[0].text(x, estimate + .015, f"{estimate:+.3f}", ha="center", fontsize=5.8)
    axes[0].axhline(0, color="#333333", lw=.7)
    axes[0].set_xlim(-.38, .38)
    axes[0].set_ylim(.065, .19)
    axes[0].set_xticks(paired_x, ["Trajectory\n(equal)", "Branch\n(weighted)"])
    axes[0].tick_params(axis="x", labelsize=5.3)
    axes[0].set_ylabel("Final-score effect\n(Surgery − SHARDED)")
    panel_title(axes[0], "A", "Average effect")

    # B: frozen trajectory-level prevalence at compact paired positions.
    for x, key in [
        (paired_x[0], "trajectory_any_positive_rate"),
        (paired_x[1], "binary_trajectory_any_fail_to_success_rate"),
    ]:
        row = base[key]
        estimate, low, high = f(row["estimate"]), f(row["ci_low"]), f(row["ci_high"])
        err(axes[1], x, estimate, low, high, fmt="o", ms=4.2, color=COLORS["POSITIVE"])
        axes[1].text(x, estimate + .07, f"{100 * estimate:.1f}%", ha="center", fontsize=5.8)
    axes[1].set_xlim(-.38, .38)
    axes[1].set_ylim(.36, .74)
    axes[1].set_xticks(paired_x, ["Any positive\n(151/237)", "Any 0→1\n(76/157)"])
    axes[1].tick_params(axis="x", labelsize=5.15)
    axes[1].set_ylabel("Trajectory proportion")
    panel_title(axes[1], "B", "Trajectory reach")

    # C: unchanged frozen relative-position profile.
    position_rows = [r for r in data if r["panel"] == "C"]
    position = np.array([f(r["mean_relative_position"]) for r in position_rows])
    positive = np.array([f(r["positive_rate"]) for r in position_rows])
    negative = np.array([f(r["negative_rate"]) for r in position_rows])
    axes[2].plot(position, positive, "o-", color=COLORS["POSITIVE"],
                 ms=3.0, lw=1.0, label="Positive")
    axes[2].plot(position, negative, "s--", color=COLORS["NEGATIVE"],
                 ms=2.8, lw=.9, label="Negative")
    axes[2].set_xlim(-.03, 1.0)
    axes[2].set_ylim(0, .44)
    axes[2].set_xlabel("Normalized assistant-turn position")
    axes[2].set_ylabel("Branch fraction")
    axes[2].legend(frameon=False, fontsize=5.4, ncol=2, loc="upper right")
    axes[2].text(.99, .04, "1,440 valid branches", transform=axes[2].transAxes,
                 ha="right", fontsize=5.2, color="#555555")
    axes[2].grid(alpha=.17)
    panel_title(axes[2], "C", "Relative-position profile")

    # D: frozen distance-bin summaries from Phase L1; no aggregation here.
    bin_labels = ["1", "2", "3", "4", "5", "6–8", "9–12", "13+"]
    distance_x = np.arange(len(distance_rows))
    mean_delta = np.array([f(r["mean_delta"]) for r in distance_rows])
    axes[3].plot(distance_x, mean_delta, "o-", color=COLORS["STATE"],
                 ms=3.4, lw=1.0)
    distance_metric = {r["metric"]: r for r in data if r["panel"] == "D"}["distance_vs_delta"]
    rho = f(distance_metric["estimate"])
    low, high = f(distance_metric["ci_low"]), f(distance_metric["ci_high"])
    axes[3].axhline(0, color="#333333", lw=.65)
    axes[3].set_xticks(distance_x, bin_labels)
    axes[3].tick_params(axis="x", labelsize=5.2)
    axes[3].set_xlim(-.25, 7.25)
    axes[3].set_ylim(-.012, .205)
    axes[3].set_xlabel("Distance to final (user turns)")
    axes[3].set_ylabel("Mean surgery effect")
    axes[3].text(
        .03, .22, f"Spearman ρ = {rho:+.3f}\n95% CI [{low:+.3f}, {high:+.3f}]",
        transform=axes[3].transAxes, ha="left", va="bottom", fontsize=5.15,
        bbox=dict(boxstyle="round,pad=.16", fc="white", ec="#AAAAAA", lw=.45, alpha=.94),
    )
    axes[3].text(.03, .10, "57/237 positive ≥6 user turns before final",
                 transform=axes[3].transAxes, ha="left", va="bottom",
                 fontsize=4.9, color="#444444")
    axes[3].grid(alpha=.12)
    panel_title(axes[3], "D", "Distance to final")
    save(fig, "Figure4_turn_surgery_structure")


def figure5() -> None:
    data = read_csv(DATA / "figure5_plot_data.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.08), layout="constrained",
        gridspec_kw={"width_ratios": [1.02, 1.42, 1.06]},
    )
    fig.get_layout_engine().set(w_pad=.012, h_pad=.002, wspace=.038, hspace=.002)

    # A: unchanged frozen controlled positive-minus-zero state contrasts.
    rows_a = [r for r in data if r["panel"] == "A"]
    order = [
        "State final anchor", "State AUC",
        "State Propagation within-trajectory final anchor",
        "State Propagation within-trajectory AUC",
    ]
    labels = ["Task-macro\nfinal", "Task-macro\nAUC",
              "Within traj.\nfinal", "Within traj.\nAUC"]
    lookup_a = {r["probe"]: r for r in rows_a}
    paired_y = [0.0, .58, 1.72, 2.30]
    for y, i, key in zip(paired_y, range(len(order)), order):
        row = lookup_a[key]
        estimate = 1000 * f(row["estimate"])
        low, high = 1000 * f(row["ci_low"]), 1000 * f(row["ci_high"])
        color = COLORS["FORMAL"] if i < 2 else COLORS["POSITIVE"]
        err(axes[0], y, estimate, low, high, horizontal=True,
            fmt="o", ms=3.8, color=color)
    axes[0].axvline(0, color="#333333", lw=.7)
    axes[0].set_yticks(paired_y, labels)
    axes[0].set_ylim(2.48, -.18)
    axes[0].set_xlabel("Positive − zero divergence (×10³)")
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=COLORS["FORMAL"],
                   markersize=3.8, label="Task-macro (118 trajectories)"),
            Line2D([], [], marker="o", linestyle="none", color=COLORS["POSITIVE"],
                   markersize=3.8, label="Within-trajectory (46 traj. / 193 pairs)"),
        ],
        frameon=False, fontsize=4.75, loc="center right", handletextpad=.3,
        borderaxespad=.1,
    )
    axes[0].grid(axis="x", alpha=.17)
    axes[0].set_title("A  Controlled state divergence", loc="left", fontsize=7.35,
                      fontweight="semibold", pad=3)

    # B: unchanged full-layer reversion profiles, with tighter vertical bounds.
    rows_b = sorted([r for r in data if r["panel"] == "B"],
                    key=lambda r: int(r["layer_or_region"]))
    layer = np.array([int(r["layer_or_region"]) for r in rows_b])
    vertical_values: list[float] = []
    for prefix, label, color in [
        ("formal_reversion", "Held-out (33 pairs)", COLORS["FORMAL"]),
        ("prospective_reversion", "Prospective (43 pairs)", COLORS["PROSPECTIVE"]),
    ]:
        mean = 1000 * np.array([f(r[f"{prefix}_difference"]) for r in rows_b])
        low = 1000 * np.array([f(r[f"{prefix}_ci_low"]) for r in rows_b])
        high = 1000 * np.array([f(r[f"{prefix}_ci_high"]) for r in rows_b])
        vertical_values.extend(low.tolist())
        vertical_values.extend(high.tolist())
        axes[1].plot(layer, mean, color=color, lw=1.05, label=label)
        axes[1].fill_between(layer, low, high, color=color, alpha=.12, linewidth=0)
    y_min, y_max = min(vertical_values), max(vertical_values)
    y_pad = .035 * (y_max - y_min)
    axes[1].set_ylim(y_min - y_pad, y_max + y_pad)
    axes[1].axhline(0, color="#333333", lw=.65)
    axes[1].axvspan(3, 12, color="#888888", alpha=.07)
    axes[1].axvspan(16, 31, color="#888888", alpha=.07)
    axes[1].set_xlim(0, 39)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("D2T reversion Δ score (×10³)")
    axes[1].legend(frameon=False, fontsize=5.3, loc="upper right",
                   borderaxespad=.15)
    axes[1].grid(alpha=.13)
    axes[1].set_title("B  State restoration", loc="left", fontsize=7.35,
                      fontweight="semibold", pad=3)

    # C: same frozen signs/CIs, rendered as an explicit three-state matrix.
    rows_c = [r for r in data if r["panel"] == "C"]
    probes = ["S1", "S1b aggregate", "State AUC", "S3 direct", "S3 full"]
    probe_labels = ["Static target\nattention", "History attention\nshift",
                    "State propagation\nAUC", "Direct-only", "Full-\ndownstream"]
    tasks = ["actions", "code", "data2text"]
    task_labels = ["Actions", "Code", "D2T"]
    lookup_c = {(r["probe"], r["task"]): r for r in rows_c}
    category_style = {
        "positive": ("#5D89AD", "+", "white"),
        "negative": ("#BE655F", "−", "white"),
        "uncertain": ("#ECEBE9", "○", "#666666"),
    }
    axes[2].set_xlim(-.5, 2.5)
    axes[2].set_ylim(4.5, -.5)
    for i, probe in enumerate(probes):
        for j, task in enumerate(tasks):
            row = lookup_c[probe, task]
            low, high = f(row["ci_low"]), f(row["ci_high"])
            if low > 0:
                category = "positive"
            elif high < 0:
                category = "negative"
            else:
                category = "uncertain"
            face, symbol, text_color = category_style[category]
            axes[2].add_patch(Rectangle((j - .47, i - .40), .94, .80,
                                        facecolor=face, edgecolor="white", lw=.6))
            axes[2].text(j, i, symbol, ha="center", va="center", fontsize=8.0,
                         fontweight="semibold", color=text_color)
    axes[2].set_xticks(range(3), task_labels)
    axes[2].set_yticks(range(5), probe_labels)
    axes[2].tick_params(length=0)
    axes[2].legend(
        handles=[
            Line2D([], [], marker="$+$", linestyle="none", markersize=5.8,
                   color=category_style["positive"][0], label="significant positive"),
            Line2D([], [], marker="$-$", linestyle="none", markersize=5.8,
                   color=category_style["negative"][0], label="significant negative"),
            Line2D([], [], marker="o", linestyle="none", markersize=3.7,
                   markerfacecolor="none", markeredgecolor=category_style["uncertain"][2],
                   label="CI includes zero"),
        ],
        frameon=False, fontsize=4.55, loc="lower center", ncol=3,
        bbox_to_anchor=(.5, -.195), handlelength=.7, handletextpad=.42,
        columnspacing=.82, borderaxespad=0,
    )
    axes[2].set_title("C  Held-out task probes", loc="left", fontsize=7.35,
                      fontweight="semibold", pad=3)
    save(fig, "Figure5_internal_signatures")


def validate_inputs() -> None:
    required = [
        DATA / "figure3_plot_data.csv",
        DATA / "figure4_plot_data.csv",
        DATA / "figure5_plot_data.csv",
        BRANCH_SOURCE,
        POSITION_SOURCE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen plotting input(s):\n" + "\n".join(missing))
    if len(read_csv(DATA / "figure3_plot_data.csv")) != 13:
        raise ValueError("Figure 3 frozen plotting table row count changed")
    if len(read_csv(DATA / "figure4_plot_data.csv")) != 17:
        raise ValueError("Figure 4 frozen plotting table row count changed")
    if len(read_csv(DATA / "figure5_plot_data.csv")) != 59:
        raise ValueError("Figure 5 frozen plotting table row count changed")


def main() -> None:
    validate_inputs()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.4, "axes.labelsize": 6.4,
        "xtick.labelsize": 5.7, "ytick.labelsize": 5.7, "axes.linewidth": .65,
        "xtick.major.width": .55, "ytick.major.width": .55, "lines.linewidth": .9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure3()
    figure4()
    figure5()
    print(f"Rendered refined Figures 3--5 to {OUT}")


if __name__ == "__main__":
    main()
