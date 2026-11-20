"""Render the data-free four-RQ conceptual framework (Figure 1)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reproduced" / "figures"
COLORS = {
    "FULL": "#3569B7", "CONCAT": "#D98B2B", "SHARDED": "#B33A3A",
    "POSITIVE": "#4E9A55", "USER": "#E5E5E5", "ASSISTANT": "#537FA6",
    "ACK": "#D69A46", "REGEN": "#C6D9E8", "STATE": "#7966A8",
    "PANEL": "#F7F7F7",
}


def message_box(ax, x, y, label, kind, w=.030, h=.090, fontsize=5.3):
    face, edge, hatch, linestyle, text_color = {
        "user": (COLORS["USER"], "#555555", None, "-", "#202020"),
        "assistant": (COLORS["ASSISTANT"], "#264A68", None, "-", "white"),
        "ack": (COLORS["ACK"], "#79521D", "////", "-", "#202020"),
        "regen": (COLORS["REGEN"], COLORS["ASSISTANT"], None, "--", "#202020"),
    }[kind]
    ax.add_patch(FancyBboxPatch(
        (x-w/2, y-h/2), w, h, boxstyle="round,pad=0.003,rounding_size=0.006",
        facecolor=face, edgecolor=edge, linewidth=.7, hatch=hatch, linestyle=linestyle,
    ))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="semibold")


def arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                mutation_scale=5, lw=.65, color="#555555"))


def sequence(ax, x0, x1, y, specs, box_w=.030):
    xs = np.linspace(x0, x1, len(specs))
    for i, (x, spec) in enumerate(zip(xs, specs)):
        message_box(ax, x, y, *spec, w=box_w)
        if i:
            arrow(ax, xs[i-1]+box_w/2+.002, x-box_w/2-.002, y)


def framework_panel(ax, x0, x1, title):
    ax.add_patch(FancyBboxPatch(
        (x0, .07), x1-x0, .87, boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor=COLORS["PANEL"], edgecolor="#B8B8B8", linewidth=.7,
    ))
    ax.text(x0+.012, .90, title, fontsize=7.2, fontweight="semibold", va="top")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.28))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    bounds = [(.01, .245), (.255, .495), (.505, .745), (.755, .99)]
    titles = ["RQ1  Robustness landscape", "RQ2  History intervention",
              "RQ3  Turn Surgery", "RQ4  Internal signatures"]
    for (x0, x1), title in zip(bounds, titles):
        framework_panel(ax, x0, x1, title)

    sequence(ax, .028, .226, .69,
             [("U1", "user"), ("A1", "assistant"), ("U2", "user"),
              ("A2", "assistant"), ("…", "user"), ("A$_T$", "assistant")], box_w=.026)
    ax.text(.127, .57, "self-generated assistant history", ha="center", fontsize=5.6, color="#264A68")
    for i, (label, color) in enumerate([("FULL", COLORS["FULL"]),
                                         ("CONCAT", COLORS["CONCAT"]),
                                         ("SHARDED", COLORS["SHARDED"])]):
        x = .054 + .073*i
        ax.add_patch(FancyBboxPatch((x-.030, .38), .060, .070, boxstyle="round,pad=0.002",
                                    facecolor="white", edgecolor=color, linewidth=.9))
        ax.text(x, .415, label, ha="center", va="center", fontsize=5.0,
                color=color, fontweight="semibold")
    ax.text(.127, .29, "5 models × 6 tasks", ha="center", fontsize=5.8)
    ax.text(.127, .16, "Where is degradation strongest?", ha="center", fontsize=5.7, fontweight="semibold")

    ax.text(.375, .79, "Realized user trajectory fixed", ha="center", fontsize=5.5,
            bbox=dict(boxstyle="round,pad=0.14", fc="#EEEEEE", ec="#999999", lw=.5))
    ax.text(.298, .62, "SHORT", ha="right", va="center", fontsize=5.0)
    sequence(ax, .320, .475, .62,
             [("U1", "user"), ("ACK", "ack"), ("U2", "user"), ("ACK", "ack"), ("A′$_T$", "regen")], box_w=.026)
    ax.text(.298, .44, "MATCHED", ha="right", va="center", fontsize=5.0)
    sequence(ax, .320, .475, .44,
             [("U1", "user"), ("ACK…", "ack"), ("U2", "user"), ("ACK…", "ack"), ("A″$_T$", "regen")], box_w=.026)
    ax.text(.375, .27, "Short vs length-matched", ha="center", fontsize=5.6)
    ax.text(.375, .15, "Does assistant history change\nthe final outcome?", ha="center", va="center",
            fontsize=5.6, fontweight="semibold", linespacing=1.0)

    ax.text(.625, .79, "Earlier history retained", ha="center", fontsize=5.5)
    sequence(ax, .520, .730, .61,
             [("U1", "user"), ("A1", "assistant"), ("U2", "user"), ("ACK", "ack"),
              ("U3", "user"), ("A′", "regen"), ("A′$_T$", "regen")], box_w=.024)
    ax.text(.625, .47, "one assistant turn replaced", ha="center", fontsize=5.5, color="#79521D")
    ax.text(.625, .36, "downstream users fixed", ha="center", fontsize=5.5)
    ax.text(.625, .27, "assistant path regenerated", ha="center", fontsize=5.5, color="#264A68")
    ax.text(.625, .15, "Which turns change\nthe final outcome?", ha="center", va="center",
            fontsize=5.6, fontweight="semibold", linespacing=1.0)

    ax.text(.872, .79, "Open-model controlled comparison", ha="center", fontsize=5.4)
    x = np.linspace(.785, .955, 7)
    ax.plot(x, .64 + np.array([0, .005, .025, .045, .062, .072, .080]), color=COLORS["POSITIVE"], lw=1.3)
    ax.plot(x, .64 + np.array([0, .002, .010, .014, .020, .022, .024]), color="#777777", lw=1.1, ls="--")
    ax.text(.775, .72, "positive", fontsize=5.0, color=COLORS["POSITIVE"], ha="left")
    ax.text(.775, .63, "zero", fontsize=5.0, color="#666666", ha="left")
    for i, label in enumerate(["State", "Attention", "Intervention"]):
        xx = .784 + i*.067
        ax.add_patch(FancyBboxPatch((xx, .39), .060, .075, boxstyle="round,pad=0.002",
                                    facecolor="white", edgecolor=COLORS["STATE"], linewidth=.75))
        ax.text(xx+.030, .427, label, ha="center", va="center", fontsize=4.6, color="#4C3E70")
    ax.text(.872, .29, "task-structured internal probes", ha="center", fontsize=5.5)
    ax.text(.872, .15, "What internal signatures accompany\ninfluential history turns?", ha="center", va="center",
            fontsize=5.5, fontweight="semibold", linespacing=1.0)

    for i, (label, kind) in enumerate([("User", "user"), ("Original assistant", "assistant"),
                                        ("Intervened ACK", "ack"), ("Regenerated assistant", "regen")]):
        xx = .285 + i*.155
        message_box(ax, xx, .045, "", kind, w=.021, h=.030)
        ax.text(xx+.015, .045, label, fontsize=4.9, va="center", ha="left")
    fig.subplots_adjust(left=.004, right=.996, top=.99, bottom=.04)
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(OUT / f"Figure1_four_rq_framework.{suffix}", facecolor="white",
                    bbox_inches="tight", pad_inches=.025, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
