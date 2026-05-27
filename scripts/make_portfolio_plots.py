"""
Generate portfolio-ready SVG plots from the repository's CSV/JSON outputs.

Outputs (plots/):
    A_stress_survival_heatmap.svg
    B_tier_a_drift_evolution.svg
    G_15N_forward_recovery.svg
    L_push_direction_polar.svg

All plots are vector (SVG), colorful publication style, English labels.
No external dependencies beyond numpy / matplotlib.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)

ACCENT = {
    "v3h_s2": "#9b9b9b",
    "v4a": "#5b8ff9",
    "v4b": "#1f6feb",
    "v4c": "#c2c2c2",
    "v4d": "#f59f00",
    "v5a_v2": "#e8590c",
}


def load_v4b_stress() -> dict[tuple[float, float], tuple[int, int]]:
    path = ROOT / "reports/robust_v4b_on_stress_grid_eval/tier_b_push_grid.csv"
    d: dict[tuple[float, float], list[int]] = defaultdict(lambda: [0, 0])
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            k = (float(row["push_magnitude"]), float(row["push_direction_deg"]))
            d[k][1] += 1
            if row["survived_to_max_steps"] == "True":
                d[k][0] += 1
    return {k: tuple(v) for k, v in d.items()}


def load_recovery_grid() -> dict[str, dict[tuple[float, float], tuple[int, int]]]:
    path = ROOT / "reports/recovery_metric_grid_v5a_v2.csv"
    by_policy: dict[str, dict[tuple[float, float], list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            p = row["policy"]
            k = (float(row["push_force_magnitude"]), float(row["direction_deg"]))
            by_policy[p][k][1] += 1
            if row["survived_to_end"] == "True":
                by_policy[p][k][0] += 1
    return {
        p: {k: tuple(v) for k, v in d.items()} for p, d in by_policy.items()
    }


def survival_rate(cell: tuple[int, int]) -> float:
    s, t = cell
    return s / t if t else float("nan")


# -----------------------------------------------------------------------------
# Plot A — stress-grid survival heatmaps (v4b / v4d / v5a_v2)
# -----------------------------------------------------------------------------
def plot_A():
    v4b = load_v4b_stress()
    recovery = load_recovery_grid()
    v4d = recovery["v4d"]
    v5a = recovery["v5a_v2"]

    forces = [5.0, 10.0, 15.0]
    dirs_4 = [0.0, 90.0, 180.0, 270.0]
    dir_labels = ["0°\n(forward)", "90°\n(left)", "180°\n(back)", "270°\n(right)"]

    def grid(d, forces, dirs):
        g = np.zeros((len(forces), len(dirs)))
        for i, f in enumerate(forces):
            for j, dd in enumerate(dirs):
                g[i, j] = survival_rate(d.get((f, dd), (0, 1))) * 100
        return g

    g_v4b = grid(v4b, forces, dirs_4)
    g_v4d = grid(v4d, forces, dirs_4)
    g_v5a = grid(v5a, forces, dirs_4)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    titles = ["v4b — Phase 2\n(Push + DR)", "v4d — Phase 2c\n(+ Boundary + Recovery Reward)",
              "v5a_v2 — Phase 2d\n(+ History Stack 5)"]
    grids = [g_v4b, g_v4d, g_v5a]
    cmap = plt.get_cmap("RdYlGn")

    for ax, g, t in zip(axes, grids, titles):
        im = ax.imshow(g, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(dirs_4)))
        ax.set_xticklabels(dir_labels, fontsize=9)
        ax.set_yticks(range(len(forces)))
        ax.set_yticklabels([f"{int(f)} N" for f in forces])
        ax.set_title(t, fontsize=11)
        for i in range(len(forces)):
            for j in range(len(dirs_4)):
                v = g[i, j]
                color = "white" if v < 35 or v > 75 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    fig.suptitle("Push-recovery survival on stress grid (0.3 s sustained, ±0.2 N·m yaw torque, 4 seeds/cell)",
                 fontsize=12, y=1.02)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("survival rate (%)")
    out = PLOTS_DIR / "A_stress_survival_heatmap.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[A] wrote {out}")


# -----------------------------------------------------------------------------
# Plot B — Tier A (quiet) lateral-drift evolution across phases
# -----------------------------------------------------------------------------
def plot_B():
    # values straight from reports/.../summary.json (abs_lateral_drift mean)
    policies = ["v3h_s2", "v4a", "v4b", "v4d", "v5a_v2"]
    drifts = [6.48, 1.40, 1.18, 3.17, 4.06]
    phases = ["Phase 1\nNominal", "Phase 2\n(Push 10 N)",
              "Phase 2\n(Push 20 N)", "Phase 2c\n(+Recovery Rwd)",
              "Phase 2d\n(+Hist. Stack 5)"]

    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    colors = [ACCENT[p] for p in policies]
    bars = ax.bar(phases, drifts, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    for b, v in zip(bars, drifts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f} m",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Highlight the regularization effect (v3h_s2 -> v4b)
    ax.annotate("",
                xy=(2, 1.4), xytext=(0, 6.6),
                arrowprops=dict(arrowstyle="->", color="#2f9e44", lw=2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(1.0, 5.0, "5.5× regularization\nfrom Push + DR\n(no algorithm change)",
            color="#2f9e44", fontsize=10, fontweight="bold", ha="center")

    # Highlight Phase 2c/2d trade-off
    ax.annotate("",
                xy=(4, 4.1), xytext=(2, 1.4),
                arrowprops=dict(arrowstyle="->", color="#a61e4d", lw=1.5,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(3.3, 3.3, "trade-off:\nrecovery-reward\nwindow bleed",
            color="#a61e4d", fontsize=9, ha="center")

    ax.set_ylabel("absolute lateral drift over 50 s (m)")
    ax.set_ylim(0, 7.5)
    ax.set_title("Tier A (quiet) nominal-locomotion drift across phases",
                 fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    out = PLOTS_DIR / "B_tier_a_drift_evolution.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[B] wrote {out}")


# -----------------------------------------------------------------------------
# Plot G — 15 N forward-direction survival recovery
# -----------------------------------------------------------------------------
def plot_G():
    v4b = load_v4b_stress()
    rec = load_recovery_grid()

    def sr(d, f, dd):
        return survival_rate(d.get((f, dd), (0, 1))) * 100

    # 4-direction view at 15 N
    dirs = [0.0, 90.0, 180.0, 270.0]
    dir_labels = ["0° forward", "90° left", "180° back", "270° right"]
    series = {
        "v4b\n(baseline)":     [sr(v4b,        15.0, d) for d in dirs],
        "v4d\n(+ recovery)":   [sr(rec["v4d"], 15.0, d) for d in dirs],
        "v5a_v2\n(+ history)": [sr(rec["v5a_v2"], 15.0, d) for d in dirs],
    }
    colors_series = [ACCENT["v4b"], ACCENT["v4d"], ACCENT["v5a_v2"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.4]})

    # Left: 0° forward direction recovery story (bar)
    forward_series = [series[k][0] for k in series]
    bars = axL.bar(list(series.keys()), forward_series,
                   color=colors_series, edgecolor="black", linewidth=0.6, width=0.6)
    for b, v in zip(bars, forward_series):
        axL.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    axL.set_ylim(0, 100)
    axL.set_ylabel("survival rate (%)")
    axL.set_title("15 N × 0.3 s forward push:\nrecovery via reward + sampling design",
                  fontsize=11)
    axL.axhline(50, ls=":", color="gray", lw=0.8)
    axL.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate the headline gain
    axL.annotate("+50 pp\n(reward + sampling)",
                 xy=(1, 75), xytext=(0.5, 90),
                 arrowprops=dict(arrowstyle="->", color="#2f9e44", lw=1.5),
                 color="#2f9e44", fontsize=10, fontweight="bold", ha="center")
    axL.annotate("trade-off:\nstack size 5\ntoo short for\n30-step push",
                 xy=(2, 0), xytext=(1.7, 35),
                 arrowprops=dict(arrowstyle="->", color="#a61e4d", lw=1.2),
                 color="#a61e4d", fontsize=9, ha="center")

    # Right: full 4-direction breakdown at 15 N
    x = np.arange(len(dirs))
    w = 0.27
    for i, (label, vals) in enumerate(series.items()):
        offset = (i - 1) * w
        axR.bar(x + offset, vals, width=w, label=label.replace("\n", " "),
                color=colors_series[i], edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + offset, vals):
            axR.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    axR.set_xticks(x)
    axR.set_xticklabels(dir_labels)
    axR.set_ylim(0, 110)
    axR.set_ylabel("survival rate (%)")
    axR.set_title("15 N stress, all 4 push directions",
                  fontsize=11)
    axR.legend(loc="upper right", fontsize=9, frameon=False)
    axR.grid(axis="y", linestyle=":", alpha=0.5)

    out = PLOTS_DIR / "G_15N_forward_recovery.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[G] wrote {out}")


# -----------------------------------------------------------------------------
# Plot L — Push direction polar diagrams
# -----------------------------------------------------------------------------
def plot_L():
    v4b = load_v4b_stress()
    rec = load_recovery_grid()

    fig = plt.figure(figsize=(13, 5.6), constrained_layout=True)

    # Left subplot — v4b across 8 directions, 3 force levels
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    dirs_8 = [0, 45, 90, 135, 180, 225, 270, 315]
    forces_v4b = [5.0, 10.0, 15.0]
    colors_force = ["#2f9e44", "#fab005", "#e03131"]

    for f, c in zip(forces_v4b, colors_force):
        rates = []
        for d in dirs_8:
            rates.append(survival_rate(v4b.get((f, float(d)), (0, 1))) * 100)
        theta = np.deg2rad(dirs_8 + [dirs_8[0]])
        rr = rates + [rates[0]]
        ax1.plot(theta, rr, "o-", color=c, lw=2, markersize=6,
                 label=f"{int(f)} N")
        ax1.fill(theta, rr, color=c, alpha=0.12)
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([25, 50, 75, 100])
    ax1.set_yticklabels(["25", "50", "75", "100%"], fontsize=8)
    ax1.set_xticks(np.deg2rad(dirs_8))
    ax1.set_xticklabels([f"{d}°" for d in dirs_8], fontsize=9)
    ax1.set_title("v4b — survival vs. push direction\n(8 directions, stress grid)",
                  fontsize=11, pad=20)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=9, frameon=False)
    ax1.grid(alpha=0.4)

    # Right subplot — 15 N comparison across 3 policies (4 directions)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    dirs_4 = [0, 90, 180, 270]
    series = {
        "v4b": [survival_rate(v4b.get((15.0, float(d)), (0, 1))) * 100 for d in dirs_4],
        "v4d": [survival_rate(rec["v4d"].get((15.0, float(d)), (0, 1))) * 100 for d in dirs_4],
        "v5a_v2": [survival_rate(rec["v5a_v2"].get((15.0, float(d)), (0, 1))) * 100 for d in dirs_4],
    }
    color_for = {"v4b": ACCENT["v4b"], "v4d": ACCENT["v4d"], "v5a_v2": ACCENT["v5a_v2"]}
    for name, rates in series.items():
        theta = np.deg2rad(dirs_4 + [dirs_4[0]])
        rr = rates + [rates[0]]
        ax2.plot(theta, rr, "o-", color=color_for[name], lw=2, markersize=7,
                 label=name)
        ax2.fill(theta, rr, color=color_for[name], alpha=0.10)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([25, 50, 75, 100])
    ax2.set_yticklabels(["25", "50", "75", "100%"], fontsize=8)
    ax2.set_xticks(np.deg2rad(dirs_4))
    ax2.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=10)
    ax2.set_title("15 N stress — policy comparison\n(4 cardinal directions)",
                  fontsize=11, pad=20)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=9, frameon=False)
    ax2.grid(alpha=0.4)

    fig.suptitle("Direction-resolved push survival",
                 fontsize=13, y=1.04)

    out = PLOTS_DIR / "L_push_direction_polar.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[L] wrote {out}")


if __name__ == "__main__":
    plot_A()
    plot_B()
    plot_G()
    plot_L()
    print("\nAll plots written to", PLOTS_DIR)
