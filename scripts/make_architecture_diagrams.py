"""
Generate architecture / structure diagrams as SVG for the portfolio:

    H_wrapper_stack.svg          gym Env wrapper composition + data flow
    I_ppo_architecture.svg       PPO MLP (separate pi/vf [256,256]) + history-stack expansion
    J_cause_isolation_flow.svg   E1-E4 cause-isolation flowchart (v4c null -> v4d/v5a_v2)

Built with matplotlib patches/arrows — single dependency, easy to re-render.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 10.5,
        "svg.fonttype": "none",
    }
)


def rounded_box(ax, x, y, w, h, text, fc="#e7f5ff", ec="#1c7ed6",
                lw=1.3, fontsize=10, weight="normal", text_color="black",
                style="round,pad=0.08", ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color)


def arrow(ax, p0, p1, color="#495057", lw=1.4, style="-|>", rad=0.0,
          mutation_scale=14):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, lw=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}",
                        mutation_scale=mutation_scale)
    ax.add_patch(a)


# =============================================================================
# H — Wrapper stack diagram
# =============================================================================
def plot_H():
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Five wrapper boxes (outermost top → innermost bottom)
    boxes = [
        # (y, color_fc, color_ec, label, sub)
        (7.0, "#fff3bf", "#f08c00",
         "WellTrainedLocomotionAntWrapper",
         "reward shaping (track_vx/vy, heading, orientation, height,\n"
         "action_rate, recovery-window), obs += compact 4 + prev_action 8"),
        (5.6, "#e9fac8", "#74b816",
         "ObservationHistoryStackWrapper   (Phase 2d only)",
         "stack_size=5, newest-first layout  [obs_t, obs_{t-1}, ..., obs_{t-4}]"),
        (4.2, "#d0bfff", "#6741d6",
         "PushDisturbanceWrapper",
         "xfrc_applied on torso; force ramp 0 → 10/15/20 N\n"
         "duration 5–30 step, optional ±0.2 N·m yaw torque, boundary sampling"),
        (2.8, "#ffd8a8", "#d9480f",
         "DomainRandomizationWrapper",
         "reset: mass×U(0.8,1.2), friction×U(0.5,1.5),\n"
         "damping×U(0.8,1.2), motor×U(0.85,1.15); step: action noise σ=0.02"),
        (1.4, "#dee2e6", "#495057",
         "gym.make(\"Ant-v5\")",
         "MuJoCo Ant-v5, dt=0.01 s, 8-DoF torque, obs 105-D, action 8-D"),
    ]
    for (y, fc, ec, label, sub) in boxes:
        rounded_box(ax, 1.0, y, 7.3, 1.1, "", fc=fc, ec=ec, lw=1.6)
        ax.text(1.2, y + 0.78, label, fontsize=11.5, fontweight="bold")
        ax.text(1.2, y + 0.25, sub, fontsize=9, color="#343a40")

    # Dashed border to indicate optional history stack
    rounded_box(ax, 1.0, 5.6, 7.3, 1.1, "", fc="none", ec="#74b816",
                lw=1.8, ls="--")

    # Right-side data flow legend
    ax.text(9.6, 8.0, "data flow", fontsize=11, fontweight="bold", ha="center")

    # action flow (top → down)
    arrow(ax, (9.0, 7.8), (9.0, 1.4), color="#1f6feb", lw=2.2,
          mutation_scale=16)
    ax.text(9.55, 4.5, "action\n(policy → sim)", fontsize=9.5,
            color="#1f6feb", ha="left", va="center", rotation=0)

    # obs flow (bottom → up)
    arrow(ax, (10.4, 1.6), (10.4, 8.0), color="#2f9e44", lw=2.2,
          mutation_scale=16)
    ax.text(10.1, 4.5, "obs\n(sim → policy)", fontsize=9.5,
            color="#2f9e44", ha="right", va="center")

    # info propagation arrow (push wrapper → outer locomotion wrapper)
    arrow(ax, (8.3, 4.3), (8.3, 7.5), color="#e8590c", lw=1.5,
          rad=0.25, mutation_scale=13, style="->")
    ax.text(8.7, 5.95, 'info["push_active"]\ninfo["push_steps_since_end"]\n→ gates recovery reward',
            fontsize=8.5, color="#e8590c", ha="left", va="center")

    ax.set_title("Wrapper composition stack — outermost (policy-facing) → innermost (sim)",
                 fontsize=12.5, pad=10)

    # Footer note on obs dim
    ax.text(0.5, 0.4,
            "obs dim: 105 (raw) → 117 (after WellTrained) → 585 (after history stack 5)   |   "
            "action dim: 8 (torque)",
            fontsize=9, color="#495057", style="italic")

    out = PLOTS_DIR / "H_wrapper_stack.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[H] wrote {out}")


# =============================================================================
# I — PPO model architecture
# =============================================================================
def plot_I():
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input box
    rounded_box(ax, 0.3, 3.0, 1.6, 1.6,
                "Observation\n\n(117 or 585)", fc="#dee2e6", ec="#495057", lw=1.5,
                weight="bold", fontsize=10.5)

    # History-stack expansion banner (top)
    rounded_box(ax, 0.3, 5.4, 4.2, 0.8,
                "Phase 2d input expansion:  obs 117 × 5 frames = 585\n"
                "newest-first  [obs_t, obs_{t-1}, …, obs_{t-4}]",
                fc="#e9fac8", ec="#74b816", lw=1.2, fontsize=9)
    arrow(ax, (2.4, 5.35), (1.5, 4.65), color="#74b816", lw=1.3, rad=-0.15)

    # Policy (π) branch — top
    pi_y = 4.3
    rounded_box(ax, 3.0, pi_y, 1.6, 0.9, "Linear\n→ 256",
                fc="#d0bfff", ec="#5f3dc4", fontsize=10)
    rounded_box(ax, 5.0, pi_y, 1.4, 0.9, "Tanh", fc="#f3f0ff", ec="#5f3dc4")
    rounded_box(ax, 6.6, pi_y, 1.6, 0.9, "Linear\n→ 256",
                fc="#d0bfff", ec="#5f3dc4", fontsize=10)
    rounded_box(ax, 8.6, pi_y, 1.4, 0.9, "Tanh", fc="#f3f0ff", ec="#5f3dc4")
    rounded_box(ax, 10.2, pi_y, 1.5, 0.9,
                "μ\n(action 8)", fc="#5f3dc4", ec="#5f3dc4",
                text_color="white", fontsize=10, weight="bold")

    # Value (V) branch — bottom
    vf_y = 1.8
    rounded_box(ax, 3.0, vf_y, 1.6, 0.9, "Linear\n→ 256",
                fc="#a5d8ff", ec="#1864ab", fontsize=10)
    rounded_box(ax, 5.0, vf_y, 1.4, 0.9, "Tanh", fc="#e7f5ff", ec="#1864ab")
    rounded_box(ax, 6.6, vf_y, 1.6, 0.9, "Linear\n→ 256",
                fc="#a5d8ff", ec="#1864ab", fontsize=10)
    rounded_box(ax, 8.6, vf_y, 1.4, 0.9, "Tanh", fc="#e7f5ff", ec="#1864ab")
    rounded_box(ax, 10.2, vf_y, 1.5, 0.9,
                "V(s)\n(scalar)", fc="#1864ab", ec="#1864ab",
                text_color="white", fontsize=10, weight="bold")

    # State-independent log_std parameter for Gaussian
    rounded_box(ax, 10.2, 5.6, 1.5, 0.6,
                "log σ  (8)\nstate-indep.",
                fc="#fff3bf", ec="#f08c00", fontsize=8.5)
    ax.text(10.95, 5.3, "Gaussian policy\nN(μ, σ)", fontsize=9,
            color="#f08c00", ha="center")

    # Connect input → first layer (both branches)
    arrow(ax, (1.9, 4.1), (3.0, 4.6), color="#5f3dc4", lw=1.4)
    arrow(ax, (1.9, 3.5), (3.0, 2.2), color="#1864ab", lw=1.4)

    # Within branches
    for y in (pi_y, vf_y):
        arrow(ax, (4.6, y + 0.45), (5.0, y + 0.45), color="#495057", lw=1.0)
        arrow(ax, (6.4, y + 0.45), (6.6, y + 0.45), color="#495057", lw=1.0)
        arrow(ax, (8.2, y + 0.45), (8.6, y + 0.45), color="#495057", lw=1.0)
        arrow(ax, (10.0, y + 0.45), (10.2, y + 0.45), color="#495057", lw=1.0)

    # Branch labels
    ax.text(7, pi_y + 1.2, "Policy network  (pi=[256, 256])",
            fontsize=11, color="#5f3dc4", ha="center", fontweight="bold")
    ax.text(7, vf_y + 1.2, "Value network  (vf=[256, 256])",
            fontsize=11, color="#1864ab", ha="center", fontweight="bold")

    # Footer with PPO hyperparams (from runs/.../config.json)
    ax.text(0.3, 0.6,
            "Algorithm: PPO (SB3 MlpPolicy)   |   net_arch = dict(pi=[256, 256], vf=[256, 256])\n"
            "lr 1e-4 (v4b) / 5e-5 (v5a_v2)   |   n_steps 2048   |   batch 128   |   "
            "n_epochs 10   |   γ=0.99   |   λ_GAE=0.95   |   clip 0.2",
            fontsize=9, color="#495057", style="italic")

    ax.set_title("PPO policy & value architecture  (Stable-Baselines3 MlpPolicy)",
                 fontsize=12.5, pad=10)

    out = PLOTS_DIR / "I_ppo_architecture.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[I] wrote {out}")


# =============================================================================
# J — E1–E4 cause-isolation flowchart
# =============================================================================
def plot_J():
    fig, ax = plt.subplots(figsize=(13.5, 8.5), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top: trigger — v4c null result
    rounded_box(ax, 4.5, 7.8, 5.0, 1.0,
                "v4c null result\n"
                "stress curriculum +1 M steps → survival Δ ≤ 3 pp, "
                "nominal drift 1.18 → 1.37 m",
                fc="#ffe3e3", ec="#c92a2a", lw=1.6, fontsize=10.5, weight="bold")

    # Forbidden shortcut
    rounded_box(ax, 10.0, 7.8, 3.7, 1.0,
                "✗ tempting shortcut:\n  PPO → SAC / RMA\n  (without cause isolation)",
                fc="#f8f9fa", ec="#868e96", lw=1.2, fontsize=9.5,
                style="round,pad=0.08", ls="--")
    arrow(ax, (9.5, 8.3), (10.0, 8.3), color="#c92a2a", lw=1.4, style="-|>")
    ax.text(9.75, 8.55, "blocked by\nmeasurement-first\nmethodology",
            fontsize=8.0, color="#c92a2a", ha="center")

    # Hypothesis row
    hypotheses = [
        ("H1: disturbance physics\nis wrong magnitude",     0.5, "#f1f3f5", "#495057"),
        ("H2: evaluation metric\nmisses real change",       4.0, "#f1f3f5", "#495057"),
        ("H3: reward shaping +\nsampling distribution",     7.5, "#f1f3f5", "#495057"),
        ("H4: observation lacks\ndisturbance info",         11.0, "#f1f3f5", "#495057"),
    ]
    hyp_y = 5.7
    hyp_w = 2.8
    for text, x, fc, ec in hypotheses:
        rounded_box(ax, x, hyp_y, hyp_w, 1.0, text, fc=fc, ec=ec, lw=1.2,
                    fontsize=10)
        arrow(ax, (7.0, 7.75), (x + hyp_w / 2, hyp_y + 1.0),
              color="#868e96", lw=1.0, rad=0.0)

    # Experiments row
    experiments = [
        ("E1\nstress_physics_audit.py\n116 cells, qvel Δv measured",
         0.5, "#e3fafc", "#0c8599"),
        ("E2\nrecovery_metric_grid.py\nsliding-window recovery + integrated err",
         4.0, "#e3fafc", "#0c8599"),
        ("E3\nv4d: boundary sampling\n+ windowed recovery reward",
         7.5, "#fff3bf", "#f08c00"),
        ("E4\nv5a_v2: ObservationHistoryStack(5)\n+ input-expansion warm-start",
         11.0, "#fff3bf", "#f08c00"),
    ]
    exp_y = 3.7
    for text, x, fc, ec in experiments:
        rounded_box(ax, x, exp_y, hyp_w, 1.3, text, fc=fc, ec=ec, lw=1.4,
                    fontsize=9.5, weight="bold")
        arrow(ax, (x + hyp_w / 2, hyp_y), (x + hyp_w / 2, exp_y + 1.3),
              color="#495057", lw=1.2)

    # Outcomes row
    outcomes = [
        ("H1 rejected\n15 N×0.3s: |Δv| 12–14 m/s\n(disturbance is real)",
         0.5, "#ffe3e3", "#c92a2a"),
        ("H2 rejected\nv4c also no recovery gain\non new metrics",
         4.0, "#ffe3e3", "#c92a2a"),
        ("H3 partial validation\n15 N forward 25 % → 75 %\n(lateral trade-off)",
         7.5, "#d3f9d8", "#2f9e44"),
        ("H4 partial validation\n10 N avg 69 % → 88 %\nrecovery 2.4× faster",
         11.0, "#d3f9d8", "#2f9e44"),
    ]
    out_y = 1.6
    for text, x, fc, ec in outcomes:
        rounded_box(ax, x, out_y, hyp_w, 1.3, text, fc=fc, ec=ec, lw=1.4,
                    fontsize=9.5)
        arrow(ax, (x + hyp_w / 2, exp_y), (x + hyp_w / 2, out_y + 1.3),
              color="#495057", lw=1.2)

    # Bottom — next-step candidates (only if E3/E4 ceilings remain)
    rounded_box(ax, 4.0, 0.0, 6.0, 1.0,
                "Next single-variable interventions (priority order):\n"
                "1) longer history stack (10–20 frames)  →  2) RecurrentPPO\n"
                "3) RMA-lite teacher–student  →  4) only THEN consider SAC",
                fc="#e7f5ff", ec="#1864ab", lw=1.5, fontsize=9.5)
    arrow(ax, (9.0, 1.5), (8.5, 1.0), color="#2f9e44", lw=1.3, rad=-0.15)

    ax.set_title("E1–E4 cause-isolation cycle  —  v4c null result decomposition",
                 fontsize=13, pad=10, fontweight="bold")

    out = PLOTS_DIR / "J_cause_isolation_flow.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[J] wrote {out}")


if __name__ == "__main__":
    plot_H()
    plot_I()
    plot_J()
    print("\nAll diagrams written to", PLOTS_DIR)
