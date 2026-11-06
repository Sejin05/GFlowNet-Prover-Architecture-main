"""Training Visualization Engine for GFlowNet-Prover.

Generates high-resolution publication-quality training graphs (PNG/SVG) and
in-terminal ASCII progress curves for Windows CMD / PowerShell.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_training_graphs(
    history: Dict[str, Any],
    output_path: str | Path = "plots/training_progress.png",
    dpi: int = 200,
    title_suffix: str = "",
) -> Path:
    """Generate a comprehensive 6-panel training progress dashboard figure.

    Visualizes:
    1. Trajectory Balance Loss (Raw & EMA smoothed) with Huber clip markers.
    2. Terminal Reward and Exploit Mode Discovery count.
    3. Spearman Rank Correlation rho(R, P_F) with Gate Certification threshold.
    4. Policy Entropy H(P_F) and exploration diversity.
    5. Gradient Norm ||grad||_2 with clip boundary.
    6. Learning Rate Schedule (Warmup + Cosine Annealing).
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend (no GUI window required)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    steps = history.get("steps", [])
    if not steps:
        steps = list(range(1, len(history.get("loss_tb", [])) + 1))

    loss_tb = history.get("loss_tb", [])
    loss_ema = history.get("loss_ema", [])
    rewards = history.get("rewards", [])
    modes = history.get("modes", [])
    spearman_rho = history.get("spearman_rho", [])
    entropy = history.get("entropy", [])
    grad_norm = history.get("grad_norm", [])
    lr_policy = history.get("lr_policy", [])
    huber_clipped_steps = history.get("huber_clipped_steps", [])
    huber_clipped_vals = history.get("huber_clipped_vals", [])
    phase = history.get("phase", 2)
    phase_name = {
        1: "Phase 1: MSTM Pre-Training",
        2: "Phase 2: Curriculum Trajectory Balance",
        3: "Phase 3: Continual Learning (EWC)",
    }.get(phase, f"Phase {phase}")

    # Set up dark modern theme
    plt.style.use("fast")
    fig = plt.figure(figsize=(16, 11), facecolor="#0f172a")
    fig.suptitle(
        f"GFlowNet-Prover: Model Training & Convergence Dynamics\n[{phase_name}] {title_suffix}".strip(),
        fontsize=16,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.25, top=0.91, bottom=0.07, left=0.08, right=0.95)

    def style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor("#1e293b")
        ax.set_title(title, fontsize=11, fontweight="bold", color="#38bdf8", pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color="#94a3b8")
        ax.set_ylabel(ylabel, fontsize=9, color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # -------------------------------------------------------------
    # 1. Loss Dynamics (TB Loss & EMA with Huber clipping)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Trajectory Balance Loss (Huber-Clipped L_TB)", "Training Step", "Loss")
    if loss_tb:
        ax1.plot(steps, loss_tb, color="#38bdf8", alpha=0.35, linewidth=1.0, label="Raw L_TB")
    if loss_ema:
        ax1.plot(steps, loss_ema, color="#f59e0b", linewidth=2.0, label="Smoothed EMA Loss")
    if huber_clipped_steps and huber_clipped_vals:
        ax1.scatter(
            huber_clipped_steps,
            huber_clipped_vals,
            color="#ef4444",
            s=16,
            zorder=5,
            alpha=0.75,
            label="Huber Clipped (> delta)",
        )
    ax1.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    # -------------------------------------------------------------
    # 2. Reward Progression & Mode Discovery
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Terminal Reward & Vulnerability Mode Discovery", "Training Step", "Terminal Reward R(tau)")
    if rewards:
        ax2.plot(steps, rewards, color="#10b981", linewidth=1.8, label="Mean Reward R(tau)")
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Exploit Modes Found", fontsize=9, color="#a855f7")
    ax2_twin.tick_params(colors="#a855f7", labelsize=8)
    if modes:
        ax2_twin.step(steps, modes, color="#a855f7", linewidth=2.0, linestyle="--", where="post", label="Modes Found")
        ax2_twin.set_ylim(0, max(16, max(modes) + 2))
    ax2.set_facecolor("#1e293b")

    # Combined legend for twin axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    # -------------------------------------------------------------
    # 3. Spearman Rank Correlation (Flow Consistency)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "GFlowNet Policy Quality: Spearman Correlation rho(R, P_F)", "Training Step", "Spearman Rank rho")
    if spearman_rho:
        ax3.plot(steps, spearman_rho, color="#06b6d4", linewidth=2.0, label="Spearman rho")
        ax3.axhline(0.82, color="#22c55e", linestyle=":", linewidth=1.8, label="Gate Target (>= 0.82)")
        ax3.set_ylim(0.0, 1.05)
    ax3.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    # -------------------------------------------------------------
    # 4. Policy Entropy H(P_F) (Exploration Diversity)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Policy Exploration Entropy H(P_F)", "Training Step", "Entropy (nats)")
    if entropy:
        ax4.plot(steps, entropy, color="#ec4899", linewidth=2.0, label="Entropy H(P_F)")
        ax4.axhline(0.35, color="#f43f5e", linestyle=":", linewidth=1.5, label="Min Entropy Floor (0.35)")
    ax4.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    # -------------------------------------------------------------
    # 5. Gradient Norm & Numerical Stability Guard
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, "Gradient Norm ||grad||_2 & Stability", "Training Step", "Norm")
    if grad_norm:
        ax5.plot(steps, grad_norm, color="#6366f1", linewidth=1.5, label="Gradient Norm")
        ax5.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Clip Threshold (1.0)")
    ax5.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    # -------------------------------------------------------------
    # 6. Learning Rate Schedule (Warmup + Cosine Annealing)
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, "Learning Rate Schedule (Warmup + Cosine Anneal)", "Training Step", "Policy LR")
    if lr_policy:
        ax6.plot(steps, lr_policy, color="#eab308", linewidth=2.0, label="lr_policy")
        ax6.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax6.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    plt.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_file


def render_ascii_chart(
    values: List[float],
    title: str = "Metric",
    height: int = 10,
    width: int = 60,
    color_code: str = "\033[96m",
    reset_code: str = "\033[0m",
) -> str:
    """Render a compact 2D ASCII / Unicode line graph in the Windows CMD console."""
    if not values:
        return f"{title}: (no data)"

    # Resample values to fit width
    n = len(values)
    if n > width:
        chunk_size = n / width
        resampled = [
            sum(values[int(i * chunk_size) : int((i + 1) * chunk_size)])
            / max(1, len(values[int(i * chunk_size) : int((i + 1) * chunk_size)]))
            for i in range(width)
        ]
    else:
        resampled = values

    min_val = min(resampled)
    max_val = max(resampled)
    val_range = max_val - min_val if max_val != min_val else 1.0

    lines: List[str] = []
    lines.append(f"{color_code}== {title} [Min: {min_val:.4f} | Max: {max_val:.4f} | Final: {values[-1]:.4f}] =={reset_code}")

    # Build ASCII grid
    for r in range(height, -1, -1):
        threshold = min_val + (r / height) * val_range
        row_chars = []
        for v in resampled:
            norm = (v - min_val) / val_range
            bin_idx = int(round(norm * height))
            if bin_idx == r:
                row_chars.append("*" if bin_idx == height or bin_idx == 0 else "o")
            elif bin_idx > r:
                row_chars.append("|")
            else:
                row_chars.append(" ")

        label = f"{threshold:8.3f} |"
        lines.append(f"{color_code}{label}{reset_code}" + "".join(row_chars))

    axis_bottom = " " * 9 + "+" + "-" * len(resampled)
    lines.append(axis_bottom)
    return "\n".join(lines)


def render_terminal_summary(history: Dict[str, Any]) -> str:
    """Generate in-terminal ASCII visual graphs for Loss and Reward in CMD."""
    loss_ema = history.get("loss_ema", history.get("loss_tb", []))
    rewards = history.get("rewards", [])
    modes = history.get("modes", [])

    output: List[str] = []
    output.append("\n" + "=" * 70)
    output.append("          GFlowNet-Prover Training Progress & Convergence (CMD Visual)")
    output.append("=" * 70)

    if loss_ema:
        output.append(render_ascii_chart(loss_ema, title="Trajectory Balance Loss (EMA)", height=8, width=55, color_code="\033[93m"))
        output.append("")

    if rewards:
        output.append(render_ascii_chart(rewards, title="Terminal Composite Reward R(tau)", height=8, width=55, color_code="\033[92m"))
        output.append("")

    if modes:
        final_mode = modes[-1]
        output.append(f"\033[95m[Mode Discovery Status]\033[0m Discovered {final_mode}/16 Unique Smart Contract Vulnerability Modes ({(final_mode/16)*100:.1f}%)")
    output.append("=" * 70 + "\n")

    return "\n".join(output)
