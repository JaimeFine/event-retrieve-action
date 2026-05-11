from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(__file__).with_name("training_results.csv")
OUTPUT_PDF = Path(__file__).with_name("figure_main_results.pdf")


COLORS = {
    "J_perf": "#E24A33",
    "R_phys": "#348ABD",
    "Success": "#2CA02C",
    "Collision": "#FF6F61",
    "Warning": "#6A9CFF",
}


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#333333", length=4, width=0.8)
    ax.grid(True, axis="y", color="#D9DDE3", linewidth=0.9, alpha=0.85)
    ax.grid(False, axis="x")
    ax.set_facecolor("#FCFCFD")
    ax.margins(x=0.015)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = [
        "episodes",
        "perf_losses",
        "phys_losses",
        "success_rates",
        "collision_rates",
        "warning_rates",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=False,
        gridspec_kw={"hspace": 0.7},
    )
    fig.patch.set_facecolor("white")

    # Top panel: losses
    ax = axes[0]
    line_jperf, = ax.plot(
        df["episodes"],
        df["perf_losses"],
        color=COLORS["J_perf"],
        linewidth=1.6,
        label="J_perf",
    )
    line_rphys, = ax.plot(
        df["episodes"],
        df["phys_losses"],
        color=COLORS["R_phys"],
        linewidth=1.6,
        label="R_phys",
    )
    ax.set_title("Training Losses")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    style_axis(ax)

    # Bottom panel: performance
    ax = axes[1]
    line_success, = ax.plot(
        df["episodes"],
        df["success_rates"],
        color=COLORS["Success"],
        linewidth=1.6,
        label="Success",
    )
    line_collision, = ax.plot(
        df["episodes"],
        df["collision_rates"],
        color=COLORS["Collision"],
        linewidth=1.6,
        label="Collision",
    )
    line_warning, = ax.plot(
        df["episodes"],
        df["warning_rates"],
        color=COLORS["Warning"],
        linewidth=1.6,
        label="Warning",
    )
    ax.set_title("Performance")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Rate")
    style_axis(ax)

    loss_legend = fig.legend(
        [line_jperf, line_rphys],
        ["J_perf", "R_phys"],
        loc="lower center",
        bbox_to_anchor=(0.33, 0.02),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D9DDE3",
        framealpha=0.96,
        borderpad=0.7,
        labelspacing=0.5,
        columnspacing=1.4,
        handlelength=2.2,
        title="Loss Metrics",
    )
    perf_legend = fig.legend(
        [line_success, line_collision, line_warning],
        ["Success", "Collision", "Warning"],
        loc="lower center",
        bbox_to_anchor=(0.73, 0.02),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D9DDE3",
        framealpha=0.96,
        borderpad=0.7,
        labelspacing=0.5,
        columnspacing=1.4,
        handlelength=2.2,
        title="Performance Metrics",
    )
    loss_legend.get_frame().set_linewidth(0.9)
    perf_legend.get_frame().set_linewidth(0.9)

    fig.subplots_adjust(top=0.95, bottom=0.22, left=0.11, right=0.97)

    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
