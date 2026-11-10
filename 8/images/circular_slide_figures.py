"""Slide versions of the circular-example figures, lecture 8.

Run from the repository root:
    python3 8/images/circular_slide_figures.py

The orientation diagrams reuse orientation_diagrams.py without changing
geometry. The response curves reuse the saved circular-data.npz scan and
first-order formulas from circular_burn.py. No new numerical integration.
"""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True

OUT = Path(__file__).resolve().parent
QA = OUT.parent / "build" / "circular-slides-qa"
QA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(QA / "mplconfig"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import orientation_diagrams as diagram
import circular_burn as circular

PAPER = "#FBF9F3"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight",
                metadata={"Title": name, "Author": "Lecture 8"})
    fig.savefig(QA / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def orientation_backgrounds():
    """Keep the manual's single diagrams intact; change only their canvas."""
    def save_orientation(fig, name, chat_preview=False):
        fig.set_facecolor(PAPER)
        for ax in fig.axes:
            ax.set_facecolor(PAPER)
        # No tight cropping: retain the source figure's exact 4.8 x 3.6 in size.
        fig.savefig(OUT / f"{name}-slide.pdf", facecolor=PAPER,
                    metadata={"Title": f"{name}-slide", "Author": "Lecture 8"})
        fig.savefig(QA / f"{name}-slide.png", dpi=180, facecolor=PAPER)
        plt.close(fig)

    original_save = diagram.save
    diagram.save = save_orientation
    try:
        # orientation_diagrams.py originally uses these two font defaults.
        # Other slide figures have larger defaults, which must not leak here.
        with plt.rc_context({"font.size": 14, "axes.titlesize": 15,
                             "figure.facecolor": PAPER, "axes.facecolor": PAPER}):
            for law in ("orbital", "inertial"):
                diagram.orientation(law)
    finally:
        diagram.save = original_save


def orientations():
    for law, title in [("orbital", "Трансверсальное ускорение в ОСК"),
                       ("inertial", "Ускорение по инерциальной оси Y")]:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2),
                                 gridspec_kw={"width_ratios": [1.12, 1]})
        fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.88, wspace=.12)
        diagram.circle_geometry(axes[0], law)
        diagram.contribution(axes[1], law)
        for ax in axes:
            for artist in ax.texts:
                artist.set_fontsize(artist.get_fontsize() + 2)
        axes[0].set_title(title, fontsize=17, pad=9)
        axes[1].set_title("Вклады в вектор эксцентриситета", fontsize=17, pad=9)
        save(fig, f"circle-{law}-slide")


def responses():
    data = np.load(OUT / "circular-data.npz")
    theta = data["theta_rad"]
    full = np.linspace(0, np.pi, 301)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.3), constrained_layout=True)
    for ax, j, title, ylabel in [(axes[0], 0, "Большая полуось", r"$k_a$"),
                                 (axes[1], 1, "Эксцентриситет", r"$k_e$")]:
        ref = 2 * circular.R0 * circular.DV / circular.V0 if j == 0 else 2 * circular.DV / circular.V0
        ax.axhline(1, color=circular.COL["impulse"], lw=1.2,
                   ls=(0, (4, 3)), label="Импульс: первый порядок")
        for law, label in [("rtn", "В текущей ОСК"),
                            ("inertial", "Инерциальное направление")]:
            model = np.array([circular.linear_elements(th, law)[j] for th in full]) / ref
            ax.plot(np.rad2deg(full), model, color=circular.COL[law], lw=2.3, label=label)
            ids = np.arange(2, len(theta), 8)
            ax.scatter(np.rad2deg(theta[ids]), data[f"exact_{law}"][ids, j] / ref,
                       s=31, facecolor=PAPER, edgecolor=circular.COL[law], linewidth=1, zorder=5)
        ax.set(title=title, ylabel=ylabel, ylim=(.59, 1.035))
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 60, 120, 180])
        ax.set_xlabel(r"Протяжённость $\theta$, град")
        ax.grid(alpha=.18)
    axes[0].legend(loc="lower left", frameon=False, fontsize=14)
    axes[1].text(.035, .28, "Линии — линейно\nКружки — нелинейно",
                 transform=axes[1].transAxes, va="top", fontsize=15)
    axes[1].annotate(r"$2/\pi$", xy=(180, 2 / np.pi), xytext=(125, .68),
                     color=circular.COL["rtn"], fontsize=18,
                     arrowprops=dict(arrowstyle="-", color=circular.COL["rtn"], lw=.8))
    axes[1].annotate(r"$0{,}75$", xy=(180, .75), xytext=(135, .80),
                     color=circular.COL["inertial"], fontsize=18,
                     arrowprops=dict(arrowstyle="-", color=circular.COL["inertial"], lw=.8))
    save(fig, "circular-response-slide")


if __name__ == "__main__":
    orientation_backgrounds()
    orientations()
    responses()
