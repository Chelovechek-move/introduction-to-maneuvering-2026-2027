"""Slide layouts of the handbook's geometry diagrams, lecture 10.

Run from the repository root:
    python3 10/images/geometry_slide_figures.py

All vectors, angles, curves and labels come from geometry_diagrams.py.
Only the canvas, text sizes and output crops change. Individual panels
omit the common formulas below the handbook's drawings; these formulas
are set in TeX on the corresponding slides. Handbook outputs are untouched.
"""
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True

OUT = Path(__file__).resolve().parent
QA = OUT.parent / "build" / "geometry-slides-qa"
WORK = QA / "source-data"
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(QA / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import geometry_diagrams as geometry

PAPER = "#FBF9F3"
INK = "#302B28"


def output(fig, name, axes=None):
    """Save an entire layout or a vector crop around the selected axes."""
    visibility = [(ax, ax.get_visible()) for ax in fig.axes]
    if axes is not None:
        for ax in fig.axes:
            ax.set_visible(ax in axes)
    fig.canvas.draw()
    if axes is None:
        bbox = "tight"
    else:
        renderer = fig.canvas.get_renderer()
        bbox = Bbox.union([ax.get_tightbbox(renderer) for ax in axes])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(.045)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=bbox, pad_inches=.045,
                facecolor=PAPER,
                metadata={"Title": name, "Author": "Lecture 10"})
    fig.savefig(QA / f"{name}.png", bbox_inches=bbox, pad_inches=.045,
                dpi=180, facecolor=PAPER)
    for ax, visible in visibility:
        ax.set_visible(visible)


def paper_and_type(fig, stem):
    """Keep the manual's palette and make its labels readable on slides."""
    fig.set_facecolor(PAPER)
    for ax in fig.axes:
        ax.set_facecolor(PAPER)
        for patch in ax.patches:
            if to_rgba(patch.get_facecolor()) == to_rgba("white"):
                patch.set_facecolor(PAPER)
    for text in fig.findobj(Text):
        # White labels inside the mass blocks retain their original fit.
        is_white = to_rgba(text.get_color()) == to_rgba("white")
        text.set_fontsize(text.get_fontsize() + (1 if is_white else 2))
        box = text.get_bbox_patch()
        if box is not None:
            box.set_facecolor(PAPER)
    # The equations are supplied as TeX beside or below the slide figures.
    for text in list(fig.texts):
        text.remove()


def save_slide(fig, stem):
    paper_and_type(fig, stem)
    axes = fig.axes
    if stem == "orbits":
        fig.set_size_inches(10.0, 2.95)
        axes[0].set_position([.01, .05, .47, .92])
        axes[1].set_position([.61, .19, .37, .77])
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(.48, .76), frameon=False,
                       fontsize=13, borderpad=0)
        for text in axes[0].texts:
            if text.get_text() == "Земля":
                text.set_position((0, -2900))
        output(fig, "geometry-orbits-slide")
    elif stem == "local-frame":
        fig.set_size_inches(6.25, 5.85)
        axes[0].set_position([.035, .025, .93, .95])
        output(fig, "geometry-local-frame-slide")
    elif stem == "staging":
        fig.set_size_inches(6.5, 5.6)
        axes[0].set_position([.025, .025, .95, .95])
        for text in axes[0].texts:
            text.set_fontsize(text.get_fontsize() + 4)
        output(fig, "geometry-staging-slide")
    elif stem == "gravity-turn":
        fig.set_size_inches(9.65, 2.95)
        axes[0].set_position([.01, .03, .47, .83])
        axes[1].set_position([.57, .03, .42, .83])
        output(fig, "geometry-gravity-turn-slide")
    elif stem == "sequence":
        output(fig, "geometry-sequence-frames-slide", axes[1:])
        fig.set_size_inches(10.0, 2.7)
        axes[0].set_position([.01, .24, .98, .44])
        output(fig, "geometry-sequence-path-slide", axes[:1])
    elif stem == "coast-insertion":
        fig.set_size_inches(9.4, 3.8)
        axes[0].set_position([.01, .015, .57, .96])
        axes[1].set_position([.66, .015, .33, .96])
        for text in axes[0].texts:
            if text.get_text() == "Целевая окружность":
                text.set_position((0, 1.79))
                text.set_horizontalalignment("center")
            elif text.get_text() == "Пассивная дуга":
                text.set_position((.25, -1.79))
                text.set_verticalalignment("top")
        output(fig, "geometry-coast-orbit-slide", axes[:1])
    elif stem == "rotation":
        output(fig, "geometry-rotation-pad-slide", axes[:1])
        output(fig, "geometry-rotation-air-slide", axes[1:])
    elif stem == "launch-plane":
        output(fig, "geometry-launch-plane-earth-slide", axes[:1])
        output(fig, "geometry-launch-plane-local-slide", axes[1:])
    elif stem == "launch-window":
        fig.set_size_inches(10.2, 3.4)
        axes[0].set_position([.01, .01, .48, .90])
        axes[1].set_position([.56, .01, .43, .90])
        for text in axes[0].texts:
            if text.get_text() == r"$P(t)$":
                x, y = text.get_position()
                text.set_position((x, y + .18))
        for text in axes[1].texts:
            if text.get_text().startswith("След целевой плоскости"):
                text.set_position((-.02, -.91))
                text.set_verticalalignment("top")
            elif text.get_text() in ("$Z$", "к читателю"):
                text.set_verticalalignment("top")
        output(fig, "geometry-launch-window-plane-slide", axes[:1])
        output(fig, "geometry-launch-window-top-slide", axes[1:])
    else:
        raise ValueError(stem)
    plt.close(fig)


def main():
    original_out, original_save = geometry.OUT, geometry.save
    original_ink = geometry.C["ink"]
    geometry.OUT = WORK  # orbits() writes its own illustrative JSON here.
    geometry.save = save_slide
    geometry.C["ink"] = INK
    try:
        with plt.rc_context({"figure.facecolor": PAPER,
                             "axes.facecolor": PAPER,
                             "text.color": INK,
                             "axes.labelcolor": INK}):
            for draw in (geometry.orbits, geometry.local_frame,
                         geometry.sequence, geometry.gravity_turn,
                         geometry.staging, geometry.coast_insertion,
                         geometry.rotation, geometry.launch_plane,
                         geometry.launch_window):
                draw()
    finally:
        geometry.OUT, geometry.save = original_out, original_save
        geometry.C["ink"] = original_ink


if __name__ == "__main__":
    main()
