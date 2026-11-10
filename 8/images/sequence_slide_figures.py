#!/usr/bin/env python3
"""Slide variants of the manual's sequence and geometry figures.

Run from the repository root: python3 8/images/sequence_slide_figures.py
The source figure and its geometry are kept in sequence_diagrams.py.
Only the paper colour and one geometry-label position are changed here.
The two panels and the PDF page size remain unchanged, so the existing
Beamer trim values also apply to perturbed-geometry-slide.pdf.
"""
from pathlib import Path
import importlib.util
import sys

sys.dont_write_bytecode = True

OUT = Path(__file__).resolve().parent
BUILD = OUT.parent / "build"
PAPER = "#FBF9F3"

spec = importlib.util.spec_from_file_location(
    "sequence_diagrams", OUT / "sequence_diagrams.py")
diagrams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagrams)


def save_slide(fig, source_name):
    """Keep the manual's plot coordinates, curves, dimensions and panels."""
    fig.patch.set_facecolor(PAPER)
    for ax in fig.axes:
        ax.set_facecolor(PAPER)
        for text in ax.texts:
            if text.get_bbox_patch() is not None:
                text.get_bbox_patch().set_facecolor(PAPER)

    if source_name == "perturbed-geometry":
        label = next(text for text in fig.axes[0].texts
                     if text.get_text() == "Линия апсид\nповернулась")
        # Put the caption inside the free lower-left part of the ellipse.
        # It no longer crosses the blue curve along its left edge.
        label.set_position((-1.60, -.85))

    name = f"{source_name}-slide"
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.10)
    fig.savefig(BUILD / f"{name}.png", dpi=170,
                bbox_inches="tight", pad_inches=.10)
    diagrams.plt.close(fig)


if __name__ == "__main__":
    diagrams.save = save_slide
    diagrams.compensation()
    diagrams.geometry()
    print("Created sequence slide figures; manual figures unchanged.")
