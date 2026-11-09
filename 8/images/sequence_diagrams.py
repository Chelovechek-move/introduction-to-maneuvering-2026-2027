#!/usr/bin/env python3
"""Two schematic figures for chapters 4-6 of the lecture 8 manual.

Reproduce from repository root: python3 8/images/sequence_diagrams.py
PDF figures are written beside this script; PNG previews go to 8/build.
There is no orbital propagation in this script. The first figure is exact
one-dimensional velocity addition (100+20, 99+21, 121-1 m/s). The second
uses illustrative ellipse rotation and a secular-plus-periodic curve solely
to explain geometry transport and osculating versus mean elements.
All illustrative parameters are explicit below; neither curve represents
a numerical J2 transfer or a fitted mean-element model.
"""
from pathlib import Path
import os

OUT = Path(__file__).resolve().parent
BUILD = OUT.parent / "build"
BUILD.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "mplconfig"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle

for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))

C = {"blue": "#187AA5", "green": "#087F78", "orange": "#B57714",
     "purple": "#79539C", "ink": "#25323F", "gray": "#89939B"}
plt.rcParams.update({
    "font.family": "PT Serif", "font.size": 13,
    "mathtext.fontset": "custom", "mathtext.rm": "PT Serif",
    "mathtext.it": "PT Serif:italic", "mathtext.bf": "PT Serif:bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.10)
    fig.savefig(BUILD / f"{name}.png", dpi=170,
                bbox_inches="tight", pad_inches=.10)
    plt.close(fig)


def arrow(ax, x0, x1, y, color, label, label_dy=.17, lw=2.8):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=lw,
                                mutation_scale=15, color=color,
                                shrinkA=0, shrinkB=0))
    ax.text((x0+x1)/2, y+label_dy, label, ha="center", va="bottom",
            color=color)


def compensation():
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.1, 3.7),
        gridspec_kw={"width_ratios": [1.55, 1]})
    left.set_title("Последующий манёвр можно пересчитать", fontsize=14, pad=18)
    left.axvline(120, color=C["gray"], ls="--", lw=1.2, ymin=.07, ymax=.88)
    for y, first, second, color in [(1.35, 100, 20, C["blue"]),
                                    (.25, 99, 21, C["green"])]:
        arrow(left, 0, first, y, color, f"{first} м/с")
        arrow(left, first, first+second, y, color, f"{second} м/с")
        left.plot([0, first, 120], [y]*3, "o", color=color, ms=4)
    left.text(0, 1.98, "План: 100 + 20 = 120", color=C["blue"], fontsize=13)
    left.text(0, .87, "После ошибки: 99 + 21 = 120", color=C["green"], fontsize=13)
    left.text(120, -.28, "Цель\n120 м/с", ha="center", va="top")
    left.text(0, -.28, "0", ha="center", va="top")
    left.set_xlim(-3, 135)
    left.set_ylim(-.65, 2.2)
    left.axis("off")

    right.set_title("После последнего манёвра", fontsize=14, pad=18)
    right.text(117.45, 1.96, "Сообщено 121 м/с вместо 120", fontsize=12.5)
    right.axvline(120, color=C["gray"], ls="--", lw=1.2, ymin=.15, ymax=.79)
    arrow(right, 117.4, 121, 1.20, C["orange"], "До 121 м/с")
    arrow(right, 121, 120, .35, C["purple"], "−1 м/с", label_dy=.17)
    right.plot(121, 1.20, "o", color=C["orange"], ms=5)
    right.plot(120, .35, "o", color=C["purple"], ms=5)
    right.text(120, -.22, "Цель\n120 м/с", ha="center", va="top")
    right.text(117.45, -.98, "Сумма модулей: 121 + 1 = 122 м/с", fontsize=12)
    right.set_xlim(117.25, 122.4)
    right.set_ylim(-1.1, 2.2)
    right.axis("off")
    fig.subplots_adjust(left=.035, right=.98, top=.80, bottom=.10, wspace=.24)
    save(fig, "error-compensation")


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def geometry():
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.1, 4.1),
        gridspec_kw={"width_ratios": [1.05, 1]})
    # Illustrative confocal ellipses; no physical length/time is assigned.
    e, p = .42, 1.35
    turn = np.deg2rad(48.)
    phi = np.linspace(0, 2*np.pi, 601)
    radius = p/(1+e*np.cos(phi))
    xy = np.vstack([radius*np.cos(phi), radius*np.sin(phi)])
    xy2 = rotation(turn) @ xy
    left.plot(*xy, color=C["gray"], lw=1.25, ls="--")
    left.plot(*xy2, color=C["blue"], lw=1.7)
    left.add_patch(Circle((0, 0), .16, color="#A8C7D7", zorder=4))
    peri = p/(1+e)
    old = np.array([peri, 0.])
    new = rotation(turn) @ old
    left.plot([0, old[0]], [0, old[1]], color=C["gray"], lw=.9, ls=":")
    left.plot([0, new[0]], [0, new[1]], color=C["blue"], lw=.9, ls=":")
    left.plot(*old, "o", color=C["gray"], ms=5)
    left.plot(*new, "o", color=C["blue"], ms=5)
    left.annotate("", xy=old+np.array([0, .75]), xytext=old,
                  arrowprops=dict(arrowstyle="-|>", color=C["gray"], lw=1.8))
    left.annotate("", xy=new+np.array([0, .93]), xytext=new,
                  arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2.3))
    mapped = rotation(turn) @ np.array([0., .93])
    left.annotate("", xy=new+mapped, xytext=new,
                  arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.3))
    left.text(old[0]+.12, -.13, "Исходное\nположение", fontsize=11.5,
              color=C["gray"], va="top")
    left.text(.90, 1.26, "Сохраняем\nвектор ИСК", color=C["orange"],
              fontsize=12, ha="left")
    left.text(-.50, 1.55, "Перенос\nчерез ОСК", color=C["green"],
              fontsize=12, ha="right",
              bbox=dict(facecolor="white", edgecolor="none", pad=.7))
    left.text(-2.1, -.23, "Линия апсид\nповернулась", fontsize=11.5,
              color=C["blue"])
    left.set_xlim(-2.4, 1.85)
    left.set_ylim(-2.2, 2.0)
    left.set_aspect("equal")
    left.axis("off")
    left.set_title("Изменилась геометрия орбиты", fontsize=14, pad=12)

    # Symbolic slow evolution + periodic perturbation, explicitly schematic.
    t = np.linspace(0, 5., 701)
    mean = 1.05-.13*t
    osc = mean+.095*np.sin(2*np.pi*t/.85)
    right.plot(t, osc, color=C["blue"], lw=1.5, label="Оскулирующий элемент")
    right.plot(t, mean, color=C["orange"], lw=2.2, label="Средний элемент")
    right.set_xlabel("Время, условные единицы", fontsize=12)
    right.set_ylabel("Значение элемента", fontsize=12)
    right.set_xticks([0, 1, 2, 3, 4, 5])
    right.set_yticks([])
    right.set_xlim(0, 5)
    right.grid(axis="x", alpha=.15)
    right.set_title("Быстрые колебания и средняя эволюция", fontsize=14, pad=12)
    right.legend(loc="upper right", fontsize=11.2, frameon=False)
    right.text(.98, .03, "Схема, не численный прогноз", transform=right.transAxes,
               ha="right", color=C["gray"], fontsize=11)
    fig.subplots_adjust(left=.025, right=.985, top=.82, bottom=.17, wspace=.32)
    save(fig, "perturbed-geometry")


if __name__ == "__main__":
    compensation()
    geometry()
    print("Created error-compensation.pdf and perturbed-geometry.pdf (schematic).")
