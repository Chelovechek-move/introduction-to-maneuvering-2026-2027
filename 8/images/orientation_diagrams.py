"""Four vector diagrams of two attitude laws, lecture 8.

Run from the repository root: python3 8/images/orientation_diagrams.py
All geometry is idealized. The circle has a symmetric 120 degree window
around +X. Figures 1–2 contain only the orbital geometry and direction law;
the circular-example figures also show the response sketches. These use the first-order
eccentricity-vector equations on the reference circle, with constant A.
No trajectory integration or nonlinear result is represented here.
Final PDFs and chat previews of figures 1–2 are saved beside this script;
the other PNG previews go to build.
"""

from pathlib import Path
import os

OUT = Path(__file__).resolve().parent
QA = OUT.parent / "build" / "orientation-qa"
QA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(QA / "mplconfig"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, FancyArrowPatch

for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))

C = dict(orbital="#187AA5", inertial="#087F78", velocity="#B57714",
         radial="#79539C", ink="#302B28", faint="#D8DDE1", grey="#7B858B")
plt.rcParams.update({
    "font.family": "PT Serif", "font.size": 14,
    "mathtext.fontset": "custom", "mathtext.rm": "PT Serif",
    "mathtext.it": "PT Serif:italic", "mathtext.bf": "PT Serif:bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": C["ink"], "axes.titlesize": 15,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def basis(phi):
    return np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])


def arrow(ax, p, v, color, lw=2, scale=14, zorder=5, style="-|>"):
    p, v = np.asarray(p), np.asarray(v)
    ax.add_patch(FancyArrowPatch(p, p+v, arrowstyle=style,
                                mutation_scale=scale, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, zorder=zorder))


def axes_off(ax, xlim, ylim):
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def canvas(left_title, right_title):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.025, right=.98, bottom=.13, top=.86, wspace=.20)
    for ax, title in zip(axes, [left_title, right_title]):
        ax.set_title(title, pad=10)
    return fig, axes


def save(fig, name, chat_preview=False):
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": name, "Author": "Lecture 8"})
    fig.savefig(QA / f"{name}.png", dpi=180)
    if chat_preview:
        fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def circle_geometry(ax, law, annotated=True):
    phi = np.linspace(0, 2*np.pi, 601)
    ax.plot(np.cos(phi), np.sin(phi), color=C["faint"], lw=1.6)
    window = np.linspace(-np.pi/3, np.pi/3, 101)
    col = C[law]
    ax.plot(np.cos(window), np.sin(window), color=col, lw=3.3, solid_capstyle="round")
    for ph in np.linspace(-np.pi/3, np.pi/3, 5):
        er, et = basis(ph)
        direction = et if law == "orbital" else np.array([0, 1])
        arrow(ax, er, .37*direction, col, lw=2)
    if annotated:
        for ph in [-np.pi/3, np.pi/3]:
            er, _ = basis(ph)
            ax.plot([0, er[0]], [0, er[1]], color=C["grey"], lw=.85, linestyle="--")
    arrow(ax, (-.08, 0), (1.76, 0), C["grey"], lw=1, scale=10)
    arrow(ax, (0, -.12), (0, 1.40), C["grey"], lw=1, scale=10)
    ax.text(1.72, -.04, "$X$", fontsize=15, va="top")
    ax.text(-.06, 1.30, "$Y$", fontsize=15, ha="right", va="bottom")
    ax.scatter(0, 0, color=C["ink"], s=20)
    ax.text(-.08, -.08, "$O$", ha="right", va="top", fontsize=14)
    if annotated:
        ax.scatter(1, 0, facecolor="white", edgecolor=C["velocity"], s=34, lw=1.5, zorder=8)
        ax.text(1.08, -.18, r"$\varphi=0$", color=C["velocity"], ha="left", fontsize=14)
        ax.text(.67, -.99, r"$-\theta/2$", ha="left", fontsize=14)
        ax.text(.67, .90, r"$+\theta/2$", ha="left", fontsize=14)
        ax.add_patch(Arc((0, 0), .70, .70, theta1=-60, theta2=60,
                         color=C["radial"], lw=1.3))
        ax.text(.42, .04, r"$\theta$", color=C["radial"], fontsize=15)
        ax.text(-.83, -.90, r"$r_0,\ V_0$", fontsize=15, color=C["grey"])
    axes_off(ax, (-1.16, 1.96), (-1.20, 1.47))


def orientation(law):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.subplots_adjust(left=.045, right=.985, bottom=.15, top=.86)
    circle_geometry(ax, law, annotated=False)
    title = ("Трансверсальное направление в ОСК" if law == "orbital"
             else "Постоянное направление в ИСК")
    ax.set_title(title, pad=8)
    formula = (r"$\mathbf{u}(t)=\mathbf{e}_t(t)$" if law == "orbital"
               else r"$\mathbf{u}(t)=\mathbf{e}_Y=\mathrm{const}$")
    fig.text(.5, .045, formula, ha="center", fontsize=16, color=C[law])
    save(fig, f"orientation-{law}", chat_preview=True)


def orientation_comparison():
    """One side-by-side PNG for displaying both cleaned diagrams in chat."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    fig.subplots_adjust(left=.025, right=.985, bottom=.15, top=.86, wspace=.12)
    for ax, law in zip(axes, ["orbital", "inertial"]):
        circle_geometry(ax, law, annotated=False)
        title = ("Трансверсальное направление в ОСК" if law == "orbital"
                 else "Постоянное направление в ИСК")
        ax.set_title(title, pad=8, fontsize=14)
    fig.text(.255, .045, r"$\mathbf{u}(t)=\mathbf{e}_t(t)$",
             ha="center", fontsize=16, color=C["orbital"])
    fig.text(.755, .045, r"$\mathbf{u}(t)=\mathbf{e}_Y=\mathrm{const}$",
             ha="center", fontsize=16, color=C["inertial"])
    fig.savefig(OUT / "orientation-comparison.png", dpi=220)
    plt.close(fig)


def contribution(ax, law):
    ph = np.deg2rad(40.)
    if law == "orbital":
        w = .80*np.array([np.cos(ph), np.sin(ph)])
    else:
        w = .47*np.array([1+np.cos(ph)**2, np.sin(ph)*np.cos(ph)])
    lo, hi = np.array([w[0], -w[1]]), w
    total = np.array([2*w[0], 0.])
    arrow(ax, (-.15, 0), (1.98, 0), C["grey"], lw=.9, scale=10)
    arrow(ax, (0, -.72), (0, 1.44), C["grey"], lw=.9, scale=10)
    ax.text(1.87, -.04, "$X$", va="top", fontsize=15)
    ax.text(-.07, .75, "$Y$", ha="right", fontsize=15)
    for vec, col in [(hi, C["radial"]), (lo, C["inertial"])]:
        arrow(ax, (0, 0), vec, col, lw=2.2)
        ax.plot([vec[0], total[0]], [vec[1], 0], "--", lw=1.1, color=col, alpha=.65)
    arrow(ax, (0, 0), total, C[law], lw=3.0, scale=15, zorder=7)
    ax.scatter(0, 0, s=20, color=C["ink"], zorder=8)
    ax.text(hi[0]+.08, hi[1]+.08, r"$\dot{\mathbf{e}}(+\varphi)$", color=C["radial"],
            fontsize=15, ha="left", va="bottom")
    ax.text(lo[0]+.08, lo[1]-.08, r"$\dot{\mathbf{e}}(-\varphi)$", color=C["inertial"],
            fontsize=15, ha="left", va="top")
    ax.text(total[0]+.07, .10, "Сумма", color=C[law], fontsize=14, ha="left", va="bottom")
    ax.text(.86, -.98, r"$\Delta e_y=0,\qquad\Delta\mathbf{e}\parallel X$", ha="center", fontsize=16)
    axes_off(ax, (-.30, 2.08), (-1.13, 1.03))


def circular_orientation(law):
    title = "Трансверсальное ускорение в ОСК" if law == "orbital" else "Ускорение по инерциальной оси Y"
    fig, (ax, response) = canvas(title, "Вклады в вектор эксцентриситета")
    circle_geometry(ax, law)
    contribution(response, law)
    formula = (r"$\mathbf{a}=A\,\mathbf{e}_t(t)$" if law == "orbital"
               else r"$\mathbf{a}=A\,\mathbf{e}_Y$")
    fig.text(.27, .045, formula, ha="center", fontsize=16, color=C[law])
    save(fig, f"circle-{law}")


if __name__ == "__main__":
    orientation("orbital")
    orientation("inertial")
    orientation_comparison()
    circular_orientation("orbital")
    circular_orientation("inertial")
