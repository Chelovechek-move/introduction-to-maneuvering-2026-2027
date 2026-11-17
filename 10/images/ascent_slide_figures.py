#!/usr/bin/env python3
"""Slide versions of the saved lecture-10 numerical example.

Run from the repository root: python3 10/images/ascent_slide_figures.py
Only ascent-results.json and the three existing NPZ trajectories are read.
The manual's curves are rearranged for 16:9 slides; no integration is run.
Vector PDFs stay in images, raster previews in ../build/ascent-slides-qa.
"""
from pathlib import Path
import json
import os

OUT = Path(__file__).resolve().parent
QA = OUT.parent / "build" / "ascent-slides-qa"
QA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(QA / "mplconfig"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))

PAPER = "#FBF9F3"
INK = "#302B28"
BLUE, TEAL, ORANGE, PURPLE, RED = [
    "#187AA5", "#087F78", "#B57714", "#79539C", "#C44847"]
GREY = "#8A9298"
plt.rcParams.update({
    "font.family": "PT Serif", "mathtext.fontset": "stix",
    "mathtext.rm": "PT Serif", "mathtext.it": "PT Serif:italic",
    "mathtext.bf": "PT Serif:bold", "font.size": 16,
    "axes.labelsize": 16, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "pdf.fonttype": 42, "ps.fonttype": 42,
})

REPORT = json.loads((OUT / "ascent-results.json").read_text())
CASES = {name: dict(np.load(OUT / f"ascent-{name}.npz"))
         for name in ("nominal", "steep", "shallow")}
D = CASES["nominal"]
T = D["t"]
CONST = REPORT["constants"]
MU = CONST["mu_km3_s2"]
RE = CONST["earth_radius_km"]
TARGET_H = CONST["target_height_km"]
VC = CONST["target_speed_km_s"]
T1 = REPORT["phases"][2]["time_end_s"]
P = REPORT["solved_parameters_s"]
TF = float(T[-1])
EDGES = [T1, T1 + P[0], T1 + P[0] + P[1]]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.045,
                metadata={"Title": name, "Author": "Lecture 10"})
    fig.savefig(QA / f"{name}.png", dpi=175, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)


def single(figsize=(9.8, 3.0)):
    return plt.subplots(figsize=figsize, layout="constrained")


def pair():
    return plt.subplots(1, 2, figsize=(10.0, 3.1), layout="constrained")


def time_axis(ax):
    for x in EDGES:
        ax.axvline(x, color=GREY, lw=.8, ls="--", alpha=.65)
    ax.set_xlim(0, TF)
    ax.set_xlabel("Время от старта, с")
    ax.set_xticks([0, 200, 400, 600])


def trajectory():
    fig, ax = single((9.8, 3.3))
    names = ["vertical", "pitch", "turn", "raise", "coast", "circularize"]
    labels = ["Вертикальный подъём", "Начальный разворот", "Гравитационный разворот",
              "Первый верхний участок", "Пассивный участок", "Довыведение"]
    for name, label, color in zip(names, labels, [BLUE, PURPLE, TEAL, ORANGE, GREY, RED]):
        mask = D["mode"] == name
        ax.plot(RE * D["phi"][mask], D["height"][mask], color=color, lw=2.7, label=label)
    ax.axhline(TARGET_H, color=BLUE, lw=1, ls=":")
    ax.set(xlabel="Дальность по поверхности, км", ylabel="Высота, км")
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=12.5,
              columnspacing=1, handlelength=1.6, labelspacing=.25)
    save(fig, "ascent-trajectory-slide")


def state_history():
    fig, axes = pair()
    axes[0].plot(T, D["height"], lw=2.5, color=BLUE)
    axes[0].axhline(TARGET_H, lw=1, ls=":", color=RED)
    axes[0].set_ylabel("Высота, км")
    axes[1].plot(T, D["vr"], lw=2.4, color=ORANGE, label=r"$V_r$")
    axes[1].plot(T, D["vt"], lw=2.4, color=TEAL, label=r"$V_t$")
    axes[1].axhline(VC, lw=1, ls=":", color=RED)
    axes[1].set_ylabel("Скорость, км/с")
    axes[1].legend(loc="upper left", ncol=2, frameon=False)
    for ax in axes:
        time_axis(ax)
    save(fig, "ascent-height-speed-slide")

    fig, ax = single()
    ax.plot(T, D["gamma"], lw=2.5, color=BLUE, label=r"$\gamma$ — скорость")
    theta = D["theta"].copy()
    theta[D["mode"] == "coast"] = np.nan
    ax.plot(T, theta, lw=2.1, color=ORANGE, label=r"$\theta$ — тяга")
    time_axis(ax)
    ax.set_ylabel("Угол, град")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "ascent-angles-slide")


def propulsion():
    fig, axes = pair()
    axes[0].plot(T, D["z"][4] / 1000, lw=2.5, color=BLUE)
    axes[0].set_ylabel("Масса, т")
    axes[1].plot(T, D["force"] / 1e6, lw=2.5, color=TEAL)
    axes[1].set_ylabel("Тяга, МН")
    for ax in axes:
        time_axis(ax)
    save(fig, "ascent-propulsion-slide")

    fig, ax = single()
    ax.plot(T, D["load"], lw=2.5, color=ORANGE)
    ax.axhline(REPORT["constraints"]["max_thrust_load_g0"], color=RED, ls=":",
               lw=1.2, label="Учебное ограничение")
    ax.set_ylabel(r"$n_T=F/(mg_0)$")
    time_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "ascent-load-slide")


def aeroloads():
    fig, axes = pair()
    early = T <= T1
    axes[0].plot(T[early], D["dynamic_pressure"][early] / 1000, lw=2.5, color=BLUE)
    axes[0].axhline(50., color=RED, ls=":", label="50 кПа")
    axes[0].set_ylabel(r"$q_{\rm dyn}$, кПа")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].plot(T[early], D["qalpha"][early] / 1000, lw=2.5, color=ORANGE)
    axes[1].set_ylabel(r"$q_{\rm dyn}|\alpha|$, кПа·рад")
    for ax in axes:
        ax.set_xlim(0, T1)
        ax.set_xticks([0, 40, 80, 120])
        ax.set_xlabel("Время от старта, с")
        for x in [CONST["vertical_duration_s"],
                  CONST["vertical_duration_s"] + CONST["pitch_duration_s"]]:
            ax.axvline(x, color=GREY, lw=.8, ls="--", alpha=.65)
    save(fig, "ascent-aeroloads-slide")


def osculating():
    fig, axes = pair()
    axes[0].plot(T, D["perigee"], lw=2.4, color=BLUE, label="Перигей")
    axes[0].plot(T, D["apogee"], lw=2.4, color=ORANGE, label="Апогей")
    axes[0].axhline(0, color=GREY, lw=.8)
    axes[0].axhline(TARGET_H, ls=":", color=RED, lw=1)
    axes[0].set_ylabel("Высота апсиды, км")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].plot(T, D["energy"], lw=2.5, color=TEAL)
    axes[1].set_ylabel(r"$\mathcal{E}$, км²/с²")
    for ax in axes:
        time_axis(ax)
    save(fig, "ascent-osculating-orbit-slide")


def comparison():
    fig, axes = pair()
    for name, label, color in [("steep", "Круче: 16°", PURPLE),
                               ("nominal", "Номинал: 20°", BLUE),
                               ("shallow", "Пологее: 24°", ORANGE)]:
        d = CASES[name]
        axes[0].plot(d["t"], d["height"], lw=2.4, color=color, label=label)
        axes[1].plot(d["t"], d["gamma"], lw=2.4, color=color)
    axes[0].axhline(TARGET_H, ls=":", color=RED, lw=1)
    axes[0].set_ylabel("Высота, км")
    axes[0].legend(loc="lower right", frameon=False, fontsize=12.5)
    axes[1].set_ylabel(r"$\gamma$, град")
    axes[1].axhline(0, color=GREY, lw=.8)
    for ax in axes:
        time_axis(ax)
    save(fig, "ascent-program-comparison-slide")


def losses():
    fig, axes = pair()
    for j, label, color in [(6, "Гравитационные", BLUE),
                            (7, "Аэродинамические", TEAL),
                            (8, "Рулевые", ORANGE)]:
        axes[0].plot(T, 1000 * D["z"][j], lw=2.3, color=color, label=label)
    axes[0].set_ylabel("Потери, м/с")
    axes[0].legend(loc="upper left", frameon=False, fontsize=12.5)
    axes[1].plot(T, D["z"][5], lw=2.4, color=BLUE, label=r"$\Delta V_{\rm th}$")
    axes[1].plot(T, D["speed"], lw=2.4, color=TEAL, label=r"$V$")
    axes[1].set_ylabel("Скорость, км/с")
    axes[1].legend(loc="upper left", frameon=False)
    for ax in axes:
        time_axis(ax)
    save(fig, "ascent-losses-slide")


def convergence():
    fig, ax = single((5.2, 3.6))
    hist = REPORT["history"]
    ax.semilogy([v["iteration"] for v in hist], [v["norm"] for v in hist],
                "o-", color=BLUE, lw=2.4, ms=5)
    ax.axhline(REPORT["residual_tolerance"], ls=":", color=RED, label="Допуск")
    ax.set(xlabel="Номер итерации", ylabel=r"$\|\mathcal{R}\|_2$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", frameon=False)
    save(fig, "ascent-convergence-slide")


def final_insertion():
    fig, axes = pair()
    mask = D["mode"] == "circularize"
    local = T[mask] - T[mask][0]
    axes[0].plot(local, 1000 * D["vr"][mask], lw=2.5, color=ORANGE)
    axes[0].set_ylabel(r"$V_r$, м/с")
    deficit = 1000 * (D["vt"][mask] - np.sqrt(MU / D["radius"][mask]))
    axes[1].plot(local, deficit, lw=2.5, color=TEAL)
    axes[1].set_ylabel(r"$V_t-\sqrt{\mu/r}$, м/с")
    for ax in axes:
        ax.axhline(0, color=GREY, lw=.8)
        ax.set_xlim(0, local[-1])
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Время от начала довыведения, с", fontsize=14)
    save(fig, "ascent-final-insertion-slide")


if __name__ == "__main__":
    trajectory()
    state_history()
    propulsion()
    aeroloads()
    osculating()
    comparison()
    losses()
    convergence()
    final_insertion()
