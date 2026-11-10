#!/usr/bin/env python3
"""Trajectory figure for the current seven-manoeuvre Keplerian example.

Run from the repository root: python3 8/images/kepler_orbit.py
Reuse saved dense data: python3 8/images/kepler_orbit.py --figures-only

The final group vectors and site offset come from kepler-results.json.
Only the final strict forecast is repeated by the same Gauss + J2 equations,
including its event search, with a 20-second output mesh. No targeting or
Cartesian propagation is performed.
Positions are reconstructed algebraically from the propagated elements for
the equatorial projection. Numerical data, verification and both figure
variants are saved here; raster previews and caches go to ../build/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build" / "trajectory"
BUILD.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "mplconfig"))

import numpy as np
import kepler_transfer as model
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

PAPER, INK = "#FBF9F3", "#302B28"
BLUE, ORANGE, PURPLE, GREY = "#187AA5", "#B57714", "#79539C", "#6D6560"
for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    "font.family": "PT Serif", "font.size": 14,
    "mathtext.fontset": "custom", "mathtext.rm": "PT Serif",
    "mathtext.it": "PT Serif:italic", "mathtext.bf": "PT Serif:bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": False, "pdf.fonttype": 42,
})


def positions(elements):
    """Inertial position [km] from classical elements; also accepts (7,N)."""
    a, e, inc, node, arg, nu = np.asarray(elements)[:6]
    r = a * (1 - e * e) / (1 + e * np.cos(nu))
    u = arg + nu
    co, so, cu, su = np.cos(node), np.sin(node), np.cos(u), np.sin(u)
    ci, si = np.cos(inc), np.sin(inc)
    return np.array([r * (co * cu - so * su * ci),
                     r * (so * cu + co * su * ci), r * su * si])


def forecast(result):
    """Repeat the strict final forecast, including the same event search."""
    model.RECORD_STEP_PASSIVE_S = model.RECORD_STEP_ACTIVE_S = 20.
    group_vectors = np.asarray(result["targeted_group_dv_kms"])
    offset = np.deg2rad(result["targeted_second_group_true_anomaly_offset_deg"])
    state, burns, trace = model.finite(group_vectors, use_j2=True, offset=offset,
                                      tight=True, record=True)
    t, z = trace["t"], trace["z"]
    burn_id = np.zeros(t.size, dtype=np.int8)
    for j, burn in enumerate(burns, 1):
        mask = trace["active"] & (t >= burn["start_s"]) & (t <= burn["end_s"])
        burn_id[mask] = j
    xyz = positions(z)
    target = np.asarray(result["target_classical_elements"])
    nu = np.linspace(0., 2*np.pi, 1501)
    target_elements = np.vstack([np.full(nu.size, v) for v in target] + [nu])
    target_xyz = positions(target_elements)
    np.savez_compressed(HERE / "kepler-orbit-data.npz", t=t, z=z,
                        burn_id=burn_id, xyz_km=xyz, target_xyz_km=target_xyz)
    reference = np.asarray(result["verification"]["tight_final"]["classical_state_km_rad_kg"])
    residual = model.residual(state)
    report = {
        "source_sha256": hashlib.sha256((HERE / "kepler-results.json").read_bytes()).hexdigest(),
        "method": "Saved group vectors and site offset; strict final forecast with event search; classical Gauss + J2",
        "output_step_limit_s": 20., "point_count": int(t.size),
        "active_point_counts": [int(np.sum(burn_id == j)) for j in range(1, 8)],
        "final_state": state.tolist(), "final_residual": residual.tolist(),
        "final_residual_norm": float(np.linalg.norm(residual)),
        "final_difference_from_saved_strict_state": (state-reference).tolist(),
        "minimum_radius_km": float(np.linalg.norm(xyz, axis=0).min()),
        "projection": "Equatorial inertial XY; same scales on both coordinate axes",
        "burns": burns,
    }
    assert report["final_residual_norm"] < result["epsilon"]
    assert report["minimum_radius_km"] > result["constants"]["earth_radius_km"]
    assert all(report["active_point_counts"])
    (HERE / "kepler-orbit-verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def draw(result, slide=False):
    data = np.load(HERE / "kepler-orbit-data.npz")
    xyz, burn_id = data["xyz_km"] / 1000, data["burn_id"]
    target = data["target_xyz_km"] / 1000
    paper = PAPER if slide else "white"
    with plt.rc_context({"figure.facecolor": paper, "axes.facecolor": paper,
                         "savefig.facecolor": paper, "font.size": 15 if slide else 14,
                         "axes.labelsize": 15 if slide else 14,
                         "axes.titlesize": 16 if slide else 15,
                         "xtick.labelsize": 14 if slide else 13,
                         "ytick.labelsize": 14 if slide else 13}):
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.15),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
        passive = xyz[:2].copy()
        passive[:, burn_id != 0] = np.nan
        for ax in axes:
            ax.add_patch(Circle((0, 0), result["constants"]["earth_radius_km"]/1000,
                                color="#92B7CC", zorder=3))
            ax.plot(*passive, color=BLUE, lw=.8, alpha=.7, zorder=2)
            ax.plot(*target[:2], color=GREY, ls="--", lw=1.1, zorder=3)
            for j in range(1, 8):
                mask = burn_id == j
                ax.plot(*xyz[:2, mask], color=ORANGE if j <= 3 else PURPLE,
                        lw=2.8, zorder=4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("$X$, тыс. км", labelpad=3)
            ax.set_ylabel("$Y$, тыс. км", labelpad=3)
            ax.grid(color="#A9CADA", linewidth=.5, alpha=.45)
        axes[0].set(xlim=(-47, 47), ylim=(-47, 47), title="Экваториальная проекция")
        axes[0].set_xticks([-40, -20, 0, 20, 40])
        axes[0].set_yticks([-40, -20, 0, 20, 40])
        peri = xyz[:2, (burn_id >= 1) & (burn_id <= 3)]
        middle = (peri.min(axis=1) + peri.max(axis=1)) / 2
        half = max(np.ptp(peri, axis=1).max() / 2 + .4, 2.25)
        axes[1].set(xlim=(middle[0]-half, middle[0]+half),
                    ylim=(middle[1]-half, middle[1]+half),
                    title="Перигейные манёвры")
        for axis in (axes[1].xaxis, axes[1].yaxis):
            axis.set_major_locator(MaxNLocator(4))
        handles = [Line2D([], [], color=GREY, ls="--", lw=1.1),
                   Line2D([], [], color=BLUE, lw=1.),
                   Line2D([], [], color=ORANGE, lw=2.8),
                   Line2D([], [], color=PURPLE, lw=2.8)]
        fig.legend(handles, [r"Цель в $t_f$", "Пассивные участки", "Манёвры 1–3", "Манёвры 4–7"],
                   loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.51, -.04),
                   fontsize=14.5 if slide else 12.5, columnspacing=1.2, handlelength=2.2)
        fig.subplots_adjust(left=.075, right=.985, bottom=.205, top=.90, wspace=.25)
        name = "kepler-orbit-slide" if slide else "kepler-orbit"
        fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
        fig.savefig(BUILD / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=.04)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    result = json.loads((HERE / "kepler-results.json").read_text())
    if not args.figures_only:
        print(json.dumps(forecast(result), ensure_ascii=False, indent=2), flush=True)
    else:
        verification = json.loads((HERE / "kepler-orbit-verification.json").read_text())
        current_hash = hashlib.sha256((HERE / "kepler-results.json").read_bytes()).hexdigest()
        if verification["source_sha256"] != current_hash:
            raise ValueError("Saved trajectory belongs to different input data; run without --figures-only")
    draw(result)
    draw(result, slide=True)
