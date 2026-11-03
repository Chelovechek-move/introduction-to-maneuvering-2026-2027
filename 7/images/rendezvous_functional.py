#!/usr/bin/env python3
"""Classical-element, multi-revolution linear-rendezvous functional for lecture 7.

Run: python3 7/images/rendezvous_functional.py [--grid 1100]
Replot only the compact slide version from verified build data: --slide-only

This is a newly declared classical-element example in the visual form of Fig. 4
of the author's masters thesis, not a reconstruction of that figure's numbers.
Reference: (a,e,omega,i,Omega,M)=(6800 km,.01,0,45 deg,0,0), central gravity.
tf=86400 s; the REQUIRED FINAL difference is (1 km,.01,1 deg,.1 deg,.1 deg,1 deg).
It is not an initial relative state. At each pair of true unwrapped longitudes
we solve [Phi(tf-t1)B(t1),Phi(tf-t2)B(t2)] dV = dyf, using the checked classical
Gauss matrix and Kepler STM from numerics.py. Controls are local RTN vectors.
The axes use L=Omega+omega+nu, in radians. Revolutions are retained in t(L).

The square is symmetric because swapping both impulses preserves their norm
sum. Only L1<L2 represents chronological controls; stationary solution markers
are restricted to that triangle. All results concern the linear equation, not
an optimized nonlinear controlled trajectory. No external CSV or MEE are used.
Derived numerical data and the preview are stored in ../build/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rendezvous-functional")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

from numerics import MU, PAPER, INK, GREEN, gauss_B, classical_phi, kepler

TF = 86400.
# INTERNAL order of numerics.py: (a,e,i,Omega,omega,M), km,s,rad.
Y0 = np.array([6800., .01, np.pi/4, 0., 0., 0.])
DY = np.array([1., .01, *np.deg2rad([.1, .1, 1., 1.])])
N = np.sqrt(MU / Y0[0]**3)
SCALE = np.array([1/Y0[0], 1., 1., 1., 1., 1.])
RHS = SCALE*DY


def time_from_longitude(L):
    """True anomaly equals L for this reference; inverse Kepler, unwrapped."""
    L = np.asarray(L)
    # E-nu is a small continuous correction. This avoids a +/-pi branch cut,
    # including the floating-point neighbors of odd multiples of pi.
    E = L-2*np.arctan2(Y0[1]*np.sin(L), 1+np.sqrt(1-Y0[1]**2)+Y0[1]*np.cos(L))
    return (E-Y0[1]*np.sin(E))/N


def longitude_from_time(t):
    t = np.asarray(t)
    E = np.array([kepler(float(m), Y0[1]) for m in np.ravel(N*t)]).reshape(t.shape)
    return E+2*np.arctan2(Y0[1]*np.sin(E), 1+np.sqrt(1-Y0[1]**2)-Y0[1]*np.cos(E))


LF = float(longitude_from_time(TF))


def block(L):
    t = float(time_from_longitude(L))
    y = Y0.copy()
    y[5] = N*t
    return (classical_phi(y, TF-t)@gauss_B(y))*SCALE[:, None]


def solve_pair(x):
    matrix = np.hstack([block(x[0]), block(x[1])])
    dv = np.linalg.solve(matrix, RHS)
    return dv, matrix


def objective(x):
    if not (0 <= x[0] < x[1] <= LF):
        return 1e10
    try:
        dv, matrix = solve_pair(x)
    except np.linalg.LinAlgError:
        return 1e10
    if np.max(np.abs(matrix@dv-RHS)) > 1e-10:
        return 1e10
    return 1000*(np.linalg.norm(dv[:3])+np.linalg.norm(dv[3:]))


def grid_values(longitudes):
    blocks = np.array([block(L) for L in longitudes])
    size = len(longitudes)
    values = np.full((size, size), np.nan)
    for j0 in range(0, size, 48):
        jj = np.arange(j0, min(j0+48, size))
        first = np.broadcast_to(blocks[None, :, :, :], (len(jj), size, 6, 3))
        second = np.broadcast_to(blocks[jj, None, :, :], first.shape)
        matrix = np.concatenate([first, second], axis=-1)
        matrix[np.arange(len(jj)), jj] = np.eye(6)
        rhs = np.broadcast_to(RHS, matrix.shape[:-1])
        dv = np.linalg.solve(matrix, rhs)
        cost = 1000*(np.linalg.norm(dv[:, :, :3], axis=-1)+np.linalg.norm(dv[:, :, 3:], axis=-1))
        residual = np.max(np.abs(np.einsum("...ij,...j->...i", matrix, dv)-RHS), axis=-1)
        cost[(residual > 1e-10) | ~np.isfinite(cost)] = np.nan
        cost[np.arange(len(jj)), jj] = np.nan
        values[jj] = cost
    return values


def local_minima(longitudes, values):
    safe = np.where(np.isfinite(values), values, np.inf)
    candidates = np.argwhere(safe == minimum_filter(safe, size=3, mode="constant", cval=np.inf))
    candidates = [(j, i) for j, i in candidates if i < j and np.isfinite(safe[j, i])]
    solutions = []
    for j, i in candidates:
        result = minimize(objective, [longitudes[i], longitudes[j]], method="Nelder-Mead",
                          bounds=[(0, LF), (0, LF)],
                          options={"xatol": 2e-9, "fatol": 1e-8, "maxiter": 500})
        x = result.x
        if result.fun >= 1e8 or not result.success:
            continue
        if any(np.linalg.norm(x-r[0]) < 1e-4 for r in solutions):
            continue
        # Check nearby admissible directions; allow boundary minima.
        epsilon = 2e-4
        directions = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]])
        neighbors = [objective(x+epsilon*d) for d in directions]
        if min(neighbors) < result.fun-2e-7:
            continue
        dv, matrix = solve_pair(x)
        residual = float(np.max(np.abs(matrix@dv-RHS)))
        solutions.append((x.copy(), float(result.fun), residual, dv.copy()))
    solutions.sort(key=lambda r: r[1])
    return solutions, len(candidates)


def stationarity_checks(solutions):
    """Central differences corroborate interior stationarity and positive Hessian."""
    diagnostics = []
    h = 1e-4
    axes = np.eye(2)*h
    for x, f, _, _ in solutions:
        if min(x[0], LF-x[1], x[1]-x[0]) <= 2*h:
            continue
        fp = np.array([objective(x+d) for d in axes])
        fm = np.array([objective(x-d) for d in axes])
        gradient = (fp-fm)/(2*h)
        diagonal = (fp+fm-2*f)/h**2
        cross = (objective(x+axes[0]+axes[1])-objective(x+axes[0]-axes[1])
                 -objective(x-axes[0]+axes[1])+objective(x-axes[0]-axes[1]))/(4*h*h)
        hessian = [[diagonal[0],cross],[cross,diagonal[1]]]
        diagnostics.append([np.linalg.norm(gradient), np.linalg.eigvalsh(hessian)[0]])
    checks = np.array(diagnostics)
    if np.max(checks[:,0]) > 1e-3 or np.min(checks[:,1]) <= 0:
        raise AssertionError("A reported interior minimum failed stationarity/curvature checks")
    return {"interior_minimum_count": len(checks),
            "boundary_minimum_count": len(solutions)-len(checks),
            "stationarity_difference_step_rad": h,
            "max_interior_gradient_norm_mps_per_rad": float(np.max(checks[:,0])),
            "minimum_interior_hessian_eigenvalue_mps_per_rad2": float(np.min(checks[:,1]))}


def draw_map(longitude, values, positions, output, build_dir, slide=False):
    """Keep the full scientific figure and provide a large-type slide variant."""
    style = {"font.size": 14 if slide else 12, "axes.labelsize": 17 if slide else 15,
             "xtick.labelsize": 14 if slide else 11, "ytick.labelsize": 14 if slide else 11,
             "axes.spines.top": True, "axes.spines.right": True}
    with plt.rc_context(style):
        if slide:
            fig = plt.figure(figsize=(7,6))
            # A physical square occupies 87.5% of the canvas height. Explicit
            # axes avoid the large margins produced by equal aspect + layout.
            ax = fig.add_axes([.103, .105, .875*6/7, .875])
            ax.tick_params(pad=2, length=3)
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2
        else:
            fig, ax = plt.subplots(figsize=(8.3,7.1), layout="constrained")
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(PAPER)
        step = longitude[1]-longitude[0]
        extent = (-step/2, LF+step/2, -step/2, LF+step/2)
        image = ax.imshow(np.ma.masked_invalid(values), origin="lower", extent=extent,
                          interpolation="none", cmap=cmap, vmin=50, vmax=600,
                          aspect="equal", rasterized=True)
        ax.scatter(positions[:,0], positions[:,1], s=7 if slide else 10, c="white", edgecolors=INK,
                   linewidths=.2 if slide else .35, zorder=3)
        ax.scatter(positions[:50,0], positions[:50,1], s=10 if slide else 14, c=GREEN, edgecolors=INK,
                   linewidths=.2 if slide else .35, zorder=4)
        ticks = np.arange(0, LF+.01, (8 if slide else 4)*np.pi)
        labels = [f"{v:.1f}" if v else "0" for v in ticks]
        ax.set(xticks=ticks, yticks=ticks, xlim=(0,LF), ylim=(0,LF),
               xlabel=r"$L_1$ [рад]", ylabel=r"$L_2$ [рад]")
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.grid(color=INK, linewidth=.25, alpha=.25)
        if not slide:
            handles = [Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor=c,
                              markeredgecolor=INK, markeredgewidth=.4) for c in ["white",GREEN]]
            ax.legend(handles, ["Локальные минимумы", "50 лучших из найденных"], loc="lower right",
                      fontsize=10, facecolor=PAPER, framealpha=.94, edgecolor="#D6D0C5")
        if slide:
            color_axes = fig.add_axes([.873, .105, .018, .875])
            bar = fig.colorbar(image, cax=color_axes, extend="max", ticks=[50,200,400,600])
            bar.ax.tick_params(labelsize=14, pad=2, length=3)
            bar.set_label(r"$J$ [м/с]", fontsize=17, labelpad=3)
            fig.canvas.draw()
            bounds = ax.get_position().bounds
            size = fig.get_size_inches()
            tight = fig.get_tightbbox(fig.canvas.get_renderer())
            if tight.x0 < 0 or tight.y0 < 0 or tight.x1 > size[0] or tight.y1 > size[1]:
                raise AssertionError("Slide labels extend outside the PDF canvas")
            print(json.dumps({"slide_canvas_mm": (size*25.4).tolist(),
                              "heatmap_bbox_fraction": list(bounds),
                              "heatmap_height_fraction": bounds[3],
                              "heatmap_area_fraction": bounds[2]*bounds[3],
                              "heatmap_size_mm": [bounds[2]*size[0]*25.4,bounds[3]*size[1]*25.4],
                              "heatmap_size_when_canvas_height62mm": [bounds[2]*size[0]/size[1]*62,bounds[3]*62],
                              "labels_at62mm_pt": 17*62/(size[1]*25.4),
                              "ticks_at62mm_pt": 14*62/(size[1]*25.4),
                              "label_ink_bbox_inches": list(tight.bounds)}))
        else:
            bar = fig.colorbar(image, ax=ax, fraction=.046, pad=.025, extend="max")
            bar.set_label(r"$J$ [м/с]", fontsize=15)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300 if slide else 240)
        preview = "rendezvous-functional-slide-preview.png" if slide else "rendezvous-functional-preview.png"
        fig.savefig(build_dir/preview, dpi=200 if slide else 160)
        plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=1100)
    parser.add_argument("--output", type=Path, default=here/"rendezvous-functional.pdf")
    parser.add_argument("--build-dir", type=Path, default=here.parent/"build")
    parser.add_argument("--reuse-search", action="store_true",
                        help="Reuse local build/ minimum coordinates; recompute grid, costs and all checks")
    parser.add_argument("--slide-only", action="store_true",
                        help="Only draw the compact slide PDF from previously verified local build data")
    args = parser.parse_args()
    if args.grid < 100:
        raise ValueError("Use at least 100 longitude samples")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    slide_output = args.output.with_name(args.output.stem+"-slide.pdf")
    if args.slide_only:
        cache = np.load(args.build_dir/"rendezvous-functional-data.npz")
        for key, expected in [("reference_internal",Y0), ("final_difference_internal",DY), ("tf_s",TF)]:
            if not np.array_equal(cache[key],expected):
                raise ValueError("Cached data belong to another rendezvous problem")
        draw_map(cache["longitude_rad"], cache["J_mps"], cache["minima_L_rad"],
                 slide_output, args.build_dir, slide=True)
        print(f"Slide figure: {slide_output}")
        return
    longitude = np.linspace(0, LF, args.grid)
    values = grid_values(longitude)
    if args.reuse_search:
        cache = np.load(args.build_dir/"rendezvous-functional-data.npz")
        if not np.allclose(cache["longitude_rad"], longitude, rtol=0, atol=1e-12):
            raise ValueError("Cached search belongs to another longitude grid")
        for key, expected in [("reference_internal",Y0), ("final_difference_internal",DY), ("tf_s",TF)]:
            if key in cache and not np.array_equal(cache[key],expected):
                raise ValueError("Cached search belongs to another rendezvous problem")
        solutions = []
        for x in cache["minima_L_rad"]:
            dv, matrix = solve_pair(x)
            solutions.append((x, objective(x), float(np.max(np.abs(matrix@dv-RHS))), dv))
        solutions.sort(key=lambda r:r[1])
        safe_values = np.where(np.isfinite(values), values, np.inf)
        seeds = safe_values == minimum_filter(safe_values, size=3, mode="constant", cval=np.inf)
        seed_count = int(np.sum(seeds & np.isfinite(values) & np.triu(np.ones_like(values,dtype=bool),1).T))
    else:
        solutions, seed_count = local_minima(longitude, values)
    if len(solutions) < 50:
        raise RuntimeError(f"Only {len(solutions)} verified local minima; inspect the grid/search")
    stationarity = stationarity_checks(solutions)

    time_samples = np.linspace(0, TF, 1001)
    # Explicitly cover the old atan2 branch boundaries and adjacent machine
    # numbers: a regular time grid alone would miss these edge cases.
    boundaries = np.pi*np.arange(int(LF/np.pi)+1)
    edge_longitudes = np.concatenate([boundaries, np.nextafter(boundaries,-np.inf),
                                     np.nextafter(boundaries,np.inf)])
    time_samples = np.concatenate([time_samples, boundaries/N, time_from_longitude(edge_longitudes)])
    roundtrip_error = float(np.max(np.abs(time_from_longitude(longitude_from_time(time_samples))-time_samples)))
    if roundtrip_error > 2e-9:
        raise AssertionError("Longitude/time conversion lost revolutions")
    safe = np.isfinite(values) & np.isfinite(values.T) & (values < 1000) & (values.T < 1000)
    symmetry_error = float(np.max(np.abs(values[safe]-values.T[safe])))
    if symmetry_error > 1e-6:
        raise AssertionError("Swapping impulses changed characteristic velocity")
    validation = []
    for x in [solutions[0][0], np.array([.7, 7.4]), np.array([20., 58.]), np.array([70., 95.])]:
        dv, matrix = solve_pair(x)
        original_residual = (matrix@dv-RHS)/SCALE
        validation.append({"L_rad": x.tolist(), "t_s": time_from_longitude(x).tolist(),
                           "dV_RTN_mps": (1000*dv).reshape(2,3).tolist(),
                           "residual_km_rad_internal_order": original_residual.tolist(),
                           "J_mps": objective(x)})

    np.savez_compressed(args.build_dir/"rendezvous-functional-data.npz", longitude_rad=longitude,
                        J_mps=values, minima_L_rad=np.array([r[0] for r in solutions]),
                        minima_J_mps=np.array([r[1] for r in solutions]),
                        reference_internal=Y0, final_difference_internal=DY, tf_s=TF)
    positions = np.array([r[0] for r in solutions])
    draw_map(longitude, values, positions, args.output, args.build_dir)
    draw_map(longitude, values, positions, slide_output, args.build_dir, slide=True)

    public = [0,1,4,2,3,5]
    report = {"model": "central-field, classical Gauss B + exact Kepler first-variation Phi",
              "public_element_order": ["a_km","e","omega_rad","i_rad","Omega_rad","M_rad"],
              "reference_elements": Y0[public].tolist(), "required_final_difference": DY[public].tolist(),
              "tf_s": TF, "mu_km3_s2": MU, "longitude_units": "rad", "J_units": "m/s",
              "longitude_range_rad": [0,LF], "longitude_span_revolutions": LF/(2*np.pi),
              "grid_shape": list(values.shape), "grid_min_J_mps": float(np.nanmin(values)),
              "grid_seed_count": seed_count, "verified_minimum_count": len(solutions),
              "best_found_J_mps": solutions[0][1], "best_found_L_rad": positions[0].tolist(),
              "max_time_roundtrip_error_s": roundtrip_error,
              "max_permutation_symmetry_error_for_J_below1000_mps": symmetry_error,
              "max_scaled_residual_at_minima": max(r[2] for r in solutions),
              "best_50_cutoff_J_mps": solutions[49][1], "validation_pairs": validation,
              **stationarity,
              "scope": "linear terminal equation only; local grid-seeded search, no global optimality claim",
              "display_color_limits_mps": [50,600], "output": str(args.output.resolve())}
    (args.build_dir/"rendezvous-functional-checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
