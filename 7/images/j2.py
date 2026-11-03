"""J2 illustrations for lecture 7; lengths km, time s, angles rad.

The planar test is the equatorial specialization of Schweighart--Sedwick
(2002), Eq. (45).  The CW frequency is matched to the actual chief period.
The second figure is an independent averaged nodal-geometry calculation,
not a validation of the full three-dimensional SS model.
"""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lecture7-matplotlib")
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numerics import save_slide_variant

for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))

OUT = Path(__file__).resolve().parent
BUILD = OUT.parent / "build"
BUILD.mkdir(exist_ok=True)
MU, RE, J2 = 398600.4418, 6378.1363, 1.08262668e-3
RADIUS, AMP = 7000.0, 0.1
N = np.sqrt(MU / RADIUS**3)
S = 1.5 * J2 * (RE / RADIUS)**2
C = np.sqrt(1 + S)
OMEGA = N * C
BETA = N * np.sqrt(1 - S)
PERIOD = 2 * np.pi / OMEGA
COL = dict(exact="#302B28", cw="#B57714", ss="#187AA5", other="#087F78")

plt.rcParams.update({
    "font.family": "PT Serif", "font.size": 10,
    "mathtext.fontset": "custom", "mathtext.rm": "PT Serif",
    "mathtext.it": "PT Serif:italic", "mathtext.bf": "PT Serif:bold",
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "figure.facecolor": "#FBF9F3", "axes.facecolor": "#FBF9F3",
    "text.color": "#302B28", "axes.labelcolor": "#302B28",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def gravity(r):
    d = np.linalg.norm(r)
    z2 = (r[2] / d)**2
    return -MU * r / d**3 + (1.5 * J2 * MU * RE**2 / d**5) * np.array([
        r[0] * (5 * z2 - 1), r[1] * (5 * z2 - 1), r[2] * (5 * z2 - 3)])


def propagate(tolerance):
    # rho=(A,0,0), rhodot=(0,-2*OMEGA*A,0) in the chief RTN.
    rc = np.array([RADIUS, 0., 0.])
    vc = np.array([0., RADIUS * OMEGA, 0.])
    rd = rc + np.array([AMP, 0., 0.])
    vd = vc + np.array([0., -OMEGA * AMP, 0.])
    state = np.r_[rc, vc, rd, vd]

    def rhs(t, y):
        return np.r_[y[3:6], gravity(y[:3]), y[9:12], gravity(y[6:9])]

    times = np.linspace(0, 30 * PERIOD, 9001)
    sol = solve_ivp(rhs, (times[0], times[-1]), state, t_eval=times,
                    rtol=tolerance, atol=tolerance * .01, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    rc, vc, rd = sol.y[:3].T, sol.y[3:6].T, sol.y[6:9].T
    er = rc / np.linalg.norm(rc, axis=1)[:, None]
    en = np.cross(rc, vc)
    en /= np.linalg.norm(en, axis=1)[:, None]
    et = np.cross(en, er)
    dr = rd - rc
    exact = np.column_stack([(dr * b).sum(axis=1) for b in (er, et, en)])
    ss = np.column_stack([AMP * np.cos(BETA * times),
                          -2 * OMEGA / BETA * AMP * np.sin(BETA * times),
                          np.zeros_like(times)])
    cw = np.column_stack([AMP * np.cos(OMEGA * times),
                          -2 * AMP * np.sin(OMEGA * times),
                          np.zeros_like(times)])
    return times, exact, ss, cw


def make_planar():
    t, exact, ss, cw = propagate(2e-13)
    _, exact_check, _, _ = propagate(8e-14)
    numerical_delta_m = float(np.max(np.linalg.norm(exact - exact_check, axis=1)) * 1000)
    e_ss = np.linalg.norm(ss - exact, axis=1) * 1000
    e_cw = np.linalg.norm(cw - exact, axis=1) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), constrained_layout=True)
    sel = t >= 27 * PERIOD
    for y, name, color, ls in [(exact, "Нелинейная модель $J_2$", COL["exact"], "-"),
                               (ss, "SS: плоский частный случай", COL["ss"], "--"),
                               (cw, r"CW: частота $\omega_{\rm ref}$", COL["cw"], "-")]:
        axes[0].plot(t[sel] / PERIOD, y[sel, 1] * 1000, color=color, ls=ls,
                     lw=1.7, label=name)
    axes[0].set(xlabel="Время, витки", ylabel=r"$y$, м",
                title="Продольное движение на последних витках")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].semilogy(t[1:] / PERIOD, np.maximum(e_cw[1:], 1e-6),
                     color=COL["cw"], label="CW")
    axes[1].semilogy(t[1:] / PERIOD, np.maximum(e_ss[1:], 1e-6),
                     color=COL["ss"], label="SS")
    axes[1].set(xlabel="Время, витки", ylabel="Ошибка положения, м",
                title="Сравнение с нелинейным прогнозом")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.22)
    fig.savefig(OUT / "j2-relative.pdf")
    save_slide_variant(fig, "j2-relative")
    plt.close(fig)
    return dict(radius_km=RADIUS, amplitude_m=AMP*1000, inclination_deg=0,
                s=S, omega_rad_s=OMEGA, beta_rad_s=BETA, period_s=PERIOD,
                max_cw_error_m=float(e_cw.max()), max_ss_error_m=float(e_ss.max()),
                numerical_check_delta_m=numerical_delta_m,
                convention="same physical initial states; CW tuned to actual chief period",
                scope="equatorial in-plane SS special case, no full spatial SS validation")


def make_nodal():
    inclination = np.deg2rad(35.)
    di, dnode = 1e-5, 1e-5
    i1, i2 = inclination + di, inclination
    node_rate = lambda inc: -1.5 * N * J2 * (RE / RADIUS)**2 * np.cos(inc)
    t = np.linspace(0, 30 * PERIOD, 1601)
    dn = dnode + (node_rate(i1) - node_rate(i2)) * t
    # atan2(norm cross,dot) avoids cancellation in acos for close normals.
    normals1 = np.column_stack([np.sin(i1)*np.sin(dn),
                                -np.sin(i1)*np.cos(dn), np.full_like(t, np.cos(i1))])
    normals2 = np.tile([0., -np.sin(i2), np.cos(i2)], (len(t), 1))
    phi = np.arctan2(np.linalg.norm(np.cross(normals1, normals2), axis=1),
                     np.sum(normals1*normals2, axis=1))
    amplitude = RADIUS * phi * 1000
    slope = RADIUS * 1000 * np.sin(i1)*np.sin(i2)*np.sin(dnode)/np.sin(phi[0]) * (
        node_rate(i1)-node_rate(i2))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    axes[0].plot(t/PERIOD, np.rad2deg(dn)*3600, color=COL["ss"], lw=1.8)
    axes[0].set(xlabel="Время, витки", ylabel=r"$\Omega_1-\Omega_2$, угл. с",
                title="Разность долгот узлов")
    axes[1].plot(t/PERIOD, amplitude, color=COL["other"], lw=1.8,
                 label=r"$A(t)$")
    axes[1].plot(t/PERIOD, amplitude[0]+slope*t, color=COL["cw"], ls="--",
                 label="Линейная огибающая")
    axes[1].set(xlabel="Время, витки", ylabel="Поперечная огибающая, м",
                title="Изменение взаимного наклона")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.22)
    fig.savefig(OUT / "j2-frequencies.pdf")
    save_slide_variant(fig, "j2-frequencies")
    plt.close(fig)
    return dict(inclination_deg=35, delta_i_rad=di, delta_node_initial_rad=dnode,
                amplitude_initial_m=float(amplitude[0]), amplitude_final_m=float(amplitude[-1]),
                amplitude_slope_m_s=float(slope),
                scope="averaged nodal geometry; not a full SS propagation")


if __name__ == "__main__":
    data = {"planar": make_planar(), "nodal": make_nodal()}
    (BUILD / "j2-results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    p = data["planar"]
    def ru(x, fmt=".3f"):
        return format(x, fmt).replace(".", "{,}")
    (OUT / "j2-values.tex").write_text(
        "% Generated by images/j2.py; scientific units stated in the manuscript.\n"
        + rf"\newcommand{{\JTwoCWError}}{{{ru(p['max_cw_error_m'])}}}"+"\n"
        + rf"\newcommand{{\JTwoSSError}}{{{ru(p['max_ss_error_m'])}}}"+"\n"
        + rf"\newcommand{{\JTwoPeriod}}{{{ru(p['period_s'], '.2f')}}}"+"\n")
    print(json.dumps(data, ensure_ascii=False, indent=2))
