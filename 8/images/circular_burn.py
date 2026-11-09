"""A finite-burn example near a circular orbit, lecture 8.

Reproduce from the repository root with
    python3 8/images/circular_burn.py
Regenerate only figures from the saved scan with
    python3 8/images/circular_burn.py --figures-only

Units: km, s, rad internally; delta-v is reported in m/s.  This example has
constant ACCELERATION magnitude, no mass variation and no perturbations.
It isolates geometric effects; it is not a constant-thrust engine model.

The reference circle is in the XY plane and moves counterclockwise.
At t=0 the unforced reference state is (r0,0,0; 0,V0,0).
The burn window is [-tau/2,+tau/2], theta=n0*tau.  All compared exact
trajectories start from the same circular state at the start of this window.
The impulse reference is applied at t=0; its exact elements are analytic.
For the RTN law, acceleration is tangent to the CURRENT exact orbit.
For the inertial law, acceleration always points along the fixed +Y axis.

Linear equations and notation agree with manual/near-circular-transfer.tex,
eq:linear-element-dynamics.  The finite-arc geometry is discussed in
Baranov, Spacecraft Manoeuvring in the Vicinity of a Near-Circular Orbit,
chapter 6, secs. 6.1.1 and 6.1.3.  Our numerical values are independently
calculated and the constant-acceleration assumption is stated explicitly.

Let D=int |a_c|dt, k=sinc(theta/2), sinc(x)=sin(x)/x.  For constant
components in RTN and a burn centred at phi_c, the mean response matrix
mapping component integrals [D_S,D_T,D_W] to [da/r0,de_x,de_y,di_x,di_y]
is (1/V0) times
    [[0,             2,                 0],
     [k sin(phi_c),  2k cos(phi_c),      0],
     [-k cos(phi_c), 2k sin(phi_c),      0],
     [0,             0,                 k cos(phi_c)],
     [0,             0,                 k sin(phi_c)]].
Here i_x=i*cos(Omega), i_y=i*sin(Omega), and phi=n0*t.
For pure current-RTN transverse acceleration and phi_c=0:
    da = 2*r0*D/V0; de_x = 2*D/V0*k; de_y=0.
For fixed +Y acceleration, S=A*sin(phi), T=A*cos(phi):
    da = 2*r0*D/V0*k;
    de_x = (2*D/V0)*(3/4 + sinc(theta)/4); de_y=0.
Thus sinc is not a universal delta-v efficiency factor.  Different
orbital quantities and different attitude laws have different responses.
All formulas are first order in D/V0 on a circular reference orbit.
"""

from pathlib import Path
import json
import os

OUT = Path(__file__).resolve().parent
BUILD = OUT.parent / "build"
BUILD.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "mplconfig"))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))

MU = 398600.4418
R0 = 7000.0
V0 = np.sqrt(MU / R0)
N0 = V0 / R0
PERIOD = 2 * np.pi / N0
DV_M_S = 1.0
DV = DV_M_S / 1000
COL = dict(rtn="#187AA5", inertial="#087F78", impulse="#B57714",
           accent="#79539C", ink="#302B28", light="#D7DCE0")

plt.rcParams.update({
    "font.family": "PT Serif", "font.size": 14,
    "mathtext.fontset": "custom", "mathtext.rm": "PT Serif",
    "mathtext.it": "PT Serif:italic", "mathtext.bf": "PT Serif:bold",
    "axes.titlesize": 15, "axes.labelsize": 14, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": COL["ink"], "axes.labelcolor": COL["ink"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#687178", "xtick.color": "#687178",
    "ytick.color": "#687178", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def sinc(x):
    return np.sinc(np.asarray(x) / np.pi)


def mean_matrix(theta, phi_c=0.0):
    k = float(sinc(theta / 2))
    c, s = np.cos(phi_c), np.sin(phi_c)
    return np.array([[0, 2, 0], [k*s, 2*k*c, 0], [-k*c, 2*k*s, 0],
                     [0, 0, k*c], [0, 0, k*s]]) / V0


def linear_elements(theta, law, dv=DV):
    k = float(sinc(theta / 2))
    if law == "rtn":
        return np.array([2 * R0 * dv / V0, 2 * dv / V0 * k, 0.0])
    if law == "inertial":
        return np.array([2 * R0 * dv / V0 * k,
                         2 * dv / V0 * (.75 + .25 * float(sinc(theta))), 0.0])
    raise ValueError(law)


def elements(y):
    r, v = y[:3], y[3:]
    h = np.cross(r, v)
    radius = np.linalg.norm(r)
    a = 1 / (2 / radius - np.dot(v, v) / MU)
    e = np.cross(v, h) / MU - r / radius
    return np.r_[a - R0, e[:2]]


def exact_elements(theta, law, dv=DV, rtol=3e-13, atol=3e-14, steps=80):
    duration = theta / N0
    htheta = theta / 2
    c, s = np.cos(htheta), np.sin(htheta)
    state = np.array([R0*c, -R0*s, 0, V0*s, V0*c, 0])
    accel = dv / duration

    def rhs(t, y):
        r, v = y[:3], y[3:]
        radius = np.linalg.norm(r)
        if law == "rtn":
            er = r / radius
            en = np.cross(r, v)
            en /= np.linalg.norm(en)
            direction = np.cross(en, er)
        elif law == "inertial":
            direction = np.array([0., 1., 0.])
        else:
            raise ValueError(law)
        return np.r_[v, -MU*r/radius**3 + accel*direction]

    sol = solve_ivp(rhs, (-duration/2, duration/2), state, method="DOP853",
                    rtol=rtol, atol=atol, max_step=duration/steps)
    if not sol.success:
        raise RuntimeError(sol.message)
    return elements(sol.y[:, -1])


def impulse_elements(dv=DV):
    # Exact tangential impulse applied at r=(r0,0), v=(0,V0).
    q = dv / V0
    return np.array([R0 / (1-2*q-q*q) - R0, 2*q+q*q, 0.])


def style_axes(ax, theta=False):
    ax.grid(alpha=.18)
    if theta:
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.set_xlabel(r"Угловая протяжённость $\theta$, град")


def geometry():
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.05), constrained_layout=True)
    p = np.linspace(-np.pi, np.pi, 501)
    ph = np.linspace(-np.pi/3, np.pi/3, 100)
    for ax, law, title in zip(axes, ["rtn", "inertial"],
                              ["Направление в текущей RTN", "Фиксированное направление в XY"]):
        color = COL[law]
        ax.plot(np.cos(p), np.sin(p), color=COL["light"], lw=1.4)
        ax.plot(np.cos(ph), np.sin(ph), color=color, lw=3.5, solid_capstyle="round")
        for phi in np.linspace(-np.pi/3, np.pi/3, 5):
            xy = np.array([np.cos(phi), np.sin(phi)])
            vec = np.array([-np.sin(phi), np.cos(phi)]) if law == "rtn" else np.array([0., 1.])
            ax.annotate("", xy=xy+.48*vec, xytext=xy,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7,
                                        mutation_scale=11))
        ax.scatter([0], [0], s=25, c=COL["ink"], zorder=4)
        ax.plot([0, np.cos(np.pi/3)], [0, np.sin(np.pi/3)], "--", color="#ADB5BB", lw=1)
        ax.plot([0, np.cos(np.pi/3)], [0, -np.sin(np.pi/3)], "--", color="#ADB5BB", lw=1)
        ax.plot([0, 1], [0, 0], color="#ADB5BB", lw=.8)
        ax.scatter([1], [0], s=29, c=COL["impulse"], zorder=6)
        ax.annotate("Центр окна\n" + r"$\varphi=0$", (1, 0), xytext=(1.16, -.29),
                    ha="left", va="center", fontsize=11,
                    arrowprops=dict(arrowstyle="-", lw=.7, color=COL["ink"]))
        arc = FancyArrowPatch((.23, -.28), (.23, .28), connectionstyle="arc3,rad=.48",
                              arrowstyle="<->", color=COL["accent"], mutation_scale=10, lw=1.1)
        ax.add_patch(arc)
        ax.text(.43, -.015, r"$\theta$", color=COL["accent"], fontsize=13, ha="center")
        ax.set_title(title, pad=11)
        formula = r"$\mathbf{a}_c=A\,\mathbf{e}_t(t)$" if law == "rtn" else r"$\mathbf{a}_c=A\,\mathbf{e}_Y$"
        ax.text(-1.1, -.93, formula, color=color, fontsize=14)
        ax.set_aspect("equal")
        ax.set_xlim(-1.22, 1.72)
        ax.set_ylim(-1.15, 1.3)
        ax.axis("off")
    fig.savefig(OUT / "circular-directions.pdf", bbox_inches="tight")
    fig.savefig(BUILD / "circular-directions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def responses(theta, exact, linear):
    deg = np.rad2deg(theta)
    full = np.linspace(0, np.pi, 301)
    x = np.rad2deg(full)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.15), constrained_layout=True)
    for ax, j, title, ylabel in [
            (axes[0], 0, "Большая полуось", r"$\Delta a\,/\,(2r_0\Delta V/V_0)$"),
            (axes[1], 1, "Проекция вектора эксцентриситета", r"$\Delta e_x\,/\,(2\Delta V/V_0)$")]:
        ref = 2 * R0 * DV / V0 if j == 0 else 2 * DV / V0
        ax.axhline(1, color=COL["impulse"], lw=1.1, ls=(0, (4, 3)), label="Импульс: первый порядок")
        for law, label in [("rtn", "В текущей ОСК"), ("inertial", "Инерциальное направление")]:
            model = np.array([linear_elements(th, law)[j] for th in full]) / ref
            ax.plot(x, model, color=COL[law], lw=2, label=label)
            ids = np.arange(2, len(theta), 8)
            ax.scatter(deg[ids], exact[law][ids, j]/ref, s=27,
                       facecolor="white", edgecolor=COL[law], linewidth=.9, zorder=5)
        ax.set(title=title, ylabel=ylabel, ylim=(.59, 1.035))
        style_axes(ax, theta=True)
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].text(.035, .20, "Линии — линейная модель\nКружки — нелинейный прогноз",
                 transform=axes[1].transAxes, va="top", fontsize=12)
    axes[1].annotate(r"$2/\pi\approx0{,}637$", xy=(180, 2/np.pi),
                     xytext=(104, .69), color=COL["rtn"], fontsize=13,
                     arrowprops=dict(arrowstyle="-", color=COL["rtn"], lw=.8))
    axes[1].annotate(r"$0{,}75$", xy=(180, .75), xytext=(145, .805),
                     color=COL["inertial"], fontsize=13,
                     arrowprops=dict(arrowstyle="-", color=COL["inertial"], lw=.8))
    fig.savefig(OUT / "circular-response.pdf", bbox_inches="tight")
    fig.savefig(BUILD / "circular-response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def errors(theta, exact, linear):
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.15), constrained_layout=True)
    for law, label in [("rtn", "В текущей ОСК"), ("inertial", "Инерциальное направление")]:
        da = np.abs(exact[law][:, 0] - linear[law][:, 0]) * 1000
        de = np.linalg.norm(exact[law][:, 1:] - linear[law][:, 1:], axis=1) * R0 * 1000
        axes[0].plot(np.rad2deg(theta), da, color=COL[law], lw=2, label=label)
        axes[1].plot(np.rad2deg(theta), de, color=COL[law], lw=2, label=label)
    axes[0].set(title="Ошибка изменения полуоси", ylabel=r"$|\Delta a_{\rm exact}-\Delta a_{\rm lin}|$, м")
    axes[1].set(title="Ошибка вектора эксцентриситета", ylabel=r"$r_0\,\|\Delta\mathbf{e}_{\rm exact}-\Delta\mathbf{e}_{\rm lin}\|$, м")
    for ax in axes:
        style_axes(ax, theta=True)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "circular-linear-check.pdf", bbox_inches="tight")
    fig.savefig(BUILD / "circular-linear-check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def coefficient_table():
    """Dimensionless first-order ratios; angles are reference-circle spans.

    These are calculated from Baranov's formulas, not transcribed from a
    printed table.  They are independent of r0 and D within this model.
    """
    values = []
    for deg in [30, 60, 90, 120, 180]:
        th = np.deg2rad(deg)
        k = float(sinc(th/2))
        ki = .75 + .25*float(sinc(th))
        values.append(dict(theta_deg=deg, rtn_a_ratio=1., rtn_e_ratio=k,
                           inertial_a_ratio=k, inertial_e_ratio=ki))
    return values


def main():
    theta = np.deg2rad(np.linspace(1, 180, 73))
    exact, linear = {}, {}
    for law in ["rtn", "inertial"]:
        exact[law] = np.array([exact_elements(th, law) for th in theta])
        linear[law] = np.array([linear_elements(th, law) for th in theta])
    # Persist the full scan before drawing, so results survive plot failures.
    np.savez_compressed(OUT / "circular-data.npz", theta_rad=theta,
                        exact_rtn=exact["rtn"], exact_inertial=exact["inertial"],
                        linear_rtn=linear["rtn"], linear_inertial=linear["inertial"],
                        mean_matrices=np.array([mean_matrix(th) for th in theta]))
    selected = []
    for deg in [30, 60, 90, 120, 180]:
        th = np.deg2rad(deg)
        entry = dict(theta_deg=deg, duration_s=float(th/N0), duration_min=float(th/N0/60))
        for law in ["rtn", "inertial"]:
            lin = linear_elements(th, law)
            val = exact_elements(th, law)
            chk = exact_elements(th, law, rtol=8e-14, atol=8e-15, steps=160)
            entry[law] = dict(linear_delta_a_m=float(lin[0]*1000),
                              exact_delta_a_m=float(val[0]*1000),
                              linear_e_x=float(lin[1]), exact_e_x=float(val[1]),
                              exact_e_y=float(val[2]),
                              semimajor_linear_error_m=float(abs(val[0]-lin[0])*1000),
                              eccentricity_vector_linear_error_m=float(R0*np.linalg.norm(val[1:]-lin[1:])*1000),
                              tolerance_check_a_m=float(abs(chk[0]-val[0])*1000),
                              tolerance_check_e_m=float(R0*np.linalg.norm(chk[1:]-val[1:])*1000))
        selected.append(entry)
    scaling = []
    for dv_m_s in [.1, .5, 1, 2, 5, 10]:
        for law in ["rtn", "inertial"]:
            th = np.pi
            ex = exact_elements(th, law, dv=dv_m_s/1000)
            li = linear_elements(th, law, dv=dv_m_s/1000)
            scaling.append(dict(delta_v_m_s=dv_m_s, law=law,
                                theta_deg=180,
                                a_error_m=float(abs(ex[0]-li[0])*1000),
                                e_error_m=float(R0*np.linalg.norm(ex[1:]-li[1:])*1000)))
    coefficients = coefficient_table()
    meta = dict(
        version=1, reproduction="python3 8/images/circular_burn.py",
        mu_km3_s2=MU, r0_km=R0, v0_km_s=float(V0), n0_rad_s=float(N0),
        period_s=float(PERIOD), characteristic_velocity_m_s=DV_M_S,
        delta_v_over_v0=float(DV/V0),
        model="exact two-body ODE with constant acceleration magnitude; no mass variation, no J2",
        axes="reference circle XY, counterclockwise; reference centre phi=0 at +X, tangent +Y",
        reference="same circular state before burn; impulse at reference centre t=0",
        theta_definition="reference angular length theta=n0*tau; the exact trajectory angle differs slightly",
        rtol=3e-13, atol=3e-14, max_step="tau/80", method="DOP853",
        matrix_output="[delta_a/r0,e_x,e_y,i_x,i_y]; columns integrals of S,T,W in km/s",
        sinc_definition="sin(x)/x, with sinc(0)=1",
        formula_rtn=dict(a_ratio="1", eccentricity_ratio="sinc(theta/2)"),
        formula_inertial=dict(a_ratio="sinc(theta/2)", eccentricity_ratio="3/4+sinc(theta)/4"),
        exact_impulse=dict(delta_a_m=float(impulse_elements()[0]*1000),
                           e_x=float(impulse_elements()[1])),
        first_order_impulse=dict(delta_a_m=float(2*R0*DV/V0*1000),
                                e_x=float(2*DV/V0)),
        selected=selected, scaling=scaling, analytic_coefficient_table=coefficients,
        max_scan_linear_error_a_m={law:float(np.max(abs(exact[law][:,0]-linear[law][:,0]))*1000)
                                  for law in ["rtn", "inertial"]},
        max_scan_linear_error_e_m={law:float(np.max(np.linalg.norm(exact[law][:,1:]-linear[law][:,1:],axis=1))*R0*1000)
                                  for law in ["rtn", "inertial"]},
        source="Baranov, Spacecraft Manoeuvring in the Vicinity of a Near-Circular Orbit, chapter 6, 6.1.1 and 6.1.3; independently calculated example",
        limitations="first-order formulas apply to small changes near the reference circle; no conclusion on arbitrary finite burns or trajectory optimality")
    (OUT / "circular-results.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n")
    def texnum(x, digits=3):
        return f"{x:.{digits}f}".replace(".", "{,}")
    tex = ["% Generated by circular_burn.py. Constant acceleration; no mass or J2.",
           r"\newcommand{\CircRadius}{7000}",
           r"\newcommand{\CircVelocity}{"+texnum(V0, 6)+"}",
           r"\newcommand{\CircPeriodMinutes}{"+texnum(PERIOD/60, 3)+"}",
           r"\newcommand{\CircDVMps}{1}",
           r"\newcommand{\CircLinearDAMeters}{"+texnum(2*R0*DV/V0*1000, 2)+"}",
           r"\newcommand{\CircLinearEccentricity}{"+texnum(2*DV/V0*1e4, 6)+r"\cdot10^{-4}}",
           r"\newcommand{\CircHalfOrbitMinutes}{"+texnum(PERIOD/120, 3)+"}",
           r"\newcommand{\CircHalfOrbitRTNRatio}{"+texnum(2/np.pi, 6)+"}"]
    for law, name in [("rtn", "RTN"), ("inertial", "Inertial")]:
        row = selected[-1][law]
        tex += [r"\newcommand{\CircHalfOrbit"+name+r"DAMeters}{"+texnum(row["exact_delta_a_m"], 3)+"}",
                r"\newcommand{\CircHalfOrbit"+name+r"Eccentricity}{"+texnum(row["exact_e_x"]*1e4, 6)+r"\cdot10^{-4}}",
                r"\newcommand{\CircHalfOrbit"+name+r"AErrorMeters}{"+texnum(row["semimajor_linear_error_m"], 3)+"}",
                r"\newcommand{\CircHalfOrbit"+name+r"EErrorMeters}{"+texnum(row["eccentricity_vector_linear_error_m"], 3)+"}"]
    (OUT / "circular-values.tex").write_text("\n".join(tex)+"\n")
    geometry()
    responses(theta, exact, linear)
    errors(theta, exact, linear)
    print(json.dumps({"selected": selected, "max_error_a_m":meta["max_scan_linear_error_a_m"],
                      "max_error_e_m":meta["max_scan_linear_error_e_m"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    if "--figures-only" in sys.argv:
        coefficient_table()
        with np.load(OUT / "circular-data.npz") as data:
            theta = data["theta_rad"]
            exact = {law: data["exact_"+law] for law in ["rtn", "inertial"]}
            linear = {law: data["linear_"+law] for law in ["rtn", "inertial"]}
        geometry()
        responses(theta, exact, linear)
        errors(theta, exact, linear)
    else:
        main()
