#!/usr/bin/env python3
"""Reproduce lecture 7 calculations and figures (Python 3, NumPy/SciPy/Matplotlib).

Units in the calculation: km, s, rad; plotted controls are in m/s.
The INTERNAL classical elements are ordered (a, e, i, Omega, omega, M).
The lecture order is (a, e, omega, i, Omega, M), obtained with [0,1,4,2,3,5].
Both orders and the permuted B/Phi are exported explicitly in results.json.
No equinoctial elements are used. Angles in residuals are continued about a
fixed reference.
Run from any directory: python3 7/images/numerics.py
"""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lecture7-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np
from numpy.linalg import norm, solve
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
BUILD.mkdir(exist_ok=True)
MU = 398600.4418
EARTH = 6378.137
PAPER, INK = "#FBF9F3", "#302B28"
BLUE, GREEN, ORANGE, PURPLE = "#187AA5", "#087F78", "#B57714", "#79539C"
PT_FONTS = sorted(Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"))
for font in PT_FONTS:
    font_manager.fontManager.addfont(str(font))
FONT = "PT Serif" if PT_FONTS else "DejaVu Serif"
plt.rcParams.update({
    "font.family": FONT, "font.size": 10,
    "mathtext.fontset": "custom", "mathtext.rm": FONT,
    "mathtext.it": FONT+":italic", "mathtext.bf": FONT+":bold",
    "mathtext.fallback": "cm", "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "pdf.fonttype": 42,
})


def wrap(x):
    return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi


def continue_elements(y, reference):
    ans = np.array(y, dtype=float, copy=True)
    ans[3:] = reference[3:] + wrap(ans[3:] - reference[3:])
    return ans


def kepler(M, e):
    # Keep the revolution count: E and M need not be in [0, 2 pi).
    E = float(M)
    for _ in range(20):
        d = (E-e*np.sin(E)-M)/(1-e*np.cos(E))
        E -= d
        if abs(d) < 3e-15:
            break
    return E


def rotation(O, i, w):
    co, so, ci, si, cw, sw = np.cos(O), np.sin(O), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    return np.array([[co*cw-so*sw*ci, -co*sw-so*cw*ci, so*si],
                     [so*cw+co*sw*ci, -so*sw+co*cw*ci, -co*si],
                     [sw*si, cw*si, ci]])


def elements_to_rv(y):
    a, e, i, O, w, M = y
    E = kepler(M, e)
    r = a*(1-e*np.cos(E))
    Q = rotation(O, i, w)
    rp = a*np.array([np.cos(E)-e, np.sqrt(1-e*e)*np.sin(E), 0.])
    vp = np.sqrt(MU*a)/r*np.array([-np.sin(E), np.sqrt(1-e*e)*np.cos(E), 0.])
    return Q@rp, Q@vp


def rv_to_elements(r, v, reference=None):
    rr = norm(r)
    h = np.cross(r, v); hh = norm(h); H = h/hh
    N = np.cross([0., 0., 1.], h); nn = norm(N)
    ev = np.cross(v, h)/MU-r/rr; e = norm(ev)
    if e < 1e-8 or nn/hh < 1e-8:
        raise ValueError("The classical-element example must avoid circular/equatorial singularities")
    a = 1/(2/rr-np.dot(v, v)/MU)
    i = np.arccos(np.clip(H[2], -1, 1)); O = np.arctan2(N[1], N[0])
    w = np.arctan2(np.dot(np.cross(N, ev), H), np.dot(N, ev))
    f = np.arctan2(np.dot(np.cross(ev, r), H), np.dot(ev, r))
    E = np.arctan2(np.sqrt(1-e*e)*np.sin(f), e+np.cos(f))
    M = E-e*np.sin(E)
    y = np.array([a, e, i, O, w, M])
    return continue_elements(y, reference) if reference is not None else y


def passive(y, dt):
    z = np.array(y, copy=True)
    z[5] += np.sqrt(MU/y[0]**3)*dt
    return z


def rtn(r, v):
    R = r/norm(r); N = np.cross(r, v); N /= norm(N); T = np.cross(N, R)
    # Rows transform inertial vectors into RTN coordinates.
    return np.array([R, T, N])


def gauss_B(y):
    """Instantaneous classical Gauss matrix: dy = B(y) dV_RTN."""
    a, e, i, O, w, M = y
    E = kepler(M, e)
    f = np.arctan2(np.sqrt(1-e*e)*np.sin(E), np.cos(E)-e)
    p = a*(1-e*e); r = a*(1-e*np.cos(E)); h = np.sqrt(MU*p)
    n = np.sqrt(MU/a**3); u = w+f
    B = np.zeros((6, 3))
    B[0, :2] = 2*a*a/h*np.array([e*np.sin(f), p/r])
    B[1, :2] = np.array([p*np.sin(f), (p+r)*np.cos(f)+r*e])/h
    B[2, 2] = r*np.cos(u)/h
    B[3, 2] = r*np.sin(u)/(h*np.sin(i))
    B[4] = [-p*np.cos(f)/(h*e), (p+r)*np.sin(f)/(h*e), -r*np.sin(u)*np.cos(i)/(h*np.sin(i))]
    B[5] = -np.sqrt(1-e*e)*(B[4]+np.cos(i)*B[3])
    B[5, 0] -= 2*r/(n*a*a)
    return B


def classical_phi(y, dt):
    """Exact two-body first variation in classical elements, not a finite map."""
    A = np.eye(6)
    A[5, 0] = -1.5*np.sqrt(MU/y[0]**3)/y[0]*dt
    return A


def burn(y, dV):
    r, v = elements_to_rv(y)
    # The local impulse basis is recomputed at this actual pre-burn state.
    return rv_to_elements(r, v+rtn(r, v).T@dV, y)


def exact_endpoint(y0, U, epochs, tf):
    y = np.array(y0, copy=True); now = 0.
    for t, dV in zip(epochs, np.asarray(U).reshape(-1, 3)):
        y = passive(y, t-now)
        y = burn(y, dV)
        now = t
    return passive(y, tf-now)


def influence(y0, epochs, tf):
    return np.hstack([classical_phi(y0, tf-t)@gauss_B(passive(y0, t)) for t in epochs])


def lambert(rA, rB, duration):
    """p/area-integral Lambert solver, selected short-way, zero-revolution branch.

    The endpoints must be non-collinear, and 0 < theta < pi. This routine is
    intentionally not a general multi-revolution or antipodal Lambert solver.
    """
    ra, rb = norm(rA), norm(rB)
    ex = rA/ra
    normal = np.cross(rA, rB); normal /= norm(normal)
    ey = np.cross(normal, ex)
    theta = np.arccos(np.clip(np.dot(rA, rB)/(ra*rb), -1, 1))
    if not 1e-6 < theta < np.pi-1e-6:
        raise ValueError("The p demonstration excludes collinear endpoints")
    ct, st = np.cos(theta), np.sin(theta)

    def eccentricity(p):
        ax = p/ra-1
        ay = ((p/rb-1)-ax*ct)/st
        return ax, ay

    def time_of_flight(p):
        ax, ay = eccentricity(p)
        return quad(lambda q: (p/(1+ax*np.cos(q)+ay*np.sin(q)))**2/np.sqrt(MU*p),
                    0, theta, epsabs=1e-8, epsrel=2e-12)[0]

    # The lower admissible p is max(-A/B) along the selected short arc.
    def lower_ratio(q):
        A = 1-np.cos(q)-np.tan(theta/2)*np.sin(q)
        B = np.sin(theta-q)/(ra*st)+np.sin(q)/(rb*st)
        return -A/B
    low = -minimize_scalar(lambda q: -lower_ratio(q), bounds=(0, theta), method="bounded").fun
    lo = low*1.0001
    hi = max(ra, rb)*2
    while time_of_flight(hi) > duration:
        hi *= 2
    p = brentq(lambda q: time_of_flight(q)-duration, lo, hi, xtol=2e-9)
    f = 1-rb*(1-ct)/p
    g = ra*rb*st/np.sqrt(MU*p)
    gd = 1-ra*(1-ct)/p
    vA = (rB-f*rA)/g
    vB = (gd*rB-rA)/g
    return dict(p=p, v1=vA, v2=vB, theta=theta, ex=ex, ey=ey,
                eccentricity=eccentricity, tof=time_of_flight, p_lower=low)


def save_slide_variant(fig, name):
    """Save the same scientific plot with labels sized for lecture slides."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(14.5)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_fontsize(14.5)
        ax.tick_params(labelsize=11.5)
        ax.title.set_fontsize(13)
        for label in ax.texts:
            label.set_fontsize(12)
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(12)
    fig.savefig(HERE/f"{name}-slide.pdf", bbox_inches="tight", pad_inches=.035)


def finish(fig, name):
    fig.savefig(HERE/f"{name}.pdf", bbox_inches="tight", pad_inches=.035)
    if name in {"lambert-family", "lambert-branches", "cw-modes", "cw-spatial",
                "relative-elliptic", "rendezvous-orbits", "rendezvous-relative",
                "rendezvous-convergence", "phi-phase"}:
        save_slide_variant(fig, name)
    plt.close(fig)


def grids(axes):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(alpha=.2, linewidth=.5)


def lambert_figures():
    rA = np.array([7000., 0., 0.]); th = np.deg2rad(100)
    rB = 10000*np.array([np.cos(th), np.sin(th), 0.])
    L = lambert(rA, rB, 1800.)
    fig, ax = plt.subplots(1, 2, figsize=(9.1, 3.9), layout="constrained")
    ax[0].add_patch(Circle((0, 0), EARTH, fc="#DFE9E8", ec="#A9CADA", lw=.7))
    ax[0].text(-1500, -2800, "Земля", color="#6D6560", fontsize=9)
    q = np.linspace(0, th, 600)
    for p, color in zip([8000., 10000., 12000.], [BLUE, ORANGE, GREEN]):
        e1, e2 = L["eccentricity"](p); rr = p/(1+e1*np.cos(q)+e2*np.sin(q))
        ax[0].plot(rr*np.cos(q), rr*np.sin(q), c=color, lw=1.8,
                   label=f"{L['tof'](p)/60:.2f} мин")
        ax[1].scatter(p, L["tof"](p)/60, c=color, s=28, zorder=4)
    e1, e2 = L["eccentricity"](L["p"])
    rr = L["p"]/(1+e1*np.cos(q)+e2*np.sin(q))
    ax[0].plot(rr*np.cos(q), rr*np.sin(q), c=INK, lw=1.8,
               ls="--", label="30.00 мин")
    for r in [rA, rB]:
        ax[0].annotate("", xy=r[:2], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=.7))
    ax[0].scatter([0, rA[0], rB[0]], [0, rA[1], rB[1]], s=14, c=INK)
    ax[0].text(7450, -600, "$R_1$"); ax[0].text(-3300, 10100, "$R_2$")
    ax[0].set(xlabel=r"По оси $\mathbf{e}_1$, км", ylabel=r"По оси $\mathbf{e}_2$, км",
              xlim=(-7000, 10500), ylim=(-6700, 11500))
    ax[0].set_aspect("equal"); ax[0].legend(loc="lower right", frameon=False)
    pp = np.linspace(7000, 14000, 250)
    ax[1].plot(pp, [L["tof"](p)/60 for p in pp], c=BLUE, lw=1.7)
    ax[1].axhline(30, c=INK, ls="--", lw=.9)
    ax[1].scatter(L["p"], 30, c=INK, s=25, zorder=5)
    ax[1].annotate(f"$\\tau=30$ мин\n$p={L['p']:.1f}$ км", (L["p"], 30), (10600, 33),
                   arrowprops=dict(arrowstyle="-", color=INK, lw=.7), fontsize=9)
    ax[1].set(xlabel="Фокальный параметр $p$, км", ylabel="Время перелёта, мин", ylim=(20, 45))
    ax[1].tick_params(axis="x", labelsize=8); grids(ax)
    finish(fig, "lambert-family")

    # Only elliptical p allow an additional complete revolution.
    def eminus(p): return norm(L["eccentricity"](p))-1
    pmin = L["p_lower"]; pmax = brentq(eminus, 10000, 18000)
    pp = np.linspace(pmin+200, pmax-250, 700)
    T0 = np.array([L["tof"](p) for p in pp])
    ee = np.array([norm(L["eccentricity"](p)) for p in pp])
    periods = 2*np.pi*np.sqrt((pp/(1-ee*ee))**3/MU)
    one = lambda p: L["tof"](p)+2*np.pi*np.sqrt((p/(1-norm(L["eccentricity"](p))**2))**3/MU)
    minimum = minimize_scalar(one, bounds=(pmin+300, pmax-300), method="bounded")
    roots = [brentq(lambda p: one(p)-12000, pmin+200, minimum.x),
             brentq(lambda p: one(p)-12000, minimum.x, pmax-250)]
    fig, ax = plt.subplots(figsize=(7.9, 3.6), layout="constrained")
    ax.plot(pp, T0/60, c=BLUE, lw=1.7, label="Без полного оборота: $T_0(p)$")
    ax.plot(pp, (T0+periods)/60, c=PURPLE, lw=1.7, label="Ещё один оборот: $T_1(p)$")
    ax.axhline(200, c=INK, lw=.9, ls="--", label="Заданное время: 200 мин")
    ax.scatter(roots, [200, 200], c=PURPLE, s=30, zorder=5)
    ax.scatter(minimum.x, minimum.fun/60, c=PURPLE, s=22)
    ax.annotate(f"Минимум $T_1$: {minimum.fun/60:.1f} мин", (minimum.x, minimum.fun/60),
                (8500, 125), fontsize=9, arrowprops=dict(arrowstyle="-", color=PURPLE, lw=.8))
    ax.set(xlabel="$p$, км", ylabel="Время перелёта, мин", ylim=(0, 400), xlim=(pmin+200, pmax-250))
    ax.legend(frameon=False, loc="upper center", fontsize=8); grids(ax)
    ax.text(.02, .02, "Кривые без отсева пересечений с Землёй", transform=ax.transAxes,
            fontsize=8, color="#6D6560", va="bottom")
    finish(fig, "lambert-branches")
    return dict(p_root_km=L["p"], p_elliptic_limits_km=[pmin, pmax],
                one_revolution_minimum_s=minimum.fun, one_revolution_roots_at_12000s_km=roots,
                branch_scope="Fixed transfer plane/direction, theta=100deg; collision constraints not imposed")


def cw_rhs(n):
    def rhs(t, s):
        x, y, z, dx, dy, dz = s
        return [dx, dy, dz, 3*n*n*x+2*n*dy, -2*n*dx, -n*n*z]
    return rhs


def cw_figures():
    n = np.sqrt(MU/7000**3); P = 2*np.pi/n
    t = np.linspace(0, 2*P, 1000)
    bound = solve_ivp(cw_rhs(n), [0, 2*P], [.1, 0, 0, 0, -.2*n, 0], t_eval=t,
                      rtol=2e-12, atol=1e-14).y.T
    drift = solve_ivp(cw_rhs(n), [0, 2*P], [.1, 0, 0, 0, -.17*n, 0], t_eval=t,
                      rtol=2e-12, atol=1e-14).y.T
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45), layout="constrained")
    for ax, s, title, color in zip(axes, [bound, drift], ["Замкнутое движение", "Движение с дрейфом"], [BLUE, ORANGE]):
        ax.plot(s[:, 1]*1000, s[:, 0]*1000, c=color, lw=1.7)
        ax.scatter([0], [100], c=INK, s=18, zorder=5)
        ax.set(xlabel="$y$, вдоль орбиты, м", ylabel="$x$, радиально, м", title=title)
        ax.set_aspect("equal", adjustable="datalim")
    grids(axes); finish(fig, "cw-modes")
    t = np.linspace(0, P, 700)
    x = 100*np.cos(n*t); y = -200*np.sin(n*t); z = 120*np.sin(n*t+.6)
    fig = plt.figure(figsize=(8.4, 3.8), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    ax.set_facecolor(PAPER); ax.plot(y, x, z, c=BLUE, lw=1.8)
    ax.set(xlabel="$y$, м", ylabel="$x$, м", zlabel="$z$, м")
    ax.view_init(23, -60); ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(122)
    for s, label, col in zip([x, y, z], ["$x$", "$y$", "$z$"], [BLUE, GREEN, PURPLE]):
        ax2.plot(t/P, s, c=col, label=label, lw=1.6)
    ax2.set(xlabel="Время, витки", ylabel="Относительные координаты, м")
    ax2.yaxis.set_label_position("right")
    ax2.legend(frameon=False, ncol=3); grids(ax2)
    finish(fig, "cw-spatial")


def elliptic_figure():
    a, e = 10000., .2; n = np.sqrt(MU/a**3); P = 2*np.pi/n
    R0 = np.array([a*(1-e), 0., 0.]); V0 = np.array([0., np.sqrt(MU*(1+e)/(a*(1-e))), 0.])
    h = norm(np.cross(R0, V0)); rho0 = np.array([0., -.1, .01]); drho0 = np.zeros(3)
    r0 = R0+rho0; v0 = V0+np.cross([0, 0, h/norm(R0)**2], rho0)
    def rhs(t, y):
        return np.r_[y[3:6], -MU*y[:3]/norm(y[:3])**3, y[9:12], -MU*y[6:9]/norm(y[6:9])**3]
    opts = dict(rtol=2e-13, atol=1e-14, method="DOP853")
    exact = solve_ivp(rhs, [0, P], np.r_[R0, V0, r0, v0], dense_output=True, **opts)
    ts = np.linspace(0, P, 1301)
    rel = []
    for t in ts:
        y = exact.sol(t); rel.append(rtn(y[:3], y[3:6])@(y[6:9]-y[:3]))
    rel = np.array(rel)*1000
    def th(t, y):
        x, yy, z, dx, dy, dz = y; s = exact.sol(t); R, V = s[:3], s[3:6]
        rr = norm(R); omega = h/rr**2; odot = -2*omega*np.dot(R, V)/rr**2; k = MU/rr**3
        return [dx, dy, dz, (omega**2+2*k)*x+2*omega*dy+odot*yy,
                (omega**2-k)*yy-2*omega*dx-odot*x, -k*z]
    TH = solve_ivp(th, [0, P], np.r_[rho0, drho0], t_eval=ts, **opts).y[:3].T*1000
    CW = solve_ivp(cw_rhs(n), [0, P], np.r_[rho0, drho0], t_eval=ts, **opts).y[:3].T*1000
    eth, ecw = norm(TH-rel, axis=1), norm(CW-rel, axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(8.7, 3.6), layout="constrained")
    ax[0].plot(rel[:, 1], rel[:, 0], c=INK, lw=2.5, label="Два точных движения")
    ax[0].plot(TH[:, 1], TH[:, 0], c=GREEN, ls="--", lw=1.6, label="TH")
    ax[0].scatter([-100], [0], c=ORANGE, s=38, label="CW: неподвижная точка", zorder=6)
    ax[0].set(xlabel="$y$, м", ylabel="$x$, м", xlim=(-170, -90), ylim=(-30, 30))
    ax[0].legend(loc="lower left", fontsize=7, frameon=False); grids(ax[0])
    for err, col, label in [(ecw, ORANGE, "CW"), (eth, GREEN, "TH")]:
        ax[1].semilogy(ts[1:]/P, np.maximum(err[1:], 1e-8), c=col, lw=1.6, label=label)
    ax[1].set(xlabel="Время, витки", ylabel="Ошибка положения, м", ylim=(1e-5, 200))
    ax[1].legend(frameon=False); grids(ax[1])
    finish(fig, "relative-elliptic")
    return dict(a_km=a, e=e, period_s=P, initial_rtn_m=(rho0*1000).tolist(),
                initial_rtn_derivative_mps=(drho0*1000).tolist(),
                cw_max_position_error_m=ecw.max(), elliptic_max_position_error_m=eth.max(),
                cw_final_position_error_m=ecw[-1], elliptic_final_position_error_m=eth[-1])


def solve_example():
    target = np.array([7000., .01, *np.deg2rad([45., 20., 30., 40.])])
    deviations = np.array([-.3, 2e-5, 1e-4, -1.5e-4, 1e-4, -4e-4])
    chaser = target+deviations
    tf = 1200.; epochs = [0., tf]
    yf = passive(target, tf); yp = passive(chaser, tf)
    A = influence(chaser, epochs, tf)
    U = solve(A, yf-yp).reshape(2, 3)
    linear = U.copy()
    r0, v0 = elements_to_rv(chaser); rf, vf = elements_to_rv(yf)
    L = lambert(r0, rf, tf)
    Ulam = np.array([rtn(r0, v0)@(L["v1"]-v0), rtn(rf, L["v2"])@(vf-L["v2"])])
    scale = np.array([1., 7000., 7000., 7000., 7000., 7000.])
    history = []

    def residual(U):
        return continue_elements(exact_endpoint(chaser, U, epochs, tf), yf)-yf

    def metrics(U):
        ye = exact_endpoint(chaser, U, epochs, tf)
        re, ve = elements_to_rv(ye)
        return dict(position_m=norm(re-rf)*1000, velocity_mps=norm(ve-vf)*1000,
                    element_residual=residual(U).tolist(),
                    cost_mps=float(np.sum(norm(U, axis=1))*1000),
                    impulse_norms_mps=(norm(U, axis=1)*1000).tolist(), controls_rtn_mps=(U*1000).tolist())

    # Newton corrections use the current exact mapping and freshly recomputed RTN bases.
    for iteration in range(4):
        history.append(dict(iteration=iteration, **metrics(U)))
        if history[-1]["position_m"] < 2e-7 and history[-1]["velocity_mps"] < 2e-10:
            break
        step = 1e-6  # km/s = 1 mm/s, centered difference
        flat = U.ravel(); J = np.empty((6, 6))
        for k in range(6):
            d = np.zeros(6); d[k] = step
            J[:, k] = (residual((flat+d).reshape(2, 3))-residual((flat-d).reshape(2, 3)))/(2*step)
        dU = solve(scale[:, None]*J, -scale*residual(U)).reshape(2, 3)
        U += dU

    # Independent Cartesian numerical integration validates the analytic propagator.
    def rhs(t, s): return np.r_[s[3:], -MU*s[:3]/norm(s[:3])**3]
    checked = {}
    for label, V in [("linear", linear), ("lambert", Ulam), ("corrected", U)]:
        startv = v0+rtn(r0, v0).T@V[0]
        sol = solve_ivp(rhs, [0, tf], np.r_[r0, startv], method="DOP853", rtol=2e-13, atol=1e-14)
        rr, vv = sol.y[:3, -1], sol.y[3:, -1]
        vv += rtn(rr, vv).T@V[1]
        ya = exact_endpoint(chaser, V, epochs, tf); ra, va = elements_to_rv(ya)
        checked[label] = dict(analytic_vs_integrator_position_m=norm(rr-ra)*1000,
                              analytic_vs_integrator_velocity_mps=norm(vv-va)*1000,
                              integrated_terminal_position_m=norm(rr-rf)*1000,
                              integrated_terminal_velocity_mps=norm(vv-vf)*1000)
    # Check every entry of B against the actual finite impulse mapping.
    d = 1e-7; Bnum = np.empty((6, 3))
    for k in range(3):
        dv = np.zeros(3); dv[k] = d
        Bnum[:, k] = (burn(chaser, dv)-burn(chaser, -dv))/(2*d)
    B = gauss_B(chaser)
    error_B = np.max(np.abs(Bnum-B)/(1+np.abs(B)))
    # Check Phi independently by finite differences of passive element propagation.
    d = 1e-3; up = chaser.copy(); um = chaser.copy(); up[0] += d; um[0] -= d
    phi_da = (passive(up, tf)-passive(um, tf))/(2*d)
    error_phi = np.max(np.abs(phi_da-classical_phi(chaser, tf)[:, 0]))
    rtarget0, _ = elements_to_rv(target)
    data = dict(mu_km3_s2=MU, duration_s=tf, epochs_s=epochs,
                element_order=["a_km", "e", "i_rad", "Omega_rad", "omega_rad", "M_rad"],
                lecture_element_order=["a_km", "e", "omega_rad", "i_rad", "Omega_rad", "M_rad"],
                lecture_row_permutation=[0, 1, 4, 2, 3, 5],
                target_initial_elements_lecture=target[[0, 1, 4, 2, 3, 5]].tolist(),
                chaser_initial_elements_lecture=chaser[[0, 1, 4, 2, 3, 5]].tolist(),
                gauss_B_initial_lecture=B[[0, 1, 4, 2, 3, 5]].tolist(),
                classical_phi_lecture=classical_phi(chaser,tf)[np.ix_([0,1,4,2,3,5],[0,1,4,2,3,5])].tolist(),
                target_initial_elements=target.tolist(), chaser_initial_elements=chaser.tolist(),
                initial_deviations=deviations.tolist(), initial_distance_m=norm(r0-rtarget0)*1000,
                linear=metrics(linear), lambert=metrics(Ulam), corrected=metrics(U),
                lambert_p_km=L["p"], lambert_transfer_angle_deg=np.rad2deg(L["theta"]),
                convergence=history, validation=checked,
                gauss_matrix_centered_difference_scaled_max_error=error_B,
                phi_da_centered_difference_max_error=error_phi,
                scaled_influence_condition_number=np.linalg.cond(scale[:, None]*A))
    assert error_B < 2e-6, error_B
    assert data["corrected"]["position_m"] < 1e-4
    assert data["lambert"]["position_m"] < 1e-4
    return target, chaser, linear, Ulam, U, data


def trajectory(y0, first_impulse, times):
    y = burn(y0, first_impulse)
    return np.array([np.r_[*elements_to_rv(passive(y, t))] for t in times])


def example_figures(target, chaser, linear, Ulam, corrected, data):
    tf = data["duration_s"]; times = np.linspace(0, tf, 601)
    goal = np.array([np.r_[*elements_to_rv(passive(target, t))] for t in times])
    traj = [("Линейное", BLUE, trajectory(chaser, linear[0], times)),
            ("Ламберт", GREEN, trajectory(chaser, Ulam[0], times)),
            ("После уточнения", PURPLE, trajectory(chaser, corrected[0], times))]
    # Project the inertial trajectories onto the target's fixed initial orbital plane.
    R0, V0 = elements_to_rv(target); C = rtn(R0, V0)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6), layout="constrained")
    goal_plane = goal[:, :3]@C.T
    ax[0].add_patch(Circle((0, 0), EARTH, fc="#DFE9E8", ec="#A9CADA", lw=.6))
    ax[0].plot(goal_plane[:, 0], goal_plane[:, 1], c=INK, lw=2, label="Цель")
    for label, col, X in traj[:2]:
        q = X[:, :3]@C.T
        ax[0].plot(q[:, 0], q[:, 1], c=col, lw=1.2, ls="--", label=label)
    ax[0].set(xlabel="По начальному радиусу, км", ylabel="Поперёк начального радиуса, км"); ax[0].set_aspect("equal")
    ax[0].legend(frameon=False, fontsize=8)
    rel0 = rtn(R0, V0)@(elements_to_rv(chaser)[0]-R0)*1000
    ax[1].scatter(0, 0, c=INK, s=25, label="Цель")
    ax[1].scatter(rel0[1], rel0[0], c=BLUE, s=25, label="КА в начале")
    ax[1].annotate(f"$\\|\\rho_i\\|={data['initial_distance_m']/1000:.2f}$ км", (rel0[1]/2, rel0[0]/2), fontsize=10)
    ax[1].plot([0, rel0[1]], [0, rel0[0]], c="#A9CADA", lw=1.2)
    ax[1].set(xlabel="$y_i$, м", ylabel="$x_i$, м", title="Проекция начального разнесения"); grids(ax[1]); ax[1].legend(frameon=False)
    finish(fig, "rendezvous-orbits")

    rels = []
    for label, col, X in traj:
        rho = np.array([rtn(g[:3], g[3:])@(x[:3]-g[:3]) for x, g in zip(X, goal)])*1000
        rels.append((label, col, rho))
    fig, ax = plt.subplots(1, 2, figsize=(8.7, 3.6), layout="constrained")
    for label, col, rho in rels:
        ax[0].plot(rho[:, 1], rho[:, 0], c=col, lw=1.4, ls="--" if label=="Ламберт" else "-", label=label)
        ax[1].plot(times/60, rho[:, 2], c=col, lw=1.4, ls="--" if label=="Ламберт" else "-")
    ax[0].scatter([0], [0], c=INK, s=23, zorder=5)
    ax[0].set(xlabel="$y$, вдоль орбиты, м", ylabel="$x$, радиально, м")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(xlabel="Время, мин", ylabel="$z$, нормально к плоскости, м")
    grids(ax); finish(fig, "rendezvous-relative")

    fig, ax = plt.subplots(1, 2, figsize=(8.1, 3.25), layout="constrained")
    hist = data["convergence"]; kk = [x["iteration"] for x in hist]
    # Raw residuals remain untouched in JSON. Display only accuracy supported by
    # the independent Cartesian integration; a downward triangle denotes a bound.
    for axis, field, floor, color, bound in [
            (ax[0], "position_m", 1e-6, BLUE, "Ниже $10^{-6}$ м"),
            (ax[1], "velocity_mps", 1e-9, GREEN, "Ниже $10^{-9}$ м/с")]:
        raw = np.array([x[field] for x in hist]); clipped = raw < floor
        shown = np.maximum(raw, floor)
        axis.semilogy(kk, shown, c=color, lw=1.5)
        axis.scatter(np.asarray(kk)[~clipped], shown[~clipped], c=color, s=25, marker="o", zorder=5)
        axis.scatter(np.asarray(kk)[clipped], shown[clipped], c=color, s=40, marker="v", zorder=5, label=bound)
        axis.axhline(floor, color="#A9CADA", ls="--", lw=.8)
        axis.set_ylim(bottom=floor*.4)
        axis.legend(loc="upper right", frameon=False, fontsize=8)
    ax[0].set(xlabel="Число поправок", ylabel="Невязка положения, м", xticks=kk)
    ax[1].set(xlabel="Число поправок", ylabel="Невязка скорости, м/с", xticks=kk)
    grids(ax); finish(fig, "rendezvous-convergence")

    # Fixed-final-time cost map, each point solves a different 6x6 linear system.
    first = np.linspace(0, 1000, 81); second = np.linspace(200, tf, 81)
    cost = np.full((len(first), len(second)), np.nan)
    target_f = passive(target, tf); baseline = passive(chaser, tf)
    for j, t1 in enumerate(first):
        for k, t2 in enumerate(second):
            if t2-t1 >= 160:
                A = influence(chaser, [t1, t2], tf)
                dV = solve(A, target_f-baseline).reshape(2, 3)
                cost[j, k] = np.sum(norm(dV, axis=1))*1000
    fig, ax = plt.subplots(figsize=(7.1, 3.55), layout="constrained")
    X, Y = np.meshgrid(first/60, second/60, indexing="ij")
    levels = np.linspace(np.nanmin(cost), min(np.nanmax(cost), 6*np.nanmin(cost)), 22)
    m = ax.contourf(X, Y, cost, levels=levels, cmap="YlGnBu", extend="max")
    ax.scatter([0], [tf/60], c=ORANGE, s=55, marker="*", edgecolors=INK, linewidths=.5, zorder=6)
    ax.set(xlabel="Первый момент $t_1$, мин", ylabel="Второй момент $t_2$, мин")
    cb = fig.colorbar(m, ax=ax); cb.set_label("Сумма модулей импульсов, м/с")
    grids(ax); finish(fig, "rendezvous-times")
    data["time_scan"] = dict(scope="Two RTN impulses, fixed tf=1200s; linear classical model; t2-t1>=160s",
                            first_epoch_s=first.tolist(), second_epoch_s=second.tolist(),
                            costs_mps=[[None if not np.isfinite(q) else float(q) for q in row] for row in cost],
                            grid_min_cost_mps=float(np.nanmin(cost)))


def phase_figure():
    a, da = 7000., 100.
    n = np.sqrt(MU/a**3); dn = np.sqrt(MU/(a+da)**3)-n
    t = np.linspace(0, 24*3600, 500)
    exact = dn*t
    linear = -1.5*n/a*da*t
    fig, ax = plt.subplots(1, 2, figsize=(8.3, 3.3), layout="constrained")
    ax[0].plot(t/3600, np.rad2deg(exact), c=INK, lw=2, label="Кеплеров закон движения")
    ax[0].plot(t/3600, np.rad2deg(linear), c=BLUE, lw=1.5, ls="--", label="Линейное приближение")
    ax[0].set(xlabel="Время, ч", ylabel="$\\delta M$, градусы"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(t/3600, np.rad2deg(linear-exact), c=ORANGE, lw=1.7)
    ax[1].set(xlabel="Время, ч", ylabel="Ошибка линейной фазы, градусы")
    grids(ax); finish(fig, "phi-phase")
    return dict(a_km=a, delta_a_km=da, span_h=24., exact_delta_M_deg=float(np.rad2deg(exact[-1])),
                linear_delta_M_deg=float(np.rad2deg(linear[-1])), error_deg=float(np.rad2deg(linear[-1]-exact[-1])))


def write_values(data):
    def number(value, floor=None):
        if floor is not None and value < floor:
            exponent = int(np.round(np.log10(floor)))
            return f"$<10^{{{exponent}}}$"
        if 0 < abs(value) < 1e-3:
            exponent = int(np.floor(np.log10(abs(value))))
            mantissa = value/10**exponent
            return f"${mantissa:.3g}\\cdot10^{{{exponent}}}$"
        return f"{value:.4g}"

    vals = {
        "ExampleDuration": f"{data['duration_s']:.0f}\\,\\text{{с}}",
        "ExampleInitialDistance": f"{data['initial_distance_m']/1000:.3f}\\,\\text{{км}}",
        "ExampleLinearCost": f"{data['linear']['cost_mps']:.4f}\\,\\text{{м/с}}",
        "ExampleLambertCost": f"{data['lambert']['cost_mps']:.4f}\\,\\text{{м/с}}",
        "ExampleLinearMiss": f"{data['linear']['position_m']:.3f}\\,\\text{{м}}",
        "ExampleLambertMiss": "<10^{-6}\\,\\text{м}",
        "ExampleCorrectedMiss": "<10^{-6}\\,\\text{м}",
        "ExampleLambertP": f"{data['lambert_p_km']:.3f}\\,\\text{{км}}",
        "ExampleTransferAngle": f"{data['lambert_transfer_angle_deg']:.3f}^\\circ",
    }
    lines = ["% Generated by numerics.py. Values use km, s and classical elements."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in vals.items()]
    rows = []
    for name, label in [("linear", "Линейное"), ("lambert", "Ламберт"), ("corrected", "Уточнённое")]:
        q = data[name]
        rows.append(f"{label} & {q['impulse_norms_mps'][0]:.4f} & {q['impulse_norms_mps'][1]:.4f} & {q['cost_mps']:.4f} & {number(q['position_m'],1e-6)} \\\\")
    lines += ["% Columns: method, |dV1| m/s, |dV2| m/s, sum m/s, position miss m.",
              "\\newcommand{\\ExampleControlRows}{%", *rows, "}"]
    lines += ["% Initial target elements, followed by chaser-minus-target deviations.",
              "\\newcommand{\\ExampleInitialData}{%",
              "a_0 &= 7000\\,\\text{км}, & e_0 &= 0.01, & i_0 &=45^\\circ,\\\\",
              "\\Omega_0 &=20^\\circ, & \\omega_0 &=30^\\circ, & M_0 &=40^\\circ,\\\\",
              "\\delta a_0 &=-0.3\\,\\text{км}, & \\delta e_0 &=2\\cdot10^{-5}, & \\delta i_0 &=10^{-4}\\,\\text{рад},\\\\",
              "\\delta\\Omega_0 &=-1.5\\cdot10^{-4}\\,\\text{рад}, & \\delta\\omega_0 &=10^{-4}\\,\\text{рад}, & \\delta M_0 &=-4\\cdot10^{-4}\\,\\text{рад}.",
              "}"]
    lines += ["% Columns: method, sum dV (m/s), position miss (m), velocity miss (m/s).",
              "\\newcommand{\\ExampleComparisonRows}{%"]
    for name, label in [("linear", "Линейное"), ("lambert", "Ламберт"), ("corrected", "Уточнённое")]:
        q = data[name]
        # Report below-resolution residuals as bounds, rather than physical nanometre accuracy.
        lines.append(f"{label} & {q['cost_mps']:.5f} & {number(q['position_m'],1e-6)} & {number(q['velocity_mps'],1e-9)} \\\\")
    lines.append("}")
    lines += ["% Columns: iteration, position miss (m), velocity miss (m/s), sum dV (m/s).",
              "\\newcommand{\\ExampleCorrectionRows}{%"]
    for q in data["convergence"]:
        lines.append(f"{q['iteration']} & {number(q['position_m'],1e-6)} & {number(q['velocity_mps'],1e-9)} & {q['cost_mps']:.5f} \\\\")
    lines.append("}")
    # Explicit component rows, for a table with R/T/N columns.
    for name, macro in [("linear", "Linear"), ("lambert", "Lambert"), ("corrected", "Corrected")]:
        U = data[name]["controls_rtn_mps"]
        lines.append(f"\\newcommand{{\\Example{macro}Components}}{{%")
        lines += [f"$\\Delta V_{{{j+1}}}$ & " + " & ".join(f"{v:.4f}" for v in row) + " \\\\" for j, row in enumerate(U)]
        lines.append("}")
    (HERE/"example-values.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    data = {"lambert_geometry": lambert_figures()}
    cw_figures()
    data["elliptic_comparison"] = elliptic_figure()
    target, chaser, linear, Ulam, corrected, example = solve_example()
    example_figures(target, chaser, linear, Ulam, corrected, example)
    data["rendezvous"] = example
    data["phase_linearization"] = phase_figure()
    write_values(example)
    (BUILD/"results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: example[k] for k in ["initial_distance_m", "linear", "lambert", "corrected", "convergence", "validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
