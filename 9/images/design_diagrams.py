#!/usr/bin/env python3
"""Five compact explanatory diagrams for the lecture-9 notes.

Run from the course root: python3 9/images/design_diagrams.py
All figures are exactly 170 x 72 mm; text is at least 9 pt at that width.
The fixed-direction plot reads example-results.json and uses the same exact
Gaussian disk integral as avoidance_example.py. No example data are changed.
Parameters, sampled curves, and numerical checks are saved to design-data.json.
"""
from pathlib import Path
import hashlib
import json
import os

BASE = Path(__file__).resolve().parent
BUILD = BASE.parent / 'build'
BUILD.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BUILD / 'mplconfig'))

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon
from matplotlib.text import Text
from avoidance_example import DiskProbability

BLUE, GREEN, ORANGE, PURPLE = '#187AA5', '#087F78', '#B57714', '#79539C'
INK, GRAY = '#25323F', '#687582'
MM = 1 / 25.4
FIGSIZE = (170 * MM, 72 * MM)
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 9.3,
    'axes.labelsize': 9.3, 'axes.titlesize': 9.7,
    'legend.fontsize': 9.0, 'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': .16, 'grid.linewidth': .5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'dejavuserif',
})


def arrow(ax, start, end, color=INK, lw=1.3, mutation=9, **kwargs):
    p = FancyArrowPatch(start, end, arrowstyle='-|>', color=color,
                        lw=lw, mutation_scale=mutation, **kwargs)
    ax.add_patch(p)
    return p


def plain_axes(ax):
    ax.set_aspect('equal')
    ax.set_axis_off()


def save(fig, name):
    """Preserve the declared physical size, rather than cropping the page."""
    fig.canvas.draw()
    fonts = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text()]
    if min(fonts) < 9 - 1e-9:
        raise ValueError(f'{name}: text smaller than 9 pt')
    fig.savefig(BASE / f'{name}.pdf')
    fig.savefig(BASE / f'{name}.png', dpi=200)
    plt.close(fig)
    return {'file_base': name, 'width_mm': 170, 'height_mm': 72,
            'minimum_font_pt': min(fonts)}


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def cylinder():
    """Oblique projection of a genuine circular cylinder, plus its section."""
    radius, half_length = 1., 1.6
    # Rows project the (z1,z2,s) frame into drawing coordinates.
    camera = np.array([[.72, -.23, .87], [-.20, .88, .35]])
    point = np.array([.42, .35])
    theta = np.linspace(0, 2 * np.pi, 300)
    ring = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([.025, .16, .52, .72])
    plain_axes(ax)
    ax.set_title('а) Цилиндр опасных положений', pad=8)
    caps = [(ring + [0, 0, z]) @ camera.T for z in [-half_length, half_length]]
    ax.add_patch(Polygon(np.vstack([caps[0], caps[1][::-1]]),
                         facecolor=BLUE, edgecolor='none', alpha=.055))
    # The side silhouette joins the two tangency points in this projection.
    zshift = camera[:, 2]
    normal = np.array([-zshift[1], zshift[0]])
    k = int(np.argmax(caps[0] @ normal))
    for j in [k, (k + len(theta)//2) % len(theta)]:
        ax.plot([caps[0][j, 0], caps[1][j, 0]],
                [caps[0][j, 1], caps[1][j, 1]], color=BLUE, lw=1.15)
    for i, cap in enumerate(caps):
        ax.fill(cap[:, 0], cap[:, 1], color=BLUE, alpha=.07)
        ax.plot(cap[:, 0], cap[:, 1], color=BLUE, lw=1.05,
                ls='--' if i == 0 else '-')
    middle = ring @ camera.T
    ax.fill(middle[:, 0], middle[:, 1], color=ORANGE, alpha=.22)
    ax.plot(middle[:, 0], middle[:, 1], color=ORANGE, lw=1.1)
    # A line of relative motion lies inside the cylinder at every s.
    traj = np.array([[*point, -2.0], [*point, 2.0]]) @ camera.T
    arrow(ax, traj[0], traj[1], GREEN, lw=1.6, mutation=11)
    cross = camera @ np.r_[point, 0.]
    ax.plot(*cross, 'o', color=GREEN, ms=4)
    ax.text(.68, 1.07, r'$\mathbf{v}_{\rm rel}\parallel s$', color=GREEN)
    for j, label, length, offset in [
            (0, '$z_1$', 1.65, (.05, -.08)),
            (1, '$z_2$', 1.55, (-.08, .05)),
            (2, '$s$', 2.35, (.06, -.03))]:
        end = camera[:, j] * length
        arrow(ax, (0, 0), end, GRAY, lw=.9)
        ax.text(*(end + offset), label, color=INK)
    rc_end = camera @ np.array([-.68, -.733212, 0.])
    arrow(ax, (0, 0), rc_end, ORANGE, lw=1.0, mutation=8)
    ax.text(-.31, -.57, '$R_c$', color=ORANGE)
    ax.text(-.12, -1.07, '$s=0$', color=ORANGE, ha='center')
    ax.set_xlim(-2.35, 2.4)
    ax.set_ylim(-1.5, 1.65)

    ax = fig.add_axes([.62, .17, .32, .70])
    plain_axes(ax)
    ax.set_title('б) Поперечное сечение', pad=8)
    ax.add_patch(Circle((0, 0), radius, facecolor=ORANGE,
                        edgecolor=ORANGE, alpha=.25, lw=1.1))
    arrow(ax, (-1.18, 0), (1.37, 0), GRAY, lw=.8)
    arrow(ax, (0, -1.14), (0, 1.36), GRAY, lw=.8)
    ax.text(1.35, -.17, '$z_1$')
    ax.text(.09, 1.32, '$z_2$')
    arrow(ax, (0, 0), point, GREEN, lw=1.5)
    ax.plot(*point, 'o', color=GREEN, ms=4)
    ax.plot([point[0], point[0], 0], [0, point[1], point[1]],
            ls=':', color=GREEN, lw=.9)
    ax.text(.55, .42, r'$\mathbf{z}$', color=GREEN)
    arrow(ax, (0, 0), (-.7071, -.7071), ORANGE, lw=1.1)
    ax.text(-.77, -.45, '$R_c$', color=ORANGE)
    ax.set_xlim(-1.28, 1.48)
    ax.set_ylim(-1.2, 1.5)
    fig.text(.30, .055, r'$-\infty<s<\infty$', ha='center')
    fig.text(.78, .055, r'$z_1^2+z_2^2\leq R_c^2$', ha='center', color=ORANGE)
    out = save(fig, 'concept-cylinder')
    out.update({'type': 'conceptual; cylinder drawn over finite s extent',
                'radius_arbitrary_units': radius, 'drawn_s_half_length': half_length,
                'line_transverse_coordinates': point.tolist(),
                'oblique_projection': camera.tolist(),
                'collision_check_z_norm_less_than_radius': bool(np.linalg.norm(point) < radius)})
    return out


def principal_axes():
    """Only rotate coordinates: the collision disk must remain a disk."""
    sig = np.array([48., 22.])
    angle = np.deg2rad(32.)
    Q = rotation(angle)
    beta = np.array([48., 29.])
    b = Q @ beta
    radius = 13.
    cov = Q @ np.diag(sig**2) @ Q.T
    fig = plt.figure(figsize=FIGSIZE)
    axs = [fig.add_axes([.075, .255, .37, .62]),
           fig.add_axes([.62, .255, .34, .62])]
    bounds = [(-60, 125, -50, 112), (-60, 125, -50, 112)]
    for i, ax in enumerate(axs):
        center = b if i == 0 else beta
        ang = np.rad2deg(angle) if i == 0 else 0.
        for level in [2, 1]:
            ax.add_patch(Ellipse(center, 2 * level * sig[0], 2 * level * sig[1],
                                 angle=ang, facecolor=BLUE,
                                 edgecolor=BLUE, alpha=.08 if level == 2 else .16,
                                 lw=.9))
        ax.add_patch(Circle((0, 0), radius, facecolor=ORANGE,
                            edgecolor=ORANGE, alpha=.65, lw=.9))
        arrow(ax, (0, 0), center, GREEN, lw=1.6)
        ax.plot(*center, 'o', color=GREEN, ms=4)
        ax.axhline(0, color=GRAY, lw=.7)
        ax.axvline(0, color=GRAY, lw=.7)
        lo, hi, bot, top = bounds[i]
        ax.set_xlim(lo, hi)
        ax.set_ylim(bot, top)
        ax.set_aspect('equal')
        ax.set_xticks([-40, 0, 40, 80, 120])
        ax.set_yticks([-40, 0, 40, 80])
        ax.tick_params(length=2.5, pad=2)
    ax = axs[0]
    ax.set_title('а) Исходные оси $z_1,z_2$', pad=8)
    ax.set_xlabel('$z_1$, м', labelpad=2)
    ax.set_ylabel('$z_2$, м', labelpad=2)
    for j in range(2):
        u = Q[:, j]
        ax.plot([-60*u[0], 115*u[0]], [-60*u[1], 115*u[1]],
                color=PURPLE, lw=.9, ls='--')
    ax.text(79, 47, 'Главные оси', color=PURPLE, rotation=32,
            ha='center', va='bottom', fontsize=9)
    ax.text(28, 43, r'$\mathbf{b}$', color=GREEN)
    ax.text(-29, -27, '$R_c$', color=ORANGE)
    ax = axs[1]
    ax.set_title('б) После поворота', pad=8)
    ax.set_xlabel('$x$, м', labelpad=2)
    ax.set_ylabel('$y$, м', labelpad=2)
    ax.plot([beta[0], beta[0], 0], [0, beta[1], beta[1]],
            color=GREEN, lw=1., ls=':')
    ax.text(beta[0], -14, r'$\beta_1$', color=GREEN, ha='center')
    ax.text(-9, beta[1]+4, r'$\beta_2$', color=GREEN, ha='right')
    ax.text(53, 31, r'$\mathbf{\beta}$', color=GREEN)
    ax.text(-29, -27, '$R_c$', color=ORANGE)
    fig.text(.535, .65, 'Поворот', color=PURPLE, ha='center')
    fig.add_artist(FancyArrowPatch((.49, .53), (.575, .53),
                                  transform=fig.transFigure, arrowstyle='-|>',
                                  mutation_scale=11, color=PURPLE, lw=1.2))
    fig.text(.27, .035, '$C_b$ в исходных осях',
             ha='center')
    fig.text(.77, .035, r'$\beta_1,\beta_2$; круг $R_c$ сохранён',
             ha='center')
    out = save(fig, 'concept-principal-axes')
    out.update({'rotation_deg': float(np.rad2deg(angle)), 'Q': Q.tolist(),
                'principal_sigmas_m': sig.tolist(), 'C_b_m2': cov.tolist(),
                'b_m': b.tolist(), 'beta_m': beta.tolist(), 'radius_m': radius,
                'transformation': 'orthogonal rotation Q.T; no division by sigmas',
                'norm_preservation_error_m': float(abs(np.linalg.norm(b)-np.linalg.norm(beta))),
                'covariance_diagonalization_error_m2': float(np.linalg.norm(Q.T@cov@Q-np.diag(sig**2)))})
    return out


def mahalanobis():
    sig = np.array([120., 60.])
    vectors = np.array([[120., 0.], [0., 120.]])
    white = vectors / sig
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([.10, .265, .37, .58])
    ax.set_title(r'а) До деления на $\sigma_1,\sigma_2$', pad=8)
    for k in [2, 1]:
        ax.add_patch(Ellipse((0, 0), 2*k*sig[0], 2*k*sig[1],
                             facecolor=BLUE, edgecolor=BLUE,
                             alpha=.07 if k == 2 else .14, lw=1.0))
    for i, (b, color) in enumerate(zip(vectors, [GREEN, PURPLE])):
        arrow(ax, (0, 0), b, color, lw=1.7)
        ax.plot(*b, 'o', color=color, ms=4)
    ax.text(127, 7, r'$\mathbf{b}_A$', color=GREEN)
    ax.text(13, 124, r'$\mathbf{b}_B$', color=PURPLE)
    ax.axhline(0, color=GRAY, lw=.6)
    ax.axvline(0, color=GRAY, lw=.6)
    ax.set_xlim(-255, 255)
    ax.set_ylim(-145, 155)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\beta_1$, м', labelpad=2)
    ax.set_ylabel(r'$\beta_2$, м', labelpad=2)
    ax.set_xticks([-240, -120, 0, 120, 240])
    ax.set_yticks([-120, 0, 120])
    ax.tick_params(length=2.5, pad=2)

    ax = fig.add_axes([.64, .255, .30, .62])
    ax.set_title('б) После деления', pad=6)
    for k in [2, 1]:
        ax.add_patch(Circle((0, 0), k, facecolor=BLUE, edgecolor=BLUE,
                            alpha=.07 if k == 2 else .14, lw=1.))
    ax.text(-.92, .77, '$D=1$', color=BLUE, fontsize=9)
    ax.text(-1.85, 1.53, '$D=2$', color=BLUE, fontsize=9)
    for b, color in zip(white, [GREEN, PURPLE]):
        arrow(ax, (0, 0), b, color, lw=1.7)
        ax.plot(*b, 'o', color=color, ms=4)
    ax.text(1.13, .14, '$A$', color=GREEN)
    ax.text(.15, 2.02, '$B$', color=PURPLE)
    ax.axhline(0, color=GRAY, lw=.6)
    ax.axvline(0, color=GRAY, lw=.6)
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-2.2, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\beta_1/\sigma_1$', labelpad=2)
    ax.set_ylabel(r'$\beta_2/\sigma_2$', labelpad=0)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(length=2.5, pad=2)
    fig.text(.54, .51, r'$\longrightarrow$', ha='center', color=GRAY, fontsize=12)
    fig.text(.27, .045, r'$\|\mathbf{b}_A\|=\|\mathbf{b}_B\|=120$ м', ha='center')
    fig.text(.77, .045, '$D_A=1$, $D_B=2$', ha='center')
    out = save(fig, 'design-mahalanobis')
    out.update({'principal_sigmas_m': sig.tolist(),
                'covariance_principal_m2': np.diag(sig**2).tolist(),
                'b_vectors_principal_m': vectors.tolist(),
                'whitened_vectors': white.tolist(),
                'euclidean_norms_m': np.linalg.norm(vectors, axis=1).tolist(),
                'mahalanobis_squared': np.sum(white**2, axis=1).tolist(),
                'ellipse_and_circle_D_levels': [1, 2],
                'note': 'beta denotes miss-vector components after orthogonal principal-axis rotation; then divide by sigmas'})
    return out


def fixed_direction(result):
    b0 = np.array(result['nominal_b_m'])
    G = np.array(result['G_main_s'])
    cov = np.array(result['C_b_m2'])
    radius, limit = result['HBR_m'], result['pc_limit']
    prob = DiskProbability(cov, order=128, radius=radius)
    prob_check = DiskProbability(cov, order=256, radius=radius)
    direction = np.array([1., 0., 0.])
    g = G @ direction
    minus = -brentq(lambda s: float(prob(b0-s*g))-limit, 0., .3, xtol=1e-14)
    plus = brentq(lambda s: float(prob(b0+s*g))-limit, 0., .3, xtol=1e-14)
    roots = np.array([minus, plus])
    crosses = b0 + roots[:, None] * g
    s = np.linspace(-.16, .16, 601)
    curve = np.asarray(prob(b0+s[:, None]*g))
    xy = np.linspace(-460, 460, 241)
    xx, yy = np.meshgrid(xy, xy)
    grid_b = np.column_stack([xx.ravel(), yy.ravel()])
    # Chunk to bound the memory needed by the quadrature's 128 nodes.
    grid_p = np.concatenate([prob(v) for v in np.array_split(grid_b, 24)]).reshape(xx.shape)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([.045, .22, .425, .67])
    ax.set_title(r'а) Прямая $\mathbf{b}(s)=\mathbf{b}_0+sG\mathbf{d}$', pad=7)
    ax.contourf(xx, yy, grid_p, levels=[limit, 1.], colors=[ORANGE], alpha=.13)
    ax.contour(xx, yy, grid_p, levels=[limit], colors=[ORANGE], linewidths=1.15)
    line = b0 + np.array([-.16, .16])[:, None]*g
    ax.plot(line[:, 0], line[:, 1], color=GRAY, lw=1., zorder=2)
    arrow(ax, b0, b0+.15*g, GREEN, lw=1.5)
    arrow(ax, b0, b0-.15*g, PURPLE, lw=1.5)
    ax.add_patch(Circle((0, 0), radius, facecolor=ORANGE,
                        edgecolor=ORANGE, alpha=.9))
    ax.plot(*b0, 'o', color=BLUE, ms=4.2, zorder=5)
    ax.annotate(r'$\mathbf{b}_0$', xy=b0, xytext=(180, -35), color=BLUE,
                arrowprops={'arrowstyle': '-', 'lw': .7, 'color': BLUE})
    for cross, color in zip(crosses, [PURPLE, GREEN]):
        ax.plot(*cross, 'o', color=color, ms=4.5, zorder=6)
    ax.annotate('$s_+>0$', xy=crosses[1], xytext=(-390, 320), color=GREEN,
                arrowprops={'arrowstyle':'-', 'lw':.7, 'color':GREEN})
    ax.annotate('$s_-<0$', xy=crosses[0], xytext=(160, -360), color=PURPLE,
                arrowprops={'arrowstyle':'-', 'lw':.7, 'color':PURPLE})
    ax.text(-430, -80, r'$P_c>10^{-5}$', color=ORANGE)
    ax.set_xlim(-460, 460)
    ax.set_ylim(-460, 460)
    ax.set_aspect('equal')
    ax.set_xlabel('$b_1$, м', labelpad=1)
    ax.set_ylabel('$b_2$, м', labelpad=1)
    ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400])
    ax.tick_params(length=2.5, pad=2)

    ax = fig.add_axes([.60, .22, .37, .67])
    ax.set_title('б) Условие на вероятность', pad=7)
    ax.axvspan(-.16, minus, facecolor=PURPLE, alpha=.08)
    ax.axvspan(plus, .16, facecolor=GREEN, alpha=.08)
    ax.semilogy(s, curve, color=BLUE, lw=1.6)
    ax.axhline(limit, color=ORANGE, lw=1.1, ls='--')
    ax.text(-.154, limit*1.65, r'$P_c=10^{-5}$', color=ORANGE)
    for root, color in zip(roots, [PURPLE, GREEN]):
        ax.plot(root, limit, 'o', color=color, ms=4.5)
        ax.axvline(root, color=color, lw=.85, ls=':')
    ax.text(-.135, 2.1e-8, '$s_-$', color=PURPLE, ha='center')
    ax.text(.125, 2.1e-8, '$s_+$', color=GREEN, ha='center')
    ax.set_xlim(-.16, .16); ax.set_ylim(1e-9, 2e-2)
    ax.set_xlabel('$s$, м/с', labelpad=1)
    ax.set_ylabel('$P_c(s)$', labelpad=0)
    ax.set_xticks([-.15, 0, .15])
    ax.set_yticks([1e-8, 1e-5, 1e-2])
    ax.tick_params(length=2.5, pad=2)
    fig.text(.265, .055, r'$\mathbf{d}=\mathbf{e}_r$, $t_{\rm ca}-t_m=0{,}4T$',
             ha='center')
    fig.text(.785, .055, r'$s_+=+0{,}09913$; $s_-=-0{,}11209$ м/с', ha='center')
    out = save(fig, 'design-fixed-direction')
    assert abs(plus-result['axis_solutions'][0]['dv_rtn_m_s'][0]) < 1e-10
    assert np.max(np.abs(prob_check(crosses)-limit)) < 1e-12
    out.update({'source': 'example-results.json',
                'lead_periods': result['main_lead_periods'],
                'lead_s': result['main_lead_s'], 'b0_m': b0.tolist(),
                'G_s': G.tolist(), 'd_RTN_unit': direction.tolist(),
                'Gd_s': g.tolist(), 'C_b_m2': cov.tolist(), 'R_c_m': radius,
                'pc_limit': limit, 'quadrature_order': 128,
                'verification_quadrature_order': 256,
                'signed_roots_m_s': roots.tolist(),
                'boundary_b_m': crosses.tolist(),
                'boundary_pc_order128': prob(crosses).tolist(),
                'boundary_pc_order256': prob_check(crosses).tolist(),
                'selected_minimum_absolute_impulse_m_s': float(plus),
                's_curve_m_s': s.tolist(), 'pc_curve': curve.tolist(),
                'risk_contour_grid_m': xy.tolist(),
                'risk_contour_values_saved': False,
                'risk_contour_reproduction': 'DiskProbability(C_b_m2,128,R_c_m) at meshgrid(grid,grid)'})
    return out


def impulse_map(result):
    G = np.array(result['G_main_s'])
    d = .1
    U, singular, VT = np.linalg.svd(G, full_matrices=True)
    # Deterministic sign of the principal input direction.
    vmax = VT[0].copy()
    if vmax[0] < 0:
        vmax *= -1
    vother, vnull = VT[1], VT[2]
    max_dv = d*vmax
    max_b = G @ max_dv
    angle = np.linspace(0, 2*np.pi, 401)
    boundary_dv = d*(np.outer(np.cos(angle), vmax)+np.outer(np.sin(angle), vother))
    boundary_b = boundary_dv @ G.T
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([.015, .14, .43, .74], projection='3d')
    ax.set_title('а) Три компоненты импульса', pad=5)
    phi = np.linspace(0, np.pi, 25)
    az = np.linspace(0, 2*np.pi, 45)
    sx = d*np.outer(np.sin(phi), np.cos(az))
    sy = d*np.outer(np.sin(phi), np.sin(az))
    sz = d*np.outer(np.cos(phi), np.ones_like(az))
    ax.plot_surface(sx, sy, sz, color=BLUE, alpha=.10,
                    edgecolor='none', shade=False, rstride=2, cstride=2)
    for value in [-.06, 0., .06]:
        rr = np.sqrt(d*d-value*value)
        ax.plot(rr*np.cos(az), rr*np.sin(az), value*np.ones_like(az),
                color=BLUE, lw=.55, alpha=.5)
    for k in [0, np.pi/2]:
        ax.plot(d*np.sin(phi)*np.cos(k), d*np.sin(phi)*np.sin(k), d*np.cos(phi),
                color=BLUE, lw=.6, alpha=.5)
    ax.quiver(0, 0, 0, *max_dv, color=GREEN, linewidth=2., arrow_length_ratio=.12)
    ax.scatter(*max_dv, color=GREEN, s=15, depthshade=False)
    ax.text(*(max_dv+[-.035, -.025, .025]), r'$d\,\mathbf{d}_{\max}$', color=GREEN)
    ax.set_xlabel(r'$\Delta V_r$', labelpad=-8)
    ax.set_ylabel(r'$\Delta V_t$', labelpad=-8)
    ax.set_zlabel(r'$\Delta V_n$', labelpad=-5)
    ax.set_xlim(-d, d); ax.set_ylim(-d, d); ax.set_zlim(-d, d)
    # A diagram: the scale and units are given explicitly below, avoiding
    # nine duplicated decimal tick labels on a small 3-D panel.
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
        axis.line.set_color(GRAY)
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-54)

    ax = fig.add_axes([.615, .23, .34, .65])
    ax.set_title('б) Две компоненты промаха', pad=7)
    ax.fill(boundary_b[:, 0], boundary_b[:, 1], facecolor=BLUE, alpha=.14)
    ax.plot(boundary_b[:, 0], boundary_b[:, 1], color=BLUE, lw=1.15)
    other_b = G @ (d*vother)
    ax.plot([-max_b[0], max_b[0]], [-max_b[1], max_b[1]],
            color=GREEN, lw=.8, ls=':')
    ax.plot([-other_b[0], other_b[0]], [-other_b[1], other_b[1]],
            color=PURPLE, lw=.8, ls=':')
    arrow(ax, (0, 0), max_b, GREEN, lw=1.8, mutation=10)
    ax.plot(*max_b, 'o', color=GREEN, ms=4)
    ax.annotate(r'$dG\mathbf{d}_{\max}$', xy=max_b, xytext=(-75, 430), color=GREEN,
                arrowprops={'arrowstyle':'-', 'color':GREEN, 'lw':.7})
    ax.plot(0, 0, 'o', color=GRAY, ms=3)
    ax.axhline(0, color=GRAY, lw=.6); ax.axvline(0, color=GRAY, lw=.6)
    ax.set_xlim(-650, 650); ax.set_ylim(-650, 650)
    ax.set_aspect('equal')
    ax.set_xlabel('$b_1$, м', labelpad=1)
    ax.set_ylabel('$b_2$, м', labelpad=1)
    ax.set_xticks([-600, 0, 600]); ax.set_yticks([-600, 0, 600])
    ax.tick_params(length=2.5, pad=2)
    fig.add_artist(FancyArrowPatch((.438, .52), (.55, .52),
                                  transform=fig.transFigure, arrowstyle='-|>',
                                  mutation_scale=12, color=INK, lw=1.3))
    fig.text(.495, .57, '$G$', ha='center')
    fig.text(.495, .42, r'$2\times3$', ha='center', color=GRAY)
    fig.text(.24, .055, r'$\|\Delta\mathbf{V}\|\leq d$, $d=0{,}1$ м/с', ha='center')
    fig.text(.77, .055, r'$\mathbf{b}_0=0$, $\|\mathbf{b}\|_{\max}=603{,}0$ м', ha='center')
    out = save(fig, 'design-impulse-map')
    eig = np.linalg.eigvalsh(G.T@G)
    check = np.linalg.norm(boundary_dv, axis=1)
    assert np.max(np.abs(check-d)) < 1e-14
    assert np.linalg.norm(G@vnull) < 1e-9
    assert abs(np.linalg.norm(max_b)-d*np.sqrt(eig[-1])) < 1e-9
    out.update({'source': 'G_main_s from example-results.json', 'G_s': G.tolist(),
                'assumed_b0_m': [0., 0.], 'impulse_ball_radius_m_s': d,
                'G_singular_values_s': singular.tolist(),
                'G_transpose_G_eigenvalues_s2': eig.tolist(),
                'unit_principal_impulse_direction_RTN': vmax.tolist(),
                'unit_other_input_direction_RTN': vother.tolist(),
                'unit_null_input_direction_RTN': vnull.tolist(),
                'principal_impulse_m_s': max_dv.tolist(),
                'maximum_b_m': max_b.tolist(),
                'output_ellipse_semiaxes_m': (d*singular).tolist(),
                'boundary_input_impulses_m_s': boundary_dv.tolist(),
                'boundary_output_b_m': boundary_b.tolist(),
                'max_input_boundary_norm_error_m_s': float(np.max(np.abs(check-d))),
                'kernel_residual_s': float(np.linalg.norm(G@vnull)),
                'eigenvalue_maximum_norm_error_m': float(abs(np.linalg.norm(max_b)-d*np.sqrt(eig[-1]))),
                'note': 'The great circle spanned by the two non-null right singular vectors maps to the ellipse boundary; the whole 3-D ball maps onto its filled interior.'})
    return out


def main():
    source = BASE / 'example-results.json'
    result = json.loads(source.read_text())
    records = [cylinder(), principal_axes(), mahalanobis(),
               fixed_direction(result), impulse_map(result)]
    data = {'description': 'Compact explanatory diagrams for lecture 9',
            'reproduce': 'python3 9/images/design_diagrams.py',
            'source_data_file': 'example-results.json',
            'source_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'style': {'background': 'white', 'width_mm': 170, 'height_mm': 72,
                      'minimum_text_pt': 9,
                      'palette': [BLUE, GREEN, ORANGE, PURPLE]},
            'figures': records}
    (BASE / 'design-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'figures': [r['file_base'] for r in records],
                      'signed_radial_roots_m_s': records[3]['signed_roots_m_s'],
                      'maximum_b_m': records[4]['maximum_b_m'],
                      'output_ellipse_semiaxes_m': records[4]['output_ellipse_semiaxes_m']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
