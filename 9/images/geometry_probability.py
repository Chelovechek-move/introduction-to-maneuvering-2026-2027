#!/usr/bin/env python3
"""Conceptual figures for lecture 9. Run from the course root:
python3 9/images/geometry_probability.py

Lengths in metres. Pc is the Gaussian probability integrated over a disk.
All input parameters and sampled curves are saved to concept-data.json.
"""
from pathlib import Path
import json
import os

BASE = Path(__file__).resolve().parent
BUILD = BASE.parent / 'build'
BUILD.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BUILD / 'mplconfig'))
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

BLUE, GREEN, ORANGE, PURPLE = '#187AA5', '#087F78', '#B57714', '#79539C'
INK, GRAY = '#25323F', '#687582'
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 14.5,
    'axes.labelsize': 14.5, 'axes.titlesize': 15,
    'legend.fontsize': 12.5, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .16,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def pc_diagonal(b, sigma, radius, tight=False):
    """1D normal marginal times conditional interval, diagonal covariance."""
    bx, by = np.asarray(b, float)
    sx, sy = np.asarray(sigma, float)
    def integrand(x):
        ymax = np.sqrt(max(0., radius * radius - x * x))
        interval = .5 * (erf((ymax - by)/(np.sqrt(2)*sy))
                           - erf((-ymax - by)/(np.sqrt(2)*sy)))
        return np.exp(-.5*((x-bx)/sx)**2)/(np.sqrt(2*np.pi)*sx)*interval
    return quad(integrand, -radius, radius,
                epsabs=1e-14 if tight else 1e-12,
                epsrel=2e-11 if tight else 2e-9, limit=200)[0]


def save(fig, name):
    fig.savefig(BASE / (name + '.pdf'), bbox_inches='tight', pad_inches=.05)
    fig.savefig(BASE / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def pc_polar(b, sigma, radius):
    """Independent smooth 2D polar quadrature, for verification."""
    q, w = np.polynomial.legendre.leggauss(80)
    r, wr = radius * (q + 1) / 2, radius * w / 2
    theta = 2 * np.pi * np.arange(512) / 512
    xx = r[:, None] * np.cos(theta)[None, :]
    yy = r[:, None] * np.sin(theta)[None, :]
    exponent = -.5 * (((xx-b[0])/sigma[0])**2 + ((yy-b[1])/sigma[1])**2)
    density = np.exp(exponent)/(2*np.pi*sigma[0]*sigma[1])
    return float(np.sum(wr*r*np.mean(density, axis=1)*2*np.pi))


def geometry():
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.0))
    ax = axes[0]
    ax.set_title('а) Относительное движение', pad=17)
    ax.axvline(0, color=BLUE, alpha=.28, lw=7)
    ax.plot([-175, 170], [100, 100], color=GREEN, lw=2.2)
    ax.annotate('', xy=(145, 100), xytext=(95, 100),
                arrowprops={'arrowstyle':'-|>', 'color':GREEN, 'lw':2})
    ax.text(112, 114, r'$\mathbf{v}_{\rm rel}$', color=GREEN)
    ax.plot(-120, 100, 'o', color=GREEN, ms=6)
    ax.text(-170, 120, r'$t_0$', color=GREEN)
    ax.plot(0, 100, 'o', color=BLUE, ms=7)
    ax.text(12, 113, r'$t_{\rm ca}$', color=BLUE)
    ax.annotate('', xy=(-120, 100), xytext=(0, 0),
                arrowprops={'arrowstyle':'-|>', 'color':GRAY, 'lw':1.5})
    ax.text(-118, 48, r'$\rho(t_0)$', color=GRAY)
    ax.annotate('', xy=(0, 100), xytext=(0, 0),
                arrowprops={'arrowstyle':'-|>', 'color':BLUE, 'lw':2})
    ax.text(11, 48, r'$\|\mathbf{b}\|$', color=BLUE)
    ax.add_patch(Circle((0, 0), 10, color=ORANGE, alpha=.55))
    ax.plot(0, 0, 'o', color=INK, ms=4)
    ax.text(20, -10, 'Первый объект', fontsize=13)
    ax.set_xlim(-185, 185); ax.set_ylim(-30, 150)
    ax.set_xlabel(r'Продольная координата $s$, м')
    ax.set_ylabel('Поперечное расстояние, м')
    ax.text(0, 151, r'Плоскость $s=0$', color=BLUE, ha='center', va='bottom', fontsize=12.5)
    ax = axes[1]
    ax.set_title('б) Плоскость сближения', pad=17)
    ax.add_patch(Circle((0, 0), 10, facecolor=ORANGE, edgecolor=ORANGE, alpha=.5))
    ax.annotate('', xy=(80, 60), xytext=(0, 0),
                arrowprops={'arrowstyle':'-|>', 'color':BLUE, 'lw':2})
    ax.plot(80, 60, 'o', color=BLUE, ms=7)
    ax.text(89, 66, r'$\mathbf{b}=P\rho$', color=BLUE)
    ax.annotate(r'$R_c$', xy=(7, -7), xytext=(35, -30),
                arrowprops={'arrowstyle':'-', 'color':ORANGE}, color=ORANGE)
    ax.axhline(0, lw=.7, color=GRAY); ax.axvline(0, lw=.7, color=GRAY)
    ax.text(102, -47, r'$\mathbf{v}_{\rm rel}\perp$ плоскости', fontsize=13, ha='center')
    ax.set_xlim(-50, 185); ax.set_ylim(-60, 115); ax.set_aspect('equal')
    ax.set_xlabel(r'$b_1$, м'); ax.set_ylabel(r'$b_2$, м')
    fig.tight_layout(w_pad=2.5)
    save(fig, 'concept-geometry')


def risk():
    sigma, radius = (120., 30.), 10.
    cases = [np.array([0.,80.]), np.array([140.,0.])]
    probabilities = [pc_diagonal(b, sigma, radius, True) for b in cases]
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.1), sharex=True, sharey=True)
    for j, (ax, b, pc) in enumerate(zip(axes, cases, probabilities)):
        for k, alpha in [(2,.08),(1,.18)]:
            ax.add_patch(Ellipse(b, 2*k*sigma[0], 2*k*sigma[1],
                                facecolor=BLUE, edgecolor=BLUE, alpha=alpha, lw=1.2))
        ax.add_patch(Circle((0,0),radius,facecolor=ORANGE, edgecolor=ORANGE, lw=1.5))
        ax.plot(*b,'o',color=BLUE,ms=5)
        ax.annotate('', xy=b, xytext=(0,0),
                    arrowprops={'arrowstyle':'-|>','color':GREEN,'lw':1.5})
        ax.axhline(0,color=GRAY,lw=.7); ax.axvline(0,color=GRAY,lw=.7)
        ax.set_title(('а)' if j==0 else 'б)') + rf' $\|\mathbf{{b}}\|={np.linalg.norm(b):.0f}$ м', pad=9)
        ax.text(.04,.93, rf'$P_c={pc:.5f}$', transform=ax.transAxes, va='top',
                bbox={'facecolor':'white','edgecolor':'none','alpha':.88})
        ax.set_xlim(-275,405);ax.set_ylim(-100,175)
        ax.set_aspect('equal');ax.set_xlabel(r'$b_1$, м')
        ax.grid(True)
    axes[0].set_ylabel(r'$b_2$, м')
    fig.tight_layout(w_pad=2)
    save(fig, 'concept-risk')
    return {'sigma_m':list(sigma),'radius_m':radius,
            'cases':[{'b_m':b.tolist(),'distance_m':float(np.linalg.norm(b)),
                      'pc':pc} for b,pc in zip(cases,probabilities)]}


def dilution():
    radius, b = 10., np.array([100.,0.])
    sigmas=np.geomspace(5,2000,240)
    probabilities=np.array([pc_diagonal(b,(s,s),radius,True) for s in sigmas])
    result=minimize_scalar(lambda s:-pc_diagonal(b,(s,s),radius,True),
                           bounds=(10,500),method='bounded',options={'xatol':1e-8})
    smax,pmax=float(result.x),float(-result.fun)
    fig,ax=plt.subplots(figsize=(8.3,3.6))
    ax.semilogx(sigmas,1000*probabilities,color=BLUE,lw=2.2)
    ax.axvline(smax,color=ORANGE,ls='--',lw=1.1)
    ax.plot(smax,1000*pmax,'o',color=ORANGE,ms=5)
    ax.text(smax*1.13,1000*pmax*.94, rf'$\sigma={smax:.1f}$ м',color=ORANGE)
    ax.text(.97,.93,r'$\|\mathbf{b}\|=100$ м, $R_c=10$ м',transform=ax.transAxes,
            ha='right',va='top',bbox={'facecolor':'white','edgecolor':'none'})
    ax.set_xlabel(r'Среднеквадратическая ошибка $\sigma$, м')
    ax.set_ylabel(r'$P_c\cdot10^3$')
    ax.set_ylim(0,pmax*1180);ax.set_xlim(5,2000)
    fig.tight_layout()
    save(fig,'concept-dilution')
    return {'radius_m':radius,'b_m':b.tolist(),'sigma_m':sigmas.tolist(),
            'pc':probabilities.tolist(),'maximum_sigma_m':smax,'maximum_pc':pmax,
            'pc_sigma_20':pc_diagonal(b,(20,20),radius,True),
            'pc_sigma_500':pc_diagonal(b,(500,500),radius,True)}


if __name__=='__main__':
    geometry()
    data={'risk':risk(),'dilution':dilution(),
          'method':'1D adaptive quadrature of bivariate Gaussian over disk; diagonal covariance',
          'run':'python3 9/images/geometry_probability.py'}
    verification = {}
    zero_1d=pc_diagonal((0,0),(30,30),10,True)
    zero_exact=1-np.exp(-10**2/(2*30**2))
    verification['zero_miss_isotropic_absolute_error']=float(abs(zero_1d-zero_exact))
    errors=[]
    for case in data['risk']['cases']:
        errors.append(abs(case['pc']-pc_polar(case['b_m'],data['risk']['sigma_m'],10)))
    verification['risk_1d_vs_2d_polar_max_absolute_error']=float(max(errors))
    verification['passed']=bool(verification['zero_miss_isotropic_absolute_error']<1e-12
                                 and max(errors)<1e-12)
    assert verification['passed'], verification
    data['verification']=verification
    (BASE/'concept-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'risk_cases':data['risk']['cases'],
                      'dilution_maximum_sigma_m':data['dilution']['maximum_sigma_m'],
                      'dilution_maximum_pc':data['dilution']['maximum_pc']},ensure_ascii=False,indent=2))
