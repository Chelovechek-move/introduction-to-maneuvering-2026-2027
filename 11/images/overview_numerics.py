#!/usr/bin/env python3
"""Four reproducible overview figures for lecture 11.

Run from the course root:
    python3 11/images/overview_numerics.py

Units are km/s for orbital formulae, m/s for reported impulses, and SI for
propulsion. All samples, curves, parameters, and internal consistency checks
are written to overview-data.json. The synthetic samples and limits are
teaching assumptions, not flight statistics or hardware specifications.
These are separate numerical illustrations; the tank example does not share
the 2000 kg mass model used for the propulsion plots.
"""
from pathlib import Path
import json
import os

BASE = Path(__file__).resolve().parent
BUILD = BASE.parent / 'build'
PREVIEW = BUILD / 'overview-preview'
PREVIEW.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BUILD / 'mplconfig'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

BLUE, GREEN, ORANGE, PURPLE = '#187AA5', '#087F78', '#B57714', '#79539C'
RED, INK, GRAY = '#C44847', '#25323F', '#687582'
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 14.5,
    'axes.labelsize': 14.5, 'axes.titlesize': 14.5,
    'legend.fontsize': 12.2, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .16,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
MU, RA = 398600.4418, 42164.
G0, M0, D = 9.80665, 2000., 100.


def save(fig, name):
    fig.savefig(BASE/(name+'.pdf'), bbox_inches='tight', pad_inches=.05)
    fig.savefig(PREVIEW/(name+'.png'), dpi=180, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def injection_dv(rp, inc_deg):
    """One combined plane-change/circularization impulse at the common node/apogee."""
    a = (RA+np.asarray(rp))/2
    va = np.sqrt(MU*(2/RA-1/a))
    vc = np.sqrt(MU/RA)
    angle = np.deg2rad(inc_deg)
    # Half-angle form avoids cancellation and is exactly the law of cosines.
    dv = np.sqrt((vc-va)**2+4*vc*va*np.sin(angle/2)**2)
    return va, vc, dv*1000


def propellant(isp):
    return -M0*np.expm1(-D/(G0*np.asarray(isp)))


def data():
    rng = np.random.default_rng(20261117)
    rp = rng.normal(7000., 120., 400)
    inc = rng.normal(6., .5, 400)
    va, vc, dv = injection_dv(rp, inc)
    nominal_va, _, nominal_dv = injection_dv(7000., 6.)
    # A direct vector difference is independent of the half-angle expression.
    direct = np.sqrt((vc-va*np.cos(np.deg2rad(inc)))**2
                     +(va*np.sin(np.deg2rad(inc)))**2)*1000
    rgrid = np.linspace(6550., 7450., 181)
    igrid = np.linspace(4.1, 7.9, 153)
    rmesh, imesh = np.meshgrid(rgrid, igrid)
    costmesh = injection_dv(rmesh, imesh)[2]
    isp_curve = np.linspace(220., 330., 221)
    thrust_curve = np.linspace(180., 520., 241)
    dm300, dm270 = float(propellant(300.)), float(propellant(270.))
    total_impulse = dm300*G0*300
    repeatability_residual = np.array([-.20, .10, -.35, .25, -.05, .30, -.10, .05])
    raw_percent = -1.5+repeatability_residual
    # The additive decomposition is defined in percent of the nominal impulse.
    # After removing the known additive bias, exactly the same assigned
    # pulse-to-pulse deviations remain; no statistical estimation is claimed.
    mfuel0, mox0, mixture_nom, mixture_actual = 50., 80., 1.6, 1.8
    fuel_consumed = mox0/mixture_actual
    remaining_fuel = mfuel0-fuel_consumed
    residual_error = G0*300*np.log(M0/(M0-dm300))-D
    orbital_error = float(np.max(np.abs(direct-dv)))
    checks = {
        'max_vector_difference_error_m_s': orbital_error,
        'rocket_equation_closure_m_s': float(residual_error),
        'mixture_fuel_balance_kg': float(fuel_consumed+remaining_fuel-mfuel0),
        'repeatability_mean_residual_percent': float(np.mean(repeatability_residual)),
    }
    assert orbital_error < 1e-10
    assert abs(residual_error) < 1e-10
    assert np.min(rp) > 6378.1363 and np.max(rp) < RA
    assert np.min(inc) > 0 and np.max(inc) < 90
    return {
        'note': 'All distributions, deviations, masses, and constraints are teaching assumptions.',
        'injection': {
            'model': 'Central field; common apogee on equatorial node; target circular equatorial GEO.',
            'mu_km3_s2': MU, 'common_apogee_km': RA,
            'nominal_perigee_km': 7000., 'nominal_inclination_deg': 6.,
            'sample_seed': 20261117, 'sample_count': len(rp),
            'perigee_sigma_km': 120., 'inclination_sigma_deg': .5,
            'nominal_apogee_speed_km_s': float(nominal_va),
            'target_circular_speed_km_s': float(vc),
            'nominal_dv_m_s': float(nominal_dv),
            'sample_min_dv_m_s': float(np.min(dv)), 'sample_max_dv_m_s': float(np.max(dv)),
            'sample_perigee_km': rp.tolist(), 'sample_inclination_deg': inc.tolist(),
            'sample_dv_m_s': dv.tolist(),
            'grid_perigee_km': rgrid.tolist(), 'grid_inclination_deg': igrid.tolist(),
            'grid_dv_m_s': costmesh.tolist(),
        },
        'propulsion': {
            'model': 'Constant thrust and Isp; no staging or ancillary consumption; scalar delivered D.',
            'initial_mass_kg': M0, 'required_dv_m_s': D, 'g0_m_s2': G0,
            'nominal_isp_s': 300., 'low_isp_s': 270.,
            'nominal_thrust_N': 450., 'low_thrust_N': 300., 'duration_limit_s': 600.,
            'nominal_propellant_kg': dm300, 'low_isp_propellant_kg': dm270,
            'nominal_duration_s': total_impulse/450,
            'low_thrust_duration_s': total_impulse/300,
            'minimum_thrust_for_duration_limit_N': total_impulse/600,
            'dv_at_300N_for_600s_m_s': G0*300*np.log(M0/(M0-300*600/(G0*300))),
            'isp_grid_s': isp_curve.tolist(),
            'propellant_curve_kg': propellant(isp_curve).tolist(),
            'thrust_grid_N': thrust_curve.tolist(),
            'duration_curve_s': (total_impulse/thrust_curve).tolist(),
        },
        'repeatability': {
            'model': 'Assigned deviations, not measurements. Known additive systematic error plus residual percent of nominal.',
            'nominal_pulse_m_s': 100., 'systematic_scale': .985,
            'maneuver_number': list(range(1,9)), 'systematic_error_percent': -1.5,
            'pulse_to_pulse_error_percent': repeatability_residual.tolist(),
            'raw_error_percent': raw_percent.tolist(),
            'after_known_bias_removed_percent': repeatability_residual.tolist(),
            'raw_impulse_m_s': (100+raw_percent).tolist(),
            'note': 'Removing known bias is illustrative; it does not remove pulse-to-pulse deviations or prove future stability.',
        },
        'mixture': {
            'model': 'Illustrative bipropellant main engine and fuel-only backup; routing assumed as drawn.',
            'initial_fuel_kg': mfuel0, 'initial_oxidizer_kg': mox0,
            'nominal_oxidizer_fuel_mass_ratio': mixture_nom,
            'actual_oxidizer_fuel_mass_ratio': mixture_actual,
            'fuel_consumed_when_oxidizer_exhausted_kg': fuel_consumed,
            'remaining_fuel_kg': remaining_fuel, 'remaining_oxidizer_kg': 0.,
            'backup_uses': 'fuel only',
            'note': 'Remaining fuel may serve the backup engine under the assumed routing; oxidizer cannot.',
        },
        'checks': checks,
    }


def injection_figure(d):
    q = d['injection']
    rp, inc, dv = [np.array(q[k]) for k in (
        'sample_perigee_km','sample_inclination_deg','sample_dv_m_s')]
    rr, ii = np.meshgrid(q['grid_perigee_km'], q['grid_inclination_deg'])
    zz = np.array(q['grid_dv_m_s'])
    fig, ax = plt.subplots(figsize=(9.2,4.8))
    levels = np.arange(1400, 1521, 20)
    cs = ax.contour(rr, ii, zz, levels=levels, colors=GRAY,
                    linewidths=.8, alpha=.65)
    ax.clabel(cs, fmt='%d м/с', fontsize=11.5, inline=True)
    sc = ax.scatter(rp, inc, c=dv, cmap='viridis', s=23, alpha=.85,
                    edgecolors='white', linewidths=.2, zorder=3)
    ax.scatter(7000, 6, marker='*', s=200, color=RED, edgecolor='white',
               linewidth=.9, zorder=5, label='Номинальные условия')
    ax.annotate('Номинальный манёвр:\n1453,1 м/с', xy=(7000,6), xytext=(7165,4.40),
                ha='center', color=RED, fontsize=12.5,
                arrowprops={'arrowstyle':'-','color':RED,'lw':1.1},
                bbox={'facecolor':'white','edgecolor':'none','pad':2,'alpha':.93})
    cb = fig.colorbar(sc,ax=ax,pad=.025)
    cb.set_label('Модуль апогейного импульса, м/с')
    ax.set_xlabel(r'Радиус перигея орбиты выведения $r_p$, км')
    ax.set_ylabel(r'Наклонение $i$, град')
    ax.set_xlim(6550,7450); ax.set_ylim(4.1,7.9)
    ax.legend(loc='upper right', framealpha=.95, fontsize=12)
    fig.tight_layout()
    save(fig,'overview-injection')


def propulsion_figure(d):
    q=d['propulsion']
    fig, axs = plt.subplots(1,2,figsize=(10.0,4.2))
    ax=axs[0]
    ax.plot(q['isp_grid_s'],q['propellant_curve_kg'],color=BLUE,lw=2.5)
    for isp,dm,color in [(300,q['nominal_propellant_kg'],GREEN),
                         (270,q['low_isp_propellant_kg'],ORANGE)]:
        ax.plot(isp,dm,'o',ms=7,color=color)
        ax.annotate(f'{isp} с: {dm:.2f} кг'.replace('.',','),
            xy=(isp,dm), xytext=(isp-37,dm+5.5),
            color=color,fontsize=11.8,
            arrowprops={'arrowstyle':'-','color':color,'lw':1.})
    ax.set_title('а) Удельный импульс и расход',loc='left',pad=12)
    ax.set_xlabel(r'$I_{\mathrm{sp}}$, с')
    ax.set_ylabel(r'Расход рабочего тела $\Delta m$, кг')
    ax.set_xlim(215,335);ax.set_ylim(59,94)
    ax=axs[1]
    thrust=np.array(q['thrust_grid_N'])
    duration=np.array(q['duration_curve_s'])/60
    ax.axhspan(10,23,color=RED,alpha=.07)
    ax.plot(thrust,duration,color=BLUE,lw=2.5)
    ax.axhline(10,color=RED,ls='--',lw=1.4)
    ax.text(511,10.45,'Предел 10 мин',ha='right',color=RED,fontsize=11.5)
    for f,t,color,offset in [
        (450,q['nominal_duration_s']/60,GREEN,(-120,-2.4)),
        (300,q['low_thrust_duration_s']/60,ORANGE,(43,3.4))]:
        ax.plot(f,t,'o',ms=7,color=color)
        ax.annotate(f'{f} Н: {t:.2f} мин'.replace('.',','),
            xy=(f,t),xytext=(f+offset[0],t+offset[1]),fontsize=11.8,color=color,
            arrowprops={'arrowstyle':'-','color':color,'lw':1.})
    ax.set_title('б) Тяга и длительность',loc='left',pad=12)
    ax.set_xlabel(r'Тяга $F$, Н')
    ax.set_ylabel(r'Длительность $\tau$, мин')
    ax.set_xlim(170,530);ax.set_ylim(3.5,23)
    fig.tight_layout(w_pad=2.0)
    save(fig,'overview-propulsion')


def repeatability_figure(d):
    q=d['repeatability']
    x=np.array(q['maneuver_number'])
    fig,ax=plt.subplots(figsize=(9.2,3.65))
    ax.axhline(0,color=BLUE,lw=1.15,ls='--')
    ax.axhline(-1.5,color=GRAY,lw=1.,ls=':')
    ax.plot(x,q['raw_error_percent'],'o-',color=ORANGE,lw=1.5,ms=6,
            label='До учёта смещения')
    ax.plot(x,q['after_known_bias_removed_percent'],'s-',color=GREEN,lw=1.5,ms=6,
            label='После учёта смещения')
    ax.text(4.5,-.92,'Систематическое смещение −1,5%',ha='center',
            fontsize=11.5,color=GRAY,
            bbox={'facecolor':'white','edgecolor':'none','pad':1})
    ax.set_xticks(x)
    ax.set_ylim(-2.05,.77);ax.set_xlim(.7,8.3)
    ax.set_xlabel('Номер манёвра')
    ax.set_ylabel('Отклонение импульса, %')
    ax.legend(loc='upper left',ncol=2,framealpha=.96,fontsize=12)
    fig.tight_layout()
    save(fig,'overview-repeatability')


def mixture_figure(d):
    q=d['mixture']
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.8),
                         gridspec_kw={'width_ratios':[1.28,1.]})
    ax=axs[0]
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.set_title('а) Основной и резервный двигатели',loc='left',pad=13)
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009,rounding_size=.02',
                                   facecolor=color+'12',edgecolor=color,lw=1.4))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=12.2,color=INK)
    box(.025,.69,.36,.18,'Горючее\n50 кг',BLUE)
    box(.58,.69,.36,.18,'Окислитель\n80 кг',ORANGE)
    box(.05,.31,.46,.19,'Основной\nдвухкомпонентный',GRAY)
    box(.56,.11,.42,.19,'Резервный\nоднокомпонентный',GREEN)
    def line(x,y,color,style='-'):
        ax.annotate('',xy=(x[-1],y[-1]),xytext=(x[-2],y[-2]),
                    arrowprops={'arrowstyle':'-|>','color':color,'lw':1.8,'linestyle':style})
        if len(x)>2:ax.plot(x[:-1],y[:-1],color=color,lw=1.8,ls=style)
    line([.20,.20],[.69,.50],BLUE)
    line([.76,.76,.51],[.69,.405,.405],ORANGE)
    line([.20,.015,.015,.77,.77],[.58,.58,.04,.04,.11],GREEN,'--')
    ax.text(.60,.60,'Два компонента',fontsize=11.5,color=GRAY,ha='center')
    ax.text(.32,.185,'Только\nгорючее',fontsize=11.5,color=GREEN,ha='center',va='center')
    ax.text(.49,.95,'Условная схема подачи',fontsize=11.5,color=GRAY,ha='center')
    ax=axs[1]
    ax.set_title('б) Окислитель закончился раньше',loc='left',pad=13)
    consumed=[q['fuel_consumed_when_oxidizer_exhausted_kg'],80.]
    residual=[q['remaining_fuel_kg'],0.]
    xx=np.array([0.,1.])
    ax.bar(xx,consumed,width=.48,color='#AFC2CE',label='Израсходовано')
    ax.bar(xx,residual,bottom=consumed,width=.48,color=ORANGE,label='Остаток')
    ax.text(0,52.3,'50 кг',ha='center',color=GRAY,fontsize=12)
    ax.text(1,82.3,'80 кг',ha='center',color=GRAY,fontsize=12)
    ax.annotate('Осталось\n5,56 кг',xy=(0,47.2),xytext=(.43,61.5),
                fontsize=12,color=ORANGE,ha='center',
                arrowprops={'arrowstyle':'-','color':ORANGE,'lw':1.1})
    ax.set_xticks(xx,['Горючее','Окислитель'],fontsize=12)
    ax.set_ylim(0,95);ax.set_xlim(-.55,1.55)
    ax.set_ylabel('Масса, кг')
    ax.text(.5,.97,'Фактическое O/F = 1,8',transform=ax.transAxes,
            ha='center',fontsize=11.8,color=INK)
    ax.legend(loc='upper center',bbox_to_anchor=(.50,-.10),ncol=1,frameon=False,fontsize=11.8)
    fig.tight_layout(w_pad=1.6)
    save(fig,'overview-mixture')


def macros(d):
    p=d['propulsion'];q=d['injection'];r=d['mixture']
    values={
        'OvInjectionNominalDV':q['nominal_dv_m_s'],
        'OvInjectionMinDV':q['sample_min_dv_m_s'],
        'OvInjectionMaxDV':q['sample_max_dv_m_s'],
        'OvFuelNominal':p['nominal_propellant_kg'],
        'OvFuelLowIsp':p['low_isp_propellant_kg'],
        'OvDurationNominal':p['nominal_duration_s'],
        'OvDurationLowThrust':p['low_thrust_duration_s'],
        'OvDurationNominalMinutes':p['nominal_duration_s']/60,
        'OvDurationLowThrustMinutes':p['low_thrust_duration_s']/60,
        'OvMinimumThrust':p['minimum_thrust_for_duration_limit_N'],
        'OvLimitedDV':p['dv_at_300N_for_600s_m_s'],
        'OvMixtureResidual':r['remaining_fuel_kg'],
    }
    out=['% Generated by overview_numerics.py; all values belong to stated teaching models.']
    for k,v in values.items():
        val=f'{v:.3f}'.replace('.',r'{,}')
        out.append(r'\newcommand{'+chr(92)+k+'}{'+val+'}')
    (BASE/'overview-values.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')


def main():
    d=data()
    (BASE/'overview-data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    macros(d)
    injection_figure(d)
    propulsion_figure(d)
    repeatability_figure(d)
    mixture_figure(d)
    print(json.dumps({
        'nominal_GTO_GEO_dv_m_s':d['injection']['nominal_dv_m_s'],
        'sample_dv_range_m_s':[d['injection']['sample_min_dv_m_s'],d['injection']['sample_max_dv_m_s']],
        'propellant_kg':[d['propulsion']['nominal_propellant_kg'],d['propulsion']['low_isp_propellant_kg']],
        'duration_s':[d['propulsion']['nominal_duration_s'],d['propulsion']['low_thrust_duration_s']],
        'checks':d['checks']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
