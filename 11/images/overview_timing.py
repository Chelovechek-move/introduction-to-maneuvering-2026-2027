#!/usr/bin/env python3
"""Timing contingencies for the lecture-11 overview.

Run from the course root: python3 11/images/overview_timing.py
Outputs: timing-data.json, timing-values.tex, overview-delay.pdf,
overview-missed.pdf and overview-abort.pdf. PNG previews stay in
11/build/overview-preview.

The delayed impulsive example fixes only the two apsidal radii and plane;
argument of pericentre, terminal phase and terminal time are free.
The extra impulses follow a prescribed apsidal repair, not an optimum.
The abort example is a separate finite-thrust calculation whose reference
is the fully completed finite burn, not the impulsive target.
All parameters are teaching assumptions, not real-spacecraft specifications.
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
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

MU, RE = 398600.4418, 6378.1363
RP0, RA0, D0 = 7000., 12000., 100.
M0, FORCE, ISP, G0 = 2000., 450., 300., 9.80665
C = G0 * ISP
BLUE, GREEN, ORANGE, PURPLE, RED = (
    '#187AA5', '#087F78', '#B57714', '#79539C', '#C44847')
INK, GRAY = '#25323F', '#687582'
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 14,
    'axes.labelsize': 13.5, 'axes.titlesize': 14,
    'legend.fontsize': 11.5, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': .16,
    'pdf.fonttype': 42, 'ps.fonttype': 42})


def orbit(r, v):
    """Planar classical a,e,omega,M from Cartesian geometry."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    radius = np.linalg.norm(r)
    h = float(r[0]*v[1] - r[1]*v[0])
    energy = .5*np.dot(v, v) - MU/radius
    a = -MU/(2*energy)
    ev = np.array([v[1]*h, -v[0]*h])/MU - r/radius
    e = np.linalg.norm(ev)
    omega = np.arctan2(ev[1], ev[0])
    nu = np.arctan2(ev[0]*r[1] - ev[1]*r[0], np.dot(ev, r))
    ecc = np.arctan2(np.sqrt(1-e*e)*np.sin(nu), e+np.cos(nu))
    mean = (ecc-e*np.sin(ecc)) % (2*np.pi)
    return {'a_km': float(a), 'e': float(e),
            'omega_rad': float(omega),
            'omega_deg': float(np.rad2deg(omega)),
            'M_rad': float(mean), 'rp_km': float(a*(1-e)),
            'ra_km': float(a*(1+e)),
            'period_s': float(2*np.pi*np.sqrt(a**3/MU)),
            'h_km2_s': h, 'energy_km2_s2': float(energy),
            'R_km': r.tolist(), 'V_km_s': v.tolist()}


def state(q, dt=0.):
    a, e, w = q['a_km'], q['e'], q['omega_rad']
    mean = q['M_rad'] + np.sqrt(MU/a**3)*dt
    ecc = mean
    for _ in range(20):
        correction = (ecc-e*np.sin(ecc)-mean)/(1-e*np.cos(ecc))
        ecc -= correction
        if abs(correction) < 1e-14:
            break
    rr = a*(1-e*np.cos(ecc))
    r = np.array([a*(np.cos(ecc)-e), a*np.sqrt(1-e*e)*np.sin(ecc)])
    v = np.sqrt(MU*a)/rr*np.array([-np.sin(ecc), np.sqrt(1-e*e)*np.cos(ecc)])
    cw, sw = np.cos(w), np.sin(w)
    rot = np.array([[cw, -sw], [sw, cw]])
    return rot @ r, rot @ v


def et(r):
    return np.array([-r[1], r[0]])/np.linalg.norm(r)


def speed(radius, a):
    return np.sqrt(MU*(2/radius - 1/a))


def coast_rhs(_t, z):
    r, v = z[:2], z[2:4]
    return np.r_[v, -MU*r/np.linalg.norm(r)**3]


def apsis_event(direction):
    def event(_t, z):
        return float(np.dot(z[:2], z[2:4]))
    event.direction = direction
    event.terminal = True
    return event


def coast_to_apsis(r, v, direction):
    q = orbit(r, v)
    sol = solve_ivp(coast_rhs, (0., q['period_s']*1.2), np.r_[r, v],
                    method='DOP853', rtol=2e-13, atol=2e-13,
                    max_step=35., events=apsis_event(direction))
    if not len(sol.t_events[0]):
        raise RuntimeError('Requested apsis was not found')
    return float(sol.t_events[0][0]), sol.y_events[0][0]


def calculate_delay():
    a0 = (RP0+RA0)/2
    va0 = speed(RA0, a0)
    r0, v0 = np.array([-RA0, 0.]), np.array([0., -va0])
    initial = orbit(r0, v0)
    target = orbit(r0, v0 + D0/1000*et(r0))
    samples = []
    for delay in np.linspace(0., 900., 91):
        r, v = state(initial, delay)
        after = orbit(r, v + D0/1000*et(r))
        # First repair: keep the attained perigee and lower the apogee.
        amid = (after['rp_km'] + RA0)/2
        dv_peri = (speed(after['rp_km'], amid) -
                   speed(after['rp_km'], after['a_km']))*1000
        # Second repair: raise the perigee at the restored apogee.
        dv_apo = (speed(RA0, target['a_km']) - speed(RA0, amid))*1000
        assert dv_peri <= 1e-8 and dv_apo >= -1e-8
        rp2, ra2 = after['rp_km'], RA0
        a2 = (rp2+ra2)/2
        if delay == 0.:
            tp, ta = 0., 0.
        else:
            next_peri_dt = ((2*np.pi-after['M_rad']) % (2*np.pi)) / np.sqrt(MU/after['a_km']**3)
            tp = delay + next_peri_dt
            ta = tp + np.pi*np.sqrt(a2**3/MU)
        samples.append({'delay_s': float(delay), 'after_delayed_impulse': after,
            'perigee_correction_m_s': float(dv_peri),
            'apogee_correction_m_s': float(dv_apo),
            'extra_characteristic_speed_m_s': float(abs(dv_peri)+abs(dv_apo)),
            'total_J_m_s': float(D0+abs(dv_peri)+abs(dv_apo)),
            'perigee_correction_time_s': float(tp),
            'completion_time_s': float(ta),
            'final_apsidal_radii_km': [target['rp_km'], RA0],
            'final_omega_deg': after['omega_deg'],
            'late_position_km': r.tolist(),
            'late_direction_xy': et(r).tolist()})

    checks = []
    for delay in (120., 600., 900.):
        row = next(x for x in samples if x['delay_s'] == delay)
        s = solve_ivp(coast_rhs, (0., delay), np.r_[r0, v0],
                      method='DOP853', rtol=2e-13, atol=2e-13, max_step=20.)
        analytic_r, analytic_v = state(initial, delay)
        position_error_m = np.linalg.norm(s.y[:2, -1]-analytic_r)*1000
        rb = s.y[:2, -1]
        vb = s.y[2:4, -1] + .1*et(rb)
        dtp, zp = coast_to_apsis(rb, vb, +1)
        vp = zp[2:4] + row['perigee_correction_m_s']/1000*et(zp[:2])
        dta, za = coast_to_apsis(zp[:2], vp, -1)
        vf = za[2:4] + row['apogee_correction_m_s']/1000*et(za[:2])
        final = orbit(za[:2], vf)
        rp_error = final['rp_km']-target['rp_km']
        ra_error = final['ra_km']-RA0
        time_error = delay + dtp + dta - row['completion_time_s']
        assert position_error_m < 1e-4
        assert abs(rp_error) < 2e-6 and abs(ra_error) < 2e-6
        assert abs(time_error) < 2e-5
        checks.append({'delay_s': delay, 'coast_position_error_m': float(position_error_m),
            'final_rp_error_km': float(rp_error), 'final_ra_error_km': float(ra_error),
            'completion_time_error_s': float(time_error), 'numerical_final_orbit': final})
    return {'initial': initial, 'target_apsidal_radii': target,
            'delay_samples': samples, 'independent_DOP853_checks': checks,
            'missed': {'waiting_time_s': initial['period_s'],
                       'waiting_time_h': initial['period_s']/3600,
                       'eventual_impulse_m_s': D0, 'total_J_m_s': D0},
            'assumptions': {
                'fixed': 'apogee and perigee radii, original plane',
                'free': 'argument of pericentre, final phase and final time',
                'repair': 'retrograde at next perigee, prograde at next apogee',
                'optimality': 'not optimized; cost belongs to this prescribed repair'}}


def finite_rhs(_t, z):
    r, v, mass = z[:2], z[2:4], z[4]
    ath = FORCE/mass/1000*et(r)
    ag = -MU*r/np.linalg.norm(r)**3
    # Extra states integrate work, torque and acceleration magnitude.
    return np.r_[v, ag+ath, -FORCE/C,
                  np.dot(v, ath), r[0]*ath[1]-r[1]*ath[0],
                  np.linalg.norm(ath)]


def calculate_abort(initial):
    planned_fuel = M0*(1-np.exp(-D0/C))
    duration = planned_fuel*C/FORCE
    r0, v0 = state(initial)
    base_energy = .5*np.dot(v0, v0)-MU/np.linalg.norm(r0)
    base_h = r0[0]*v0[1]-r0[1]*v0[0]
    z0 = np.r_[r0, v0, M0, 0., 0., 0.]
    rows, checks = [], []
    for fraction in (0., .25, .5, .75, 1.):
        elapsed = duration*fraction
        if fraction == 0.:
            z = z0.copy()
            check = z0.copy()
        else:
            sol = solve_ivp(finite_rhs, (0., elapsed), z0, method='DOP853',
                            rtol=2e-12, atol=2e-12, max_step=8.)
            alt = solve_ivp(finite_rhs, (0., elapsed), z0, method='RK45',
                            rtol=3e-13, atol=3e-13, max_step=.8)
            z, check = sol.y[:, -1], alt.y[:, -1]
        q = orbit(z[:2], z[2:4])
        dv = C*np.log(M0/z[4])
        fuel = M0-z[4]
        work_error = q['energy_km2_s2']-base_energy-z[5]
        torque_error = q['h_km2_s']-base_h-z[6]
        integrated_impulse_error = z[7]*1000-dv
        position_error = np.linalg.norm(z[:2]-check[:2])*1000
        mass_error = z[4]-(M0-FORCE/C*elapsed)
        assert abs(work_error) < 1e-9
        assert abs(torque_error) < 1e-6
        assert abs(integrated_impulse_error) < 1e-8
        assert position_error < 1e-4
        assert abs(mass_error) < 1e-7
        rows.append({'fraction_of_planned_duration': fraction,
            'stop_time_s': elapsed, 'mass_kg': float(z[4]),
            'fuel_used_kg': float(fuel), 'characteristic_speed_m_s': float(dv),
            'orbit_after_stop': q})
        checks.append({'fraction': fraction, 'work_balance_error_km2_s2': float(work_error),
            'angular_momentum_balance_error_km2_s': float(torque_error),
            'impulse_quadrature_minus_rocket_equation_m_s': float(integrated_impulse_error),
            'DOP853_vs_RK45_position_error_m': float(position_error),
            'mass_vs_linear_exact_error_kg': float(mass_error)})
    return {'initial_mass_kg': M0, 'force_N': FORCE, 'Isp_s': ISP,
            'g0_m_s2': G0, 'direction': 'instantaneous transverse direction',
            'planned_characteristic_speed_m_s': D0,
            'planned_duration_s': float(duration),
            'planned_fuel_kg': float(planned_fuel),
            'reference': 'fully completed finite burn (separate from impulsive delay target)',
            'orbits_compared': 'osculating orbit geometry at each cutoff, not simultaneous positions',
            'cutoffs': rows, 'numerical_checks': checks}


def orbit_xy(q, nu):
    radius = q['a_km']*(1-q['e']**2)/(1+q['e']*np.cos(nu))
    return radius*np.cos(nu+q['omega_rad']), radius*np.sin(nu+q['omega_rad'])


def save(fig, name):
    fig.savefig(BASE/(name+'.pdf'), bbox_inches='tight', pad_inches=.05)
    fig.savefig(PREVIEW/(name+'.png'), dpi=170, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def delay_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.35),
                             gridspec_kw={'width_ratios': [1.05, 1.]})
    d = data['delay']
    nu = np.linspace(0, 2*np.pi, 1000)
    ax = axes[0]
    for q, label, color, ls in [(d['initial'], 'Исходная орбита', GRAY, '--'),
           (d['target_apsidal_radii'], 'После точного манёвра', GREEN, '-')]:
        x, y = orbit_xy(q, nu)
        ax.plot(x, y, color=color, ls=ls, lw=1.8, label=label)
    ax.add_patch(Circle((0, 0), RE, facecolor='#E7F0F4', edgecolor='#A6BCC8'))
    ax.text(0, 0, 'Земля', ha='center', va='center', fontsize=11, color=GRAY)
    rp = np.array([-RA0, 0.])
    rlate = np.array(d['delay_samples'][-1]['late_position_km'])
    ax.plot(*rp, 'o', color=BLUE, ms=6)
    ax.plot(*rlate, 'o', color=ORANGE, ms=6)
    ax.annotate('По плану', xy=rp, xytext=(-13000, 6900),
                arrowprops={'arrowstyle': '-', 'color': BLUE}, color=BLUE, fontsize=12)
    ax.annotate('Через 15 мин', xy=rlate, xytext=(-13000, -9300),
                arrowprops={'arrowstyle': '-', 'color': ORANGE}, color=ORANGE, fontsize=12)
    rtrack = np.array([state(d['initial'], t)[0] for t in np.linspace(0, 900, 121)])
    ax.plot(rtrack[:, 0], rtrack[:, 1], color=ORANGE, lw=3)
    ax.set_aspect('equal')
    ax.set_xlim(-14000, 10000); ax.set_ylim(-11000, 11000)
    ax.set_xlabel('Координата вдоль линии апсид, км', fontsize=11.5)
    ax.set_ylabel('Координата в плоскости орбиты, км', fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.17), frameon=False, fontsize=11.5)
    ax = axes[1]
    t = np.array([r['delay_s']/60 for r in d['delay_samples']])
    cost = np.array([r['extra_characteristic_speed_m_s'] for r in d['delay_samples']])
    ax.plot(t, cost, color=ORANGE, lw=2.4)
    for minutes in (2., 10., 15.):
        row = d['delay_samples'][int(minutes*6)]
        ax.plot(minutes, row['extra_characteristic_speed_m_s'], 'o', color=ORANGE, ms=5)
    ax.set_xlim(0, 15.7); ax.set_ylim(0, 11.7)
    ax.set_xlabel('Задержка первого манёвра, мин')
    ax.set_ylabel('Добавочная характеристическая\nскорость, м/с', fontsize=12)
    ax.text(.04, .96, 'Две последующие коррекции\nразмеров орбиты',
            transform=ax.transAxes, va='top', fontsize=12, color=INK)
    ax.text(.04, .82, 'Цена выбранной схемы,\nне минимум по всем планам',
            transform=ax.transAxes, va='top', fontsize=10.8, color=GRAY)
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.86, wspace=.43)
    save(fig, 'overview-delay')


def missed_figure(data):
    d = data['delay']
    row = next(x for x in d['delay_samples'] if x['delay_s'] == 120.)
    p = d['initial']['period_s']/3600
    tperi = row['perigee_correction_time_s']/3600
    tf = row['completion_time_s']/3600
    fig, ax = plt.subplots(figsize=(10.6, 4.0))
    ys = [2., 1., 0.]
    ax.hlines(ys, 0, max(p, tf)+.12, colors='#C6CDD1', lw=1)
    ax.scatter([0], [2], s=55, color=BLUE, zorder=5)
    ax.text(.12, 2.16, '100 м/с; цель достигнута', color=BLUE, fontsize=12)
    for t, text, col, offset in [
        (120/3600, '100 м/с\nчерез 120 с', ORANGE, (.13, .18)),
        (tperi, f"{row['perigee_correction_m_s']:.3f} м/с\nв перигее".replace('.', ','), GREEN, (-.15, .18)),
        (tf, f"+{row['apogee_correction_m_s']:.3f} м/с\nв апогее".replace('.', ','), GREEN, (-.31, .18))]:
        ax.scatter([t], [1], s=55, color=col, zorder=5)
        ax.annotate(text, xy=(t, 1), xytext=(t+offset[0], 1+offset[1]),
                    textcoords='data', fontsize=11.5, color=col, ha='left',
                    arrowprops={'arrowstyle': '-', 'color': col, 'lw':.8})
    ax.scatter([0], [0], s=70, color=RED, marker='x', lw=2., zorder=5)
    ax.text(.12, .15, 'Манёвр пропущен', color=RED, fontsize=12)
    ax.annotate('', xy=(p, 0), xytext=(.04, 0),
                arrowprops={'arrowstyle': '-|>', 'color': GREEN, 'lw': 2.})
    ax.scatter([p], [0], s=55, color=GREEN, zorder=5)
    ax.text(p-.04, -.24, '100 м/с\nв следующем апогее',
            ha='right', va='top', fontsize=11.5, color=GREEN)
    ax.text(1.28, -.08, f'Ожидание {p:.3f} ч'.replace('.', ','), ha='center',
            va='top', color=GREEN, fontsize=12,
            bbox={'facecolor':'white','edgecolor':'none','pad':2})
    ax.set_yticks(ys, ['Номинальный план', 'Опоздание 120 с', 'Пропуск окна'])
    ax.set_xlim(-.04, max(p, tf)+.16)
    ax.set_ylim(-.83, 2.57)
    ax.set_xlabel('Время от запланированного манёвра, ч')
    ax.set_xticks(np.arange(0, 2.76, .5))
    ax.grid(axis='x', alpha=.16); ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    fig.subplots_adjust(left=.22, right=.985, bottom=.2, top=.96)
    save(fig, 'overview-missed')


def abort_figure(data):
    cutoffs = data['abort']['cutoffs']
    cols = [GRAY, BLUE, PURPLE, ORANGE, GREEN]
    fig = plt.figure(figsize=(10.5, 5.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.],
                           height_ratios=[1., 1.], wspace=.4, hspace=.18)
    ax = fig.add_subplot(gs[:, 0])
    nu = np.linspace(-.36, .36, 301)
    for r, col in zip(cutoffs, cols):
        q = r['orbit_after_stop']
        x, y = orbit_xy(q, nu)
        percentage = int(r['fraction_of_planned_duration']*100)
        ax.plot(x, y, lw=2, color=col, label=f'{percentage}% времени')
    ax.set_xlim(6800, 7700); ax.set_ylim(-1100, 1100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Координата вдоль\nначальной линии апсид, км', fontsize=11.8)
    ax.set_ylabel('Координата в плоскости орбиты, км', fontsize=11.8)
    ax.tick_params(labelsize=11)
    ax.set_title('Область перигея\nдостигнутых орбит', fontsize=13.5, pad=13)
    ax.legend(loc='center left', bbox_to_anchor=(1.02,.5),
              frameon=False, fontsize=10.8, handlelength=1.3)
    top = fig.add_subplot(gs[0, 1])
    bot = fig.add_subplot(gs[1, 1], sharex=top)
    frac = np.array([r['fraction_of_planned_duration']*100 for r in cutoffs])
    rp = np.array([r['orbit_after_stop']['rp_km'] for r in cutoffs])
    ra = np.array([r['orbit_after_stop']['ra_km']-RA0 for r in cutoffs])
    top.plot(frac, rp, color=GREEN, lw=1.7)
    bot.plot(frac, ra, color=ORANGE, lw=1.7)
    for x, yp, ya, col in zip(frac, rp, ra, cols):
        top.plot(x, yp, 'o', color=col, ms=6)
        bot.plot(x, ya, 'o', color=col, ms=6)
    top.set_ylabel('Радиус перигея, км', fontsize=12)
    bot.set_ylabel('Рост радиуса апогея, км', fontsize=12)
    bot.set_xlabel('Время работы до остановки, % плана', fontsize=12)
    top.tick_params(labelbottom=False)
    bot.set_xticks([0, 25, 50, 75, 100])
    top.set_xlim(-3, 103)
    top.set_ylim(6970, 7515)
    bot.set_ylim(min(-.03, -.03*max(ra)), 1.1*max(ra))
    top.set_title('Размеры орбиты после остановки', fontsize=13.5, pad=13)
    fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.16)
    save(fig, 'overview-abort')


def macros(data):
    def decimal(x, places):
        return f'{x:.{places}f}'.replace('.', r'{,}')
    delay, abort = data['delay'], data['abort']
    row120 = next(r for r in delay['delay_samples'] if r['delay_s'] == 120.)
    row600 = next(r for r in delay['delay_samples'] if r['delay_s'] == 600.)
    row900 = delay['delay_samples'][-1]
    half = abort['cutoffs'][2]
    full = abort['cutoffs'][-1]
    values = {
        'TimingInitialPeriod': (delay['initial']['period_s'], 3),
        'TimingInitialHours': (delay['initial']['period_s']/3600, 3),
        'TimingTargetRp': (delay['target_apsidal_radii']['rp_km'], 3),
        'TimingDelayShortCost': (row120['extra_characteristic_speed_m_s'], 3),
        'TimingDelayMediumCost': (row600['extra_characteristic_speed_m_s'], 3),
        'TimingDelayLongCost': (row900['extra_characteristic_speed_m_s'], 3),
        'TimingDelayShortPeriDV': (row120['perigee_correction_m_s'], 6),
        'TimingDelayShortApoDV': (row120['apogee_correction_m_s'], 6),
        'TimingDelayShortFinalTime': (row120['completion_time_s'], 3),
        'TimingBurnDuration': (abort['planned_duration_s'], 3),
        'TimingAbortHalfDV': (half['characteristic_speed_m_s'], 3),
        'TimingAbortHalfRp': (half['orbit_after_stop']['rp_km'], 3),
        'TimingAbortFullRp': (full['orbit_after_stop']['rp_km'], 3),
        'TimingAbortFullRa': (full['orbit_after_stop']['ra_km'], 3),
        'TimingAbortFullOmega': (full['orbit_after_stop']['omega_deg'], 3),
        'TimingDelayLongOmega': (row900['final_omega_deg'], 3)}
    lines = ['% Generated by overview_timing.py; do not edit numeric values manually.']
    lines += [rf'\newcommand{{\{k}}}{{{decimal(v,n)}}}' for k,(v,n) in values.items()]
    (BASE/'timing-values.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    delay = calculate_delay()
    abort = calculate_abort(delay['initial'])
    data = {'units': 'km, s, kg, N; impulses explicitly m/s',
        'mu_km3_s2': MU, 'earth_radius_km': RE,
        'purpose': 'Teaching examples; no real spacecraft performance assumed.',
        'delay': delay, 'abort': abort}
    (BASE/'timing-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n',
                                       encoding='utf-8')
    macros(data)
    delay_figure(data)
    missed_figure(data)
    abort_figure(data)
    print('Delay120 extra J:', delay['delay_samples'][12]['extra_characteristic_speed_m_s'])
    print('Finite duration:', abort['planned_duration_s'])
    print('Abort at half time:', abort['cutoffs'][2]['characteristic_speed_m_s'])
    print('All analytic/numerical consistency checks passed.')


if __name__ == '__main__':
    main()
