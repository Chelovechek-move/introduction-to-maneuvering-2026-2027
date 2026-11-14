#!/usr/bin/env python3
"""Independent numerical verification of the lecture 9 example.

Run from the course root:
    python3 9/images/verify_example.py

Input:  9/images/example-results.json
Output: 9/images/independent-verification.json

This script does not import avoidance_example.py. Gaussian disk probabilities
are integrated directly in polar coordinates with nested adaptive quadrature.
Two-body trajectories are reconstructed by the universal-variable f,g map,
independently of the example's classical-element propagation. The Cartesian
f,g calculation is a verification method, not the propagation model used by
the lecture's main example. Position and velocity are in km and km/s inside
the map; encounter geometry uses m and the RTN impulse input uses m/s.
"""

from pathlib import Path
import json

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


BASE = Path(__file__).resolve().parent


def stumpff(z):
    """C(z), S(z), with cancellation-free series near zero."""
    if abs(z) < 1e-5:
        return (.5-z/24+z*z/720-z**3/40320,
                1/6-z/120+z*z/5040-z**3/362880)
    if z > 0:
        x = np.sqrt(z)
        return 2*np.sin(x/2)**2/z, (x-np.sin(x))/x**3
    x = np.sqrt(-z)
    return (np.cosh(x)-1)/(-z), (np.sinh(x)-x)/x**3


def universal_fg(r0, v0, dt, mu):
    """Exact two-body map via the universal-variable Kepler equation."""
    if abs(dt) < 1e-15:
        return r0.copy(), v0.copy()
    sqmu = np.sqrt(mu)
    r0n = np.linalg.norm(r0)
    rv = r0 @ v0
    alpha = 2/r0n-v0 @ v0/mu

    def equation(chi):
        c, s = stumpff(alpha*chi*chi)
        return (rv/sqmu*chi*chi*c+(1-alpha*r0n)*chi**3*s
                + r0n*chi-sqmu*dt)

    bound = max(1., abs(sqmu*alpha*dt)*2)
    while equation(bound) < 0 or equation(-bound) > 0:
        bound *= 2
    chi = brentq(equation, -bound, bound, xtol=1e-12)
    c, s = stumpff(alpha*chi*chi)
    f = 1-chi*chi/r0n*c
    g = dt-chi**3/sqmu*s
    r = f*r0+g*v0
    rn = np.linalg.norm(r)
    fdot = sqmu/(rn*r0n)*(alpha*chi**3*s-chi)
    gdot = 1-chi*chi/rn*c
    return r, fdot*r0+gdot*v0


def disk_probability_polar(b, covariance, radius):
    """Direct 2D Gaussian disk integral in polar coordinates.

    Unlike the main script, this method neither diagonalizes the covariance
    nor analytically integrates one Gaussian coordinate.
    """
    b = np.asarray(b)
    inv = np.linalg.inv(covariance)
    denominator = 2*np.pi*np.sqrt(np.linalg.det(covariance))

    def radial(r):
        def angular(theta):
            z = r*np.array([np.cos(theta), np.sin(theta)])-b
            return np.exp(-.5*z @ inv @ z)
        return r*quad(angular, 0, 2*np.pi, epsabs=2e-13,
                      epsrel=2e-12, limit=120)[0]/denominator

    return quad(radial, 0, radius, epsabs=2e-16,
                epsrel=2e-12, limit=120)[0]


def verify(data):
    mu = data['mu_km3_s2']
    ra = np.array(data['primary_R_tca_km'])
    va = np.array(data['primary_V_tca_km_s'])
    rb = np.array(data['secondary_R_tca_km'])
    vb = np.array(data['secondary_V_tca_km_s'])
    projection = np.array(data['encounter_projection_rows'])
    covariance = np.array(data['C_b_m2'])
    radius = data['HBR_m']
    lead = data['main_lead_s']
    b0 = np.array(data['nominal_b_m'])

    rm, vm = universal_fg(ra, va, -lead, mu)
    er = rm/np.linalg.norm(rm)
    en = np.cross(rm, vm)
    en /= np.linalg.norm(en)
    et = np.cross(en, er)
    rtn = np.column_stack([er, et, en])

    def managed(dv, t=0.):
        return universal_fg(rm, vm+rtn @ np.asarray(dv)/1000., lead+t, mu)

    probability_checks = []
    cases = [('nominal', b0, data['nominal_pc'])]
    cases.extend((key, value['b_linear_m'], value['pc_linear'])
                 for key, value in data['solutions'].items())
    for name, b, pc_primary in cases:
        pc = disk_probability_polar(b, covariance, radius)
        probability_checks.append({
            'name': name, 'pc_primary': pc_primary, 'pc_polar': pc,
            'relative_difference': abs(pc-pc_primary)/pc})

    propagation_checks = []
    for key, solution in data['solutions'].items():
        dv = solution['dv_rtn_m_s']

        def stationary(t):
            r1, v1 = managed(dv, t)
            r2, v2 = universal_fg(rb, vb, t, mu)
            return (r2-r1) @ (v2-v1)

        tca = brentq(stationary, -30., 30., xtol=1e-11)
        r1, v1 = managed(dv, tca)
        r2, v2 = universal_fg(rb, vb, tca, mu)
        rho = (r2-r1)*1000.
        relative_velocity = (v2-v1)*1000.
        wh = relative_velocity/np.linalg.norm(relative_velocity)
        e1 = projection[0]-wh*(wh @ projection[0])
        e1 /= np.linalg.norm(e1)
        enew = np.vstack([e1, np.cross(wh, e1)])
        b = enew @ rho
        cnew = enew @ np.array(data['C_relative_IJK_m2']) @ enew.T
        pc = disk_probability_polar(b, cnew, radius)
        reference = solution['nonlinear']
        propagation_checks.append({
            'name': key, 'tca_s': float(tca),
            'difference_tca_s': float(tca-reference['tca_shift_s']),
            'miss_m': float(np.linalg.norm(rho)),
            'difference_miss_m': float(np.linalg.norm(rho)-reference['miss_m']),
            'b_m': b.tolist(),
            'difference_b_m': (b-np.array(reference['b_m'])).tolist(),
            'pc_polar_nonlinear': float(pc),
            'relative_difference_pc_nonlinear': float(abs(pc-reference['pc'])/pc)})

    step = .002
    g = np.column_stack([
        (projection @ (rb-managed(np.eye(3)[j]*step)[0])*1000.
         - projection @ (rb-managed(-np.eye(3)[j]*step)[0])*1000.)/(2*step)
        for j in range(3)])
    g_reference = np.array(data['G_main_s'])
    rback, vback = universal_fg(*universal_fg(ra, va, -lead, mu), lead, mu)

    execution_checks = []
    for key in ['minimum_pc_impulse', 'execution_aware_impulse']:
        reference = data['solutions'][key]['nonlinear']
        pc = disk_probability_polar(
            reference['b_m'],
            np.array(reference['C_b_with_execution_error_m2']), radius)
        pc_reference = reference['pc_with_execution_error']
        execution_checks.append({
            'name': key, 'pc_with_execution_error_polar': float(pc),
            'pc_primary': pc_reference,
            'relative_difference': float(abs(pc-pc_reference)/pc)})

    return {
        'description': 'Independent audit: Gaussian disk probability by polar '
        'double adaptive quadrature and exact two-body propagation by '
        'universal-variable f,g map; no use of example propagation or probability code.',
        'date': '2026-11-13',
        'input': 'example-results.json',
        'probability_checks': probability_checks,
        'probability_max_relative_difference': max(
            z['relative_difference'] for z in probability_checks),
        'propagation_checks': propagation_checks,
        'universal_fg_round_trip_position_error_m': float(np.linalg.norm(rback-ra)*1000.),
        'universal_fg_round_trip_velocity_error_m_s': float(np.linalg.norm(vback-va)*1000.),
        'sensitivity_step_m_s': step,
        'G_universal_fg_s': g.tolist(),
        'G_relative_difference': float(np.linalg.norm(g-g_reference)/np.linalg.norm(g_reference)),
        'G_absolute_difference_max_s': float(np.max(abs(g-g_reference))),
        'projection_orthonormal_error': float(np.linalg.norm(projection @ projection.T-np.eye(2))),
        'projection_velocity_error_m_s': float(np.linalg.norm(projection @ (vb-va))*1000.),
        'execution_covariance_probability_checks': execution_checks,
        'reproduction': {
            'command': 'python3 9/images/verify_example.py',
            'probability': 'Direct polar disk integration with nested scipy.integrate.quad; '
            'epsrel=2e-12, radial epsabs=2e-16, angular epsabs=2e-13.',
            'propagation': 'Universal-variable Kepler equation solved by brentq; '
            'reconstruct position/velocity with f,g,fdot,gdot; rebuild primary RTN '
            'at -lead, apply the exact velocity jump, propagate both objects and '
            'refind local TCA on rho dot relative_velocity.'}}


def main():
    data = json.loads((BASE/'example-results.json').read_text())
    result = verify(data)
    (BASE/'independent-verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({
        'probability_max_relative_difference': result['probability_max_relative_difference'],
        'G_relative_difference': result['G_relative_difference'],
        'max_nonlinear_miss_difference_m': max(
            abs(z['difference_miss_m']) for z in result['propagation_checks'])}, indent=2))


if __name__ == '__main__':
    main()
