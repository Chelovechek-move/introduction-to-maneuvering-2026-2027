#!/usr/bin/env python3
"""Reproducible short-encounter collision-avoidance example for lecture 9.

Run from the course root: python3 9/images/avoidance_example.py
Redraw existing data:    python3 9/images/avoidance_example.py --figures-only

Passive propagation is solely in classical elements
(a [km], e, omega, i, Omega, M [rad]); Kepler's equation is solved at output
epochs. Cartesian coordinates are used only for encounter geometry, velocity
jumps, and conversion back to elements. No Cartesian trajectory integration.
Risk is the Gaussian integral over the HBR disk, evaluated by convergent
Gauss-Legendre quadrature, not a Mahalanobis approximation.
"""
from pathlib import Path
import argparse
import json
import os
import time

BASE = Path(__file__).resolve().parent
BUILD = BASE.parent / 'build'
BUILD.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BUILD / 'mplconfig'))
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

MU = 398600.4418  # km^3/s^2
RE = 6378.1363    # km
HBR = 10.0       # m, sum of circumscribed object radii
PC_LIMIT = 1e-5
Q0 = np.array([7000., .01, *np.deg2rad([40., 51., 20.]), 0.])
NU0 = np.deg2rad(60.)
E0 = 2*np.arctan2(np.sqrt(1-Q0[1])*np.sin(NU0/2),
                  np.sqrt(1+Q0[1])*np.cos(NU0/2))
Q0[5] = E0-Q0[1]*np.sin(E0)
PERIOD = 2*np.pi*np.sqrt(Q0[0]**3/MU)
LEAD_MAIN = .4*PERIOD
CROSSING = np.deg2rad(60.)
B0 = np.array([25., -15.])  # m, secondary minus primary
SIGMA = np.array([40., 200.])
COV_ANGLE = np.deg2rad(35.)  # orientation of the 40 m principal axis
ROT = np.array([[np.cos(COV_ANGLE), -np.sin(COV_ANGLE)],
                [np.sin(COV_ANGLE), np.cos(COV_ANGLE)]])
CB = ROT @ np.diag(SIGMA**2) @ ROT.T
EXECUTION_SIGMA = .005  # m/s, independent, isotropic execution error
FIXED_DV = .10         # m/s for equal-cost comparison
PALETTE = ['#187AA5', '#087F78', '#B57714', '#79539C', '#C44847']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':13.5,
 'axes.labelsize':13.,'axes.titlesize':14.5,'legend.fontsize':11.5,
 'xtick.labelsize':11.5,'ytick.labelsize':11.5,
 'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,
 'pdf.fonttype':42,'ps.fonttype':42})


def rot1(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])


def rot3(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def kepler(q, dt=0.):
    """Exact two-body passive propagation in classical orbital elements."""
    z = q.copy()
    z[5] += np.sqrt(MU/q[0]**3)*dt
    return z


def cart(q):
    a,e,w,inc,node,M = q
    mm = (M+np.pi)%(2*np.pi)-np.pi
    ee = mm
    for _ in range(12):
        delta = (ee-e*np.sin(ee)-mm)/(1-e*np.cos(ee))
        ee -= delta
        if abs(delta)<2e-15:
            break
    co,si = np.cos(ee),np.sin(ee)
    scale = np.sqrt(1-e*e)
    r = np.array([a*(co-e),a*scale*si,0.])
    v = np.sqrt(MU/a)/(1-e*co)*np.array([-si,scale*co,0.])
    rr = rot3(node)@rot1(inc)@rot3(w)
    return rr@r,rr@v


def elements(r,v):
    rn = np.linalg.norm(r)
    hv = np.cross(r,v); hn = np.linalg.norm(hv)
    nv = np.cross([0.,0.,1.],hv); nn = np.linalg.norm(nv)
    ev = np.cross(v,hv)/MU-r/rn; e = np.linalg.norm(ev)
    a = 1/(2/rn-v@v/MU)
    inc = np.arctan2(np.linalg.norm(hv[:2]),hv[2])
    node = np.arctan2(nv[1],nv[0])
    w = np.arctan2(np.dot(np.cross(nv,ev),hv)/(nn*e*hn),np.dot(nv,ev)/(nn*e))
    nu = np.arctan2(np.dot(np.cross(ev,r),hv)/(e*rn*hn),np.dot(ev,r)/(e*rn))
    ee = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2),np.sqrt(1+e)*np.cos(nu/2))
    mm = ee-e*np.sin(ee)
    if not (a>0 and 1e-7<e<.95 and 1e-6<inc<np.pi-1e-6):
        raise ValueError('State left selected nondegenerate elliptic family')
    return np.array([a,e,w,inc,node,mm])


def rtn(q):
    r,v = cart(q)
    er = r/np.linalg.norm(r); en = np.cross(r,v); en /= np.linalg.norm(en)
    et = np.cross(en,er)
    return np.column_stack([er,et,en])


R0,V0 = cart(Q0)
RTN0 = rtn(Q0)
VB0 = RTN0@np.array([RTN0[:,0]@V0,
                     np.cos(CROSSING)*(RTN0[:,1]@V0),
                     np.sin(CROSSING)*(RTN0[:,1]@V0)])
W0 = VB0-V0
WH0 = W0/np.linalg.norm(W0)
EPLANE = np.vstack([RTN0[:,0],np.cross(WH0,RTN0[:,0])])
RB0 = R0+EPLANE.T@B0/1000.
QB0 = elements(RB0,VB0)
# This is a deliberately specified local covariance at the encounter, not a
# covariance propagated from an orbit determination or a real CDM.
CREL = EPLANE.T@CB@EPLANE + 300.**2*np.outer(WH0,WH0)


class DiskProbability:
    """2-D Gaussian disk integral, after covariance principal-axis rotation.

    x=R sin(theta) makes the disk's square-root boundary smooth. Integrate
    the independent y-normal analytically, x-normal by Gauss-Legendre.
    """
    def __init__(self,cov,order=64,radius=HBR):
        self.cov = np.asarray(cov)
        eig,self.axes = np.linalg.eigh(cov)
        self.sig = np.sqrt(eig)
        u,w = leggauss(order)
        angle = u*np.pi/2
        self.x = radius*np.sin(angle)
        self.y = radius*np.cos(angle)
        self.weight = w*np.pi/2*radius*np.cos(angle)

    def __call__(self,b):
        bb = np.asarray(b)@self.axes
        bx,by = bb[...,0,None],bb[...,1,None]
        lo = (-self.y-by)/self.sig[1]
        hi = ( self.y-by)/self.sig[1]
        # Stable normal-CDF difference also far into the positive tail.
        yy = np.where(lo>0,ndtr(-lo)-ndtr(-hi),ndtr(hi)-ndtr(lo))
        xx = np.exp(-.5*((self.x-bx)/self.sig[0])**2)/(np.sqrt(2*np.pi)*self.sig[0])
        return np.sum(xx*yy*self.weight,axis=-1)


PC = DiskProbability(CB)


def jump(lead,dv):
    """Exact velocity jump; dv is in primary's instantaneous RTN, m/s."""
    qm = kepler(Q0,-lead)
    rm,vm = cart(qm)
    return elements(rm,vm+rtn(qm)@np.asarray(dv)/1000.)


def after_jump(lead,dv,t=0.):
    return cart(kepler(jump(lead,dv),lead+t))


def frozen_b(lead,dv):
    ra,_ = after_jump(lead,dv)
    return EPLANE@(RB0-ra)*1000.


def sensitivity(lead,step=.001):
    # The matrix maps the primary's RTN impulse in m/s into relative b in m.
    return np.column_stack([(frozen_b(lead,np.eye(3)[j]*step)-
                             frozen_b(lead,-np.eye(3)[j]*step))/(2*step)
                            for j in range(3)])


def first_axis_solution(g,prob=PC,limit=PC_LIMIT):
    """Find first safe crossing on either ray of a fixed axis."""
    candidates=[]
    for sign in [-1.,1.]:
        def f(a):
            return float(prob(B0+sign*a*g))-limit
        if f(0)<=0:
            candidates.append((0.,0.))
            continue
        # With a convex disk and Gaussian density, the superlevel sets of
        # disk probability versus mean position are convex. Along a ray
        # starting inside, the first outward crossing is unique.
        hi=.001
        while f(hi)>0 and hi<10.:
            hi*=1.5
        if f(hi)<=0:
            root=brentq(f,0.,hi,xtol=2e-12)
            candidates.append((root,sign*root))
    return min(candidates)[1] if candidates else np.nan


def fibonacci_sphere(n=32):
    jj=np.arange(n)
    zz=1-2*(jj+.5)/n
    angle=jj*np.pi*(3-np.sqrt(5))
    rr=np.sqrt(1-zz*zz)
    return np.column_stack([rr*np.cos(angle),rr*np.sin(angle),zz])


def minimum_impulse(G,prob=PC,starts=18,limit=PC_LIMIT):
    """Local SLSQP candidates with several independent feasible starts.

    Optimized quantities are RTN impulse components in m/s. Risk constraint
    uses log(Pc/Pc_limit) for conditioning. Results are not global certificates.
    """
    dirs=np.vstack([np.eye(3),-np.eye(3),fibonacci_sphere(starts)])
    seeds=[]
    for direction in dirs:
        # The selected signed ray may be opposite the supplied direction.
        amp=first_axis_solution(G@direction,prob,limit)
        if np.isfinite(amp):
            seeds.append(direction*amp*1.001)
    def risk(x):
        return -np.log(max(float(prob(B0+G@x)),1e-300)/limit)
    best=None; records=[]
    for seed in seeds:
        out=minimize(lambda x:.5*x@x,seed,method='SLSQP',
              constraints=[{'type':'ineq','fun':risk}],
              options={'ftol':1e-13,'maxiter':180,'disp':False})
        val=float(prob(B0+G@out.x))
        ok=out.success and val<=limit*(1+2e-6)
        records.append({'success':bool(out.success),'valid':bool(ok),
             'norm_m_s':float(np.linalg.norm(out.x)), 'pc':val,
             'dv_rtn_m_s':out.x.tolist(),'iterations':int(out.nit)})
        if ok and (best is None or np.linalg.norm(out.x)<np.linalg.norm(best)):
            best=out.x.copy()
    if best is None:
        raise RuntimeError('No feasible SLSQP candidate')
    return best,records


def fixed_cost(G,criterion,dvnorm=FIXED_DV):
    """Multi-start optimization on the impulse sphere, for b0 != 0."""
    sols=[]
    for direction in np.vstack([np.eye(3),-np.eye(3),fibonacci_sphere(24)]):
        out=minimize(lambda u:criterion(B0+G@(dvnorm*u)),direction,
          method='SLSQP',constraints=[{'type':'eq','fun':lambda u:u@u-1}],
          options={'ftol':1e-12,'maxiter':180})
        if out.success and abs(out.x@out.x-1)<1e-7:
            sols.append((float(out.fun),dvnorm*out.x/np.linalg.norm(out.x)))
    if not sols:
        raise RuntimeError('No candidate on impulse sphere')
    return min(sols,key=lambda z:z[0])[1]


def tca_for_jump(lead,dv):
    qm=jump(lead,dv)
    def states(t):
        ra,va=cart(kepler(qm,lead+t)); rb,vb=cart(kepler(QB0,t))
        return ra,va,rb,vb
    def stationary(t):
        ra,va,rb,vb=states(t)
        return (rb-ra)@(vb-va)
    tt=brentq(stationary,-30.,30.,xtol=2e-11)
    ra,va,rb,vb=states(tt)
    rho=(rb-ra)*1000.; vel=(vb-va)*1000.
    wh=vel/np.linalg.norm(vel)
    # Parallel transport the nominal first encounter-plane axis by projection.
    e1=EPLANE[0]-wh*(wh@EPLANE[0]);e1/=np.linalg.norm(e1)
    enew=np.vstack([e1,np.cross(wh,e1)])
    cov=enew@CREL@enew.T
    b=enew@rho
    step=.001
    K=np.column_stack([-(after_jump(lead,dv+step*np.eye(3)[j],tt)[0]-
                         after_jump(lead,dv-step*np.eye(3)[j],tt)[0])*1000/(2*step)
                       for j in range(3)])
    cov_exec=cov+EXECUTION_SIGMA**2*(enew@K)@(enew@K).T
    return {'tca_shift_s':float(tt),'miss_m':float(np.linalg.norm(rho)),
       'b_m':b.tolist(),'pc':float(DiskProbability(cov)(b)),
       'pc_with_execution_error':float(DiskProbability(cov_exec)(b)),
       'C_b_with_execution_error_m2':cov_exec.tolist(),
       'C_b_m2':cov.tolist(),'relative_speed_m_s':float(np.linalg.norm(vel)),
       'radial_stationarity_m2_s':float(rho@vel)}


def jsonable(obj):
    if isinstance(obj,np.ndarray):return jsonable(obj.tolist())
    if isinstance(obj,np.generic):return jsonable(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    if isinstance(obj,dict):return {k:jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [jsonable(v) for v in obj]
    return obj


def solution_record(G,dv,lead):
    b=B0+G@dv
    return {'dv_rtn_m_s':dv,'dv_norm_m_s':np.linalg.norm(dv),
       'b_linear_m':b,'miss_linear_m':np.linalg.norm(b),
       'pc_linear':float(PC(b)),'nonlinear':tca_for_jump(lead,dv)}


def compute():
    started=time.time()
    G=sensitivity(LEAD_MAIN)
    axis=[]
    for j in range(3):
        amp=first_axis_solution(G[:,j]);dv=np.eye(3)[j]*amp
        axis.append(solution_record(G,dv,LEAD_MAIN))
    opt,logs=minimum_impulse(G)
    dv_geo=fixed_cost(G,lambda b:-b@b/1e6)
    dv_pc=fixed_cost(G,lambda b:np.log(max(float(PC(b)),1e-300)))
    cexec=CB+EXECUTION_SIGMA**2*G@G.T
    pexec=DiskProbability(cexec)
    design_limit=.8*PC_LIMIT
    safeopt,safelogs=minimum_impulse(G,pexec,limit=design_limit)
    records={'minimum_pc_impulse':solution_record(G,opt,LEAD_MAIN),
       'maximum_miss_fixed_cost':solution_record(G,dv_geo,LEAD_MAIN),
       'minimum_pc_fixed_cost':solution_record(G,dv_pc,LEAD_MAIN),
       'execution_aware_impulse':solution_record(G,safeopt,LEAD_MAIN)}
    records['minimum_pc_impulse']['pc_with_execution_error']=float(pexec(B0+G@opt))
    records['execution_aware_impulse']['pc_with_execution_error']=float(pexec(B0+G@safeopt))
    qsafe=kepler(jump(LEAD_MAIN,safeopt),LEAD_MAIN)
    element_change=qsafe-Q0
    element_change[2:]=np.arctan2(np.sin(element_change[2:]),np.cos(element_change[2:]))
    records['execution_aware_impulse']['elements_at_original_tca']=qsafe
    records['execution_aware_impulse']['orbital_element_changes']={
       'delta_a_m':element_change[0]*1000.,'delta_e':element_change[1],
       'delta_omega_deg':np.rad2deg(element_change[2]),
       'delta_i_deg':np.rad2deg(element_change[3]),
       'delta_Omega_deg':np.rad2deg(element_change[4]),
       'delta_M_deg_at_original_tca':np.rad2deg(element_change[5])}
    leads=np.linspace(.1,2.,97)*PERIOD
    gs=np.array([sensitivity(t) for t in leads])
    axisamps=np.array([[first_axis_solution(g[:,j]) for j in range(3)] for g in gs])
    selected=[]
    for frac in [.25,.4,.5,1.,2.]:
        lead=frac*PERIOD; gg=sensitivity(lead)
        oo,ll=minimum_impulse(gg,starts=10)
        selected.append({'lead_periods':frac,'lead_s':lead,'G_s':gg,
          'best':solution_record(gg,oo,lead),
          'axis_signed_m_s':[first_axis_solution(gg[:,j]) for j in range(3)],
          'optimization':ll})
    amplitudes=np.linspace(-.18,.18,241)
    pp=np.array([PC(B0+amplitudes[:,None]*g[:,1]) for g in gs])
    deviations=np.max([np.linalg.norm(sensitivity(LEAD_MAIN,st)-G)/np.linalg.norm(G)
                       for st in [.0005,.002]],axis=0)
    test_b=np.vstack([B0,*[np.asarray(v['b_linear_m']) for v in records.values()]])
    p64=PC(test_b);p128=DiskProbability(CB,128)(test_b)
    # Independent adaptive quadrature in the same smooth disk coordinates.
    eig,U=np.linalg.eigh(CB);sig=np.sqrt(eig)
    quad_repeat=[]
    for b in test_b:
        bp=np.asarray(b)@U
        def f(theta):
            x=HBR*np.sin(theta);y=HBR*np.cos(theta)
            low=(-y-bp[1])/sig[1];high=(y-bp[1])/sig[1]
            diff=ndtr(-low)-ndtr(-high) if low>0 else ndtr(high)-ndtr(low)
            return np.exp(-.5*((x-bp[0])/sig[0])**2)/(np.sqrt(2*np.pi)*sig[0])*diff*HBR*np.cos(theta)
        quad_repeat.append(quad(f,-np.pi/2,np.pi/2,epsabs=1e-17,epsrel=2e-12)[0])
    quad_repeat=np.array(quad_repeat)
    # Independent secular-equation solution for maximum affine miss on sphere.
    H=G.T@G;cc=G.T@B0;ev,U=np.linalg.eigh(H);cv=U.T@cc
    def secular(lam):return np.sum((cv/(lam-ev))**2)-FIXED_DV**2
    lam=brentq(secular,ev[-1]*(1+1e-12),ev[-1]*20,xtol=1e-7)
    dvgeo_check=U@(cv/(lam-ev))
    result={'description':'Synthetic spatial short encounter, two-body classical-element propagation',
      'software_versions':{'numpy':np.__version__,'scipy':scipy.__version__,
                           'matplotlib':matplotlib.__version__},
      'units':{'orbits':'a km; angles rad','impulses':'m/s in instantaneous primary RTN',
               'b':'m, secondary minus primary','G':'s, mapping m/s to m'},
      'mu_km3_s2':MU,'earth_radius_km':RE,'primary_elements_at_tca':Q0,
      'secondary_elements_at_tca':QB0,'primary_true_anomaly_deg':60.,
      'nominal_tca_epoch_s':0.,'primary_period_s':PERIOD,
      'main_lead_s':LEAD_MAIN,'main_lead_periods':.4,
      'nominal_b_m':B0,'nominal_miss_m':np.linalg.norm(B0),
      'nominal_pc':float(PC(B0)),'nominal_relative_speed_m_s':np.linalg.norm(W0)*1000.,
      'tangential_rotation_deg':60.,'HBR_m':HBR,'pc_limit':PC_LIMIT,
      'C_b_m2':CB,'covariance_principal_sigmas_m':SIGMA,
      'covariance_small_axis_angle_deg':np.rad2deg(COV_ANGLE),
      'C_relative_IJK_m2':CREL,'encounter_projection_rows':EPLANE,
      'primary_R_tca_km':R0,'primary_V_tca_km_s':V0,
      'secondary_R_tca_km':RB0,'secondary_V_tca_km_s':VB0,
      'G_main_s':G,'fixed_cost_dv_m_s':FIXED_DV,'axis_solutions':axis,
      'solutions':records,'lead_time_comparison':selected,
      'execution_error_sigma_m_s':EXECUTION_SIGMA,
      'execution_aware_design_pc_target':design_limit,
      'C_b_with_execution_error_m2':cexec,
      'verification':{'finite_difference_step_m_s':.001,
         'relative_G_change_steps_half_double':float(deviations),
         'pc_quadrature_order':64,'pc_quadrature_repeat_order':128,
         'max_pc_relative_change':float(np.max(np.abs(p128-p64)/np.maximum(p128,1e-280))),
         'max_pc_relative_change_adaptive_quad':float(np.max(np.abs(quad_repeat-p64)/np.maximum(quad_repeat,1e-280))),
         'maximum_miss_independent_secular_dv_difference_m_s':float(np.linalg.norm(dvgeo_check-dv_geo)),
         'maximum_miss_independent_secular_norm_difference_m':float(abs(np.linalg.norm(B0+G@dvgeo_check)-np.linalg.norm(B0+G@dv_geo))),
         'nominal_tca':tca_for_jump(LEAD_MAIN,np.zeros(3)),
         'SLSQP_min_impulse_runs':logs,'SLSQP_execution_aware_runs':safelogs},
      'limitations':['Covariance is specified at encounter, not propagated from orbit determination.',
         'Main synthesis uses frozen nominal encounter plane and fixed covariance.',
         'Nonlinear verification refinds local TCA and projects fixed 3D encounter covariance into new plane.',
         'SLSQP multistart provides candidates, not a global-optimality certificate.',
         'No third-object catalogue screening or operational flight-safety decision is performed.',
         'Two-body propagation; J2 and other perturbations are not part of this illustrative case.'],
      'runtime_s':time.time()-started}
    (BASE/'example-results.json').write_text(json.dumps(jsonable(result),ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(BASE/'example-data.npz',lead_s=leads,G=gs,
          axis_signed_dv_m_s=axisamps,map_amplitudes_m_s=amplitudes,map_pc=pp)
    write_tex(result)
    print(json.dumps({'nominal_pc':result['nominal_pc'],'G':G.tolist(),
       'opt_dv_m_s':opt.tolist(),'opt_norm':float(np.linalg.norm(opt)),
       'fixed_geo_pc':records['maximum_miss_fixed_cost']['pc_linear'],
       'fixed_risk_pc':records['minimum_pc_fixed_cost']['pc_linear'],
       'runtime_s':result['runtime_s']},indent=2))
    return jsonable(result)


def texnum(value,digits=4):
    if round(value,digits)==0.:value=0.
    return f'{value:.{digits}f}'.replace('.',r'{,}')


def scientific(value,digits=3):
    if value==0:return '0'
    exponent=int(np.floor(np.log10(abs(value))))
    mantissa=round(value/10**exponent,digits-1)
    if abs(mantissa)>=10.:
        mantissa/=10.;exponent+=1
    return texnum(mantissa,digits-1)+r'\cdot10^{'+str(exponent)+'}'


def write_tex(r):
    sol=r['solutions']; opt=sol['minimum_pc_impulse'];safe=sol['execution_aware_impulse']
    row_end=chr(92)*2+'\n'
    vals={'ExamplePeriod':texnum(PERIOD,3),'ExampleLead':texnum(LEAD_MAIN,3),
      'ExampleNominalMiss':texnum(r['nominal_miss_m'],3),
      'ExampleNominalPc':scientific(r['nominal_pc']),
      'ExampleRelativeSpeed':texnum(r['nominal_relative_speed_m_s']/1000,4),
      'ExampleOptimalDV':texnum(opt['dv_norm_m_s'],6),
      'ExampleOptimalMiss':texnum(opt['miss_linear_m'],3),
      'ExampleOptimalNonlinearPc':scientific(opt['nonlinear']['pc']),
      'ExampleOptimalNonlinearMiss':texnum(opt['nonlinear']['miss_m'],3),
      'ExampleOptimalTCA':texnum(opt['nonlinear']['tca_shift_s'],6),
      'ExampleExecutionPc':scientific(opt['pc_with_execution_error']),
      'ExampleSafeDV':texnum(safe['dv_norm_m_s'],6),
      'ExampleSafePc':scientific(safe['pc_with_execution_error']),
      'ExampleSafeNonlinearPc':scientific(safe['nonlinear']['pc_with_execution_error']),
      'ExampleSafeNonlinearMiss':texnum(safe['nonlinear']['miss_m'],3),
      'ExampleSafeTCA':texnum(safe['nonlinear']['tca_shift_s'],6),
      'ExampleGRepeat':scientific(r['verification']['relative_G_change_steps_half_double']),
      'ExamplePcRepeat':scientific(r['verification']['max_pc_relative_change'])}
    (BASE/'example-values.tex').write_text('% Generated by avoidance_example.py\n'+''.join(
      '\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    rows=[]
    for name,z in zip(['Радиальное','Трансверсальное','Нормальное'],r['axis_solutions']):
        signed=z['dv_rtn_m_s'][['Радиальное','Трансверсальное','Нормальное'].index(name)]
        rows.append(f'{name}&${texnum(signed,6)}$&${texnum(z["miss_linear_m"],2)}$'+row_end)
    (BASE/'example-axis-table.tex').write_text(
      '\\begin{tabular}{@{}lrr@{}}\\toprule\n'
      'Направление&Компонента импульса, м/с&Промах, м\\\\\\midrule\n'
      +''.join(rows)+'\\bottomrule\\end{tabular}\n')
    rows=[]
    for name,key in [('Максимальный промах','maximum_miss_fixed_cost'),
                     ('Минимальная вероятность','minimum_pc_fixed_cost'),
                     ('Минимальный импульс','minimum_pc_impulse')]:
        z=sol[key]
        rows.append(name+f'&${texnum(z["dv_norm_m_s"],6)}$&${texnum(z["miss_linear_m"],2)}$&${scientific(z["pc_linear"])}$'+row_end)
    (BASE/'example-comparison-table.tex').write_text(
      '\\begin{tabular}{@{}lrrr@{}}\\toprule\n'
      'Задача&$\\Delta V$, м/с&Промах, м&$P_c$\\\\\\midrule\n'
      +''.join(rows)+'\\bottomrule\\end{tabular}\n')
    rows=[]
    for name,q in [('Управляемый КА',Q0),('Опасный объект',QB0)]:
        values=[texnum(q[0],6),texnum(q[1],9),
                 *[texnum(np.rad2deg(x) % 360,6) for x in q[2:]]]
        rows.append(name+'&'+'&'.join('$'+z+'$' for z in values)+row_end)
    (BASE/'example-elements-table.tex').write_text(''.join(rows))
    matrix=lambda a: r'\begin{pmatrix}'+r'\\'.join('&'.join(texnum(v,3) for v in row) for row in a)+r'\end{pmatrix}'
    with (BASE/'example-values.tex').open('a') as f:
        f.write('\\newcommand{\\ExampleMatrixG}{'+matrix(r['G_main_s'])+'}\n')
        f.write('\\newcommand{\\ExampleCovarianceMatrix}{'+matrix(CB)+'}\n')
    vec=r'\begin{pmatrix}'+r'\\'.join(texnum(v,6) for v in safe['dv_rtn_m_s'])+r'\end{pmatrix}'
    safe_tex=(r'\begin{equation}'+'\n'+r'\Delta\vect V='+vec+
      r'\ \text{м/с},\qquad \norm{\Delta\vect V}=\ExampleSafeDV\ \text{м/с}.'+'\n'+
      r'\label{eq:example-safe-impulse}\end{equation}'+'\n'+
      r'\begin{equation}\begin{aligned}'+'\n'+
      r'\Delta t_{\mathrm{TCA}}&=\ExampleSafeTCA\ \text{с},\\'+'\n'+
      r'd_{\min}&=\ExampleSafeNonlinearMiss\ \text{м},\\'+'\n'+
      r'P_c\ \text{с учётом ошибок исполнения}&=\ExampleSafeNonlinearPc<10^{-5}.'+'\n'+
      r'\end{aligned}\label{eq:example-safe-verification}\end{equation}'+'\n')
    (BASE/'example-safe-result.tex').write_text(safe_tex)
    with (BASE/'example-lead-table.csv').open('w') as f:
        f.write('lead_periods,lead_s,minimum_found_dv_m_s,radial_signed_m_s,tangential_signed_m_s,normal_signed_m_s,nonlinear_pc\n')
        for z in r['lead_time_comparison']:
            raw=[z['lead_periods'],z['lead_s'],z['best']['dv_norm_m_s'],*z['axis_signed_m_s'],z['best']['nonlinear']['pc']]
            f.write(','.join('' if v is None or not np.isfinite(v) else f'{v:.12g}' for v in raw)+'\n')


def savefig(fig,name):
    fig.savefig(BASE/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(BASE/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)


def cov_ellipse(ax,cov,level=1.,**kwargs):
    val,vec=np.linalg.eigh(cov)
    angle=np.rad2deg(np.arctan2(vec[1,1],vec[0,1]))
    ax.add_patch(Ellipse((0,0),2*level*np.sqrt(val[1]),2*level*np.sqrt(val[0]),
                  angle=angle,fill=False,**kwargs))


def orbit_figure():
    """Nominal orbits with explicit Earth occlusion in an orthographic view.

    mplot3d sorts entire artists, so a single orbit line may incorrectly lie
    behind the whole Earth surface. Split each curve into short segments:
    only points behind the Earth and within its projected disk are occluded.
    All three data axes have the same scale and are centred on the Earth.
    """
    tt=np.linspace(0,PERIOD,1800)
    oa=np.array([cart(kepler(Q0,t))[0] for t in tt])
    period_b=2*np.pi*np.sqrt(QB0[0]**3/MU)
    ob=np.array([cart(kepler(QB0,t))[0] for t in np.linspace(0,period_b,1800)])
    elevation,azimuth=27.,-65.
    el,az=np.deg2rad([elevation,azimuth])
    toward_camera=np.array([np.cos(el)*np.cos(az),
                            np.cos(el)*np.sin(az),np.sin(el)])
    fig=plt.figure(figsize=(9.1,7.1))
    ax=fig.add_subplot(111,projection='3d',proj_type='ortho',
                       computed_zorder=False)
    u=np.linspace(0,2*np.pi,70);v=np.linspace(0,np.pi,35)
    ax.plot_surface(RE*np.outer(np.cos(u),np.sin(v)),
       RE*np.outer(np.sin(u),np.sin(v)),
       RE*np.outer(np.ones_like(u),np.cos(v)),
       color='#DBE8EE',alpha=.55,linewidth=0,shade=False,zorder=2)
    handles=[]
    for orbit,color,label in [(oa,PALETTE[0],'Управляемый КА'),
                               (ob,PALETTE[4],'Опасный объект')]:
        segments=np.stack([orbit[:-1],orbit[1:]],axis=1)
        middle=segments.mean(axis=1)
        depth=middle@toward_camera
        projected_radius_squared=np.einsum('ij,ij->i',middle,middle)-depth**2
        hidden=(depth<0)&(projected_radius_squared<RE**2)
        for mask,zorder in [(hidden,1),(~hidden,3)]:
            selected=np.flatnonzero(mask)
            runs=np.split(selected,np.flatnonzero(np.diff(selected)>1)+1)
            curves=[orbit[run[0]:run[-1]+2] for run in runs if len(run)]
            ax.add_collection3d(Line3DCollection(curves,
                colors=color,linewidths=2.,zorder=zorder))
        handles.append(Line2D([0],[0],color=color,lw=2.,label=label))
    marker=ax.scatter(*R0,color='black',s=45,label='Область сближения',
                      depthshade=False,zorder=4)
    lim=7600
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim),
           xlabel='$X$, км',ylabel='$Y$, км',zlabel='$Z$, км')
    ax.set_box_aspect((1,1,1));ax.view_init(elevation,azimuth)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax.legend(handles=[*handles,marker],loc='upper left',fontsize=11)
    fig.tight_layout();savefig(fig,'example-orbits')


def figures(r):
    data=np.load(BASE/'example-data.npz')
    solutions=r['solutions']
    # Encounter-plane probability contours and equally costly choices.
    xx=np.linspace(-650,650,351); yy=np.linspace(-650,650,351)
    X,Y=np.meshgrid(xx,yy)
    B=np.column_stack([X.ravel(),Y.ravel()])
    pp=PC(B).reshape(X.shape)
    fig,ax=plt.subplots(figsize=(8.4,7.2))
    levels=[1e-7,1e-6,1e-5,1e-4,1e-3]
    cs=ax.contour(X,Y,pp,levels=levels,colors=['#D2DCE4','#B1C6D4','#79539C','#87B0BF','#7F969E'],linewidths=[1,1,1.7,1,1])
    ax.clabel(cs,fmt=lambda p:f'$10^{{{int(np.log10(p))}}}$',fontsize=10)
    cov_ellipse(ax,CB,1.,color='#64727C',linestyle='--',linewidth=1.4)
    ax.add_patch(Circle((0,0),HBR,color=PALETTE[4],alpha=.65))
    ax.scatter(*B0,color='black',s=40,label='Без манёвра',zorder=5)
    for name,key,color,marker in [('Максимальный промах','maximum_miss_fixed_cost',PALETTE[0],'o'),
        ('Минимальная вероятность','minimum_pc_fixed_cost',PALETTE[1],'s'),
        ('Минимальный импульс','minimum_pc_impulse',PALETTE[3],'^')]:
        z=solutions[key];b=np.array(z['b_linear_m'])
        ax.annotate('',b,B0,arrowprops={'arrowstyle':'->','color':color,'lw':1.7})
        ax.scatter(*b,color=color,s=65,marker=marker,label=name,zorder=6)
    ax.set(xlabel='$b_1$, м',ylabel='$b_2$, м',xlim=(-650,650),ylim=(-650,650),aspect='equal')
    ax.legend(loc='upper left',framealpha=.94)
    fig.tight_layout();savefig(fig,'example-bplane')

    # Matrix column norms compare effects per 1 m/s without asserting risk.
    fig,axs=plt.subplots(2,1,figsize=(9.6,7.1),sharex=True)
    ff=data['lead_s']/PERIOD
    for j,(name,color) in enumerate(zip(['Радиальный','Трансверсальный','Нормальный'],PALETTE)):
        axs[0].plot(ff,np.linalg.norm(data['G'][:,:,j],axis=1)/1000.,color=color,label=name)
        vals=np.abs(data['axis_signed_dv_m_s'][:,j])
        vals=np.where(vals<10.,vals,np.nan)
        axs[1].plot(ff,vals,color=color)
    axs[0].set_ylabel('Смещение от 1 м/с, км')
    axs[1].set(ylabel='Достаточный $|\\Delta V|$, м/с',xlabel='Время упреждения, доли периода',ylim=(.005,10.))
    axs[1].set_yscale('log')
    selected=r['lead_time_comparison']
    axs[1].scatter([z['lead_periods'] for z in selected],
      [z['best']['dv_norm_m_s'] for z in selected],
      color=PALETTE[3],marker='D',s=45,zorder=5,
      label='Свободное направление')
    axs[1].legend(loc='upper right',fontsize=10.5)
    axs[0].legend(ncol=3,loc='upper left',fontsize=10.5)
    axs[0].axvline(.4,color='#999999',ls=':',lw=1)
    axs[1].axvline(.4,color='#999999',ls=':',lw=1)
    fig.tight_layout();savefig(fig,'example-lead-time')

    fig,ax=plt.subplots(figsize=(9.6,5.4))
    ax.grid(False)
    mesh=ax.pcolormesh(data['map_amplitudes_m_s'],ff,np.maximum(data['map_pc'],1e-7),
          norm=LogNorm(1e-7,1e-2),cmap='viridis',shading='auto',rasterized=True)
    ax.contour(data['map_amplitudes_m_s'],ff,data['map_pc'],levels=[PC_LIMIT],colors='white',linewidths=1.4)
    cb=fig.colorbar(mesh,ax=ax,pad=.025);cb.set_label('$P_c$')
    ax.set(xlabel='Трансверсальный импульс $\\Delta V_t$, м/с',
           ylabel='Время упреждения, доли периода')
    fig.tight_layout();savefig(fig,'example-risk-map')

    orbit_figure()

    # Local passage before and after the minimum-impulse manoeuvre.
    opt=np.array(solutions['minimum_pc_impulse']['dv_rtn_m_s'])
    tj=np.linspace(-.12,.12,241)
    nominal=[];managed=[]
    for t in tj:
        ra,va=cart(kepler(Q0,t));rb,vb=cart(kepler(QB0,t))
        rc,vc=after_jump(LEAD_MAIN,opt,t)
        nominal.append((rb-ra)*1000.);managed.append((rb-rc)*1000.)
    nominal=np.array(nominal);managed=np.array(managed)
    # Plot distance, the two transverse coordinates show the true local miss.
    fig,axs=plt.subplots(1,2,figsize=(10.4,4.7))
    axs[0].plot(tj,np.linalg.norm(nominal,axis=1),color='#687582',label='Без манёвра')
    axs[0].plot(tj,np.linalg.norm(managed,axis=1),color=PALETTE[1],label='После манёвра')
    axs[0].set(xlabel='Время от исходного TCA, с',ylabel='Расстояние, м')
    axs[0].legend(fontsize=10)
    i0=np.argmin(np.linalg.norm(nominal,axis=1));ic=np.argmin(np.linalg.norm(managed,axis=1))
    xn=nominal@EPLANE.T;xc=managed@EPLANE.T
    axs[1].scatter(*B0,s=50,color='#687582',label='Без манёвра')
    axs[1].scatter(*np.array(solutions['minimum_pc_impulse']['b_linear_m']),s=65,color=PALETTE[1],label='После манёвра')
    cov_ellipse(axs[1],CB,1.,color='#687582',ls='--',lw=1.3)
    axs[1].add_patch(Circle((0,0),HBR,color=PALETTE[4],alpha=.65))
    axs[1].set(xlabel='$b_1$, м',ylabel='$b_2$, м',aspect='equal',xlim=(-220,260),ylim=(-260,400))
    fig.tight_layout();savefig(fig,'example-local-encounter')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args()
    r=json.loads((BASE/'example-results.json').read_text()) if args.figures_only else compute()
    figures(r)


if __name__=='__main__':main()
