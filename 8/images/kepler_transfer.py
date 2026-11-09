#!/usr/bin/env python3
"""Seven finite-thrust manoeuvres: propagation solely in classical elements.

Run: python3 8/images/kepler_transfer.py
Redraw saved results: python3 8/images/kepler_transfer.py --figures-only
State order: (a [km], e, i, Omega, omega, nu [rad], m [kg]).
All propagation, including passive arcs, uses Gauss' planetary equations.
All stored states and plotted orbital quantities are classical elements.
"""
from pathlib import Path
import argparse
import json
import os
BASE=Path(__file__).resolve().parent
BUILD=BASE.parent/'build'
BUILD.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(BUILD/'mplconfig'))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MU,RE,J2=398600.4418,6378.1363,1.08262668e-3
G0,ISP,FORCE,M0,MDRY=9.80665,300.,450.,2000.,1075.
VE=G0*ISP/1000.                 # Effective exhaust speed, km/s.
MDOT=FORCE/(G0*ISP)             # Positive propellant flow, kg/s.
AI,EI=18500.,23/37
AF,EF=42164.,.05
II,IF=np.deg2rad([7.,3.])
OMI,OMF=np.deg2rad([30.,30.])
WI,WF=0.,0.
Z0=np.array([AI,EI,II,OMI,WI,0.,M0])
TARGET=np.array([AF,EF,IF,OMF,WF])
RP0=AI*(1-EI)
RAF=AF*(1+EF)
RPF=AF*(1-EF)
ATR=(RP0+RAF)/2
ETR=(RAF-RP0)/(RAF+RP0)
TF=10*86400.
EPSILON=1e-8
WEIGHTS=[np.ones(3)/3,np.array([.5848,.2928,.1027,.0197])]
CONSTRAINTS={'min_duration_s':60.,'max_duration_s':3600.,
             'minimum_pause_s':12*3600.,'dry_mass_kg':MDRY,
             'last_end_before_s':TF,'residual_norm_tolerance':EPSILON}
PALETTE=['#187AA5','#087F78','#B57714','#79539C','#C44847']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':14.5,
 'axes.labelsize':14.,'axes.titlesize':15.5,'legend.fontsize':12.5,
 'xtick.labelsize':12.5,'ytick.labelsize':12.5,
 'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,
 'pdf.fonttype':42,'ps.fonttype':42})


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


def residual(z):
    """Exactly five classical target elements; all angles in radians."""
    return np.array([(z[0]-AF)/AI,z[1]-EF,z[2]-IF,
                     wrap(z[3]-OMF),wrap(z[4]-WF)])


def summary(z):
    return {'a_km':float(z[0]),'e':float(z[1]),'i_deg':float(np.rad2deg(z[2])),
      'Omega_deg':float(np.rad2deg(z[3])%360),'omega_deg':float(np.rad2deg(wrap(z[4]))),
      'nu_deg':float(np.rad2deg(z[5])%360),'nu_unwrapped_rad':float(z[5]),
      'mass_kg':float(z[6]),'fuel_kg':float(M0-z[6]),
      'classical_state_km_rad_kg':z.tolist(),'residual':residual(z).tolist(),
      'residual_norm':float(np.linalg.norm(residual(z),ord=2))}


def rtn_basis(z):
    """Trigonometric orthonormal S,T,W basis; no Cartesian state or drift."""
    inc,node,arg,nu=z[2:6]
    u=arg+nu
    ci,si,co,so,cu,su=np.cos(inc),np.sin(inc),np.cos(node),np.sin(node),np.cos(u),np.sin(u)
    er=np.array([co*cu-so*su*ci,so*cu+co*su*ci,su*si])
    et=np.array([-co*su-so*cu*ci,-so*su+co*cu*ci,cu*si])
    ew=np.array([so*si,-co*si,ci])
    return np.array([er,et,ew])


def gauss_rhs(use_j2,direction=None):
    active=direction is not None
    def rhs(t,z):
        a,e,inc,node,arg,nu,m=z
        if not (a>0 and 1e-7<e<.98 and 1e-6<inc<np.pi-1e-6 and m>0):
            raise ValueError('Classical state left the selected elliptic branch')
        p=a*(1-e*e); r=p/(1+e*np.cos(nu)); h=np.sqrt(MU*p)
        sn,cn=np.sin(nu),np.cos(nu); u=arg+nu; su,cu=np.sin(u),np.cos(u)
        si,ci=np.sin(inc),np.cos(inc)
        S=T=W=0.
        if use_j2:
            k=1.5*J2*MU*RE**2/r**4
            S=k*(3*si*si*su*su-1)
            T=-k*si*si*np.sin(2*u)
            W=-k*np.sin(2*inc)*su
        if active:
            stw=(FORCE/1000./m)*(rtn_basis(z)@direction)
            S+=stw[0];T+=stw[1];W+=stw[2]
        adot=2*a*a/h*(e*sn*S+(p/r)*T)
        edot=(p*sn*S+((p+r)*cn+r*e)*T)/h
        idot=r*cu*W/h
        ndot=r*su*W/(h*si)
        wdot=(-p*cn*S+(p+r)*sn*T)/(h*e)-r*su*ci*W/(h*si)
        nudot=h/r**2+(p*cn*S-(p+r)*sn*T)/(h*e)
        return np.array([adot,edot,idot,ndot,wdot,nudot,-MDOT if active else 0.])
    return rhs


def integrator_options(tight):
    return {'method':'DOP853','rtol':2e-13 if tight else 2e-11,
      'atol':np.array([1e-9]+[1e-14]*5+[1e-10]) if tight else np.array([1e-7]+[1e-12]*5+[1e-8])}


def propagate(z,t0,t1,use_j2=False,direction=None,tight=False,record=False):
    if t1<t0-1e-8:raise ValueError('Propagation end precedes its start')
    if t1<=t0+1e-8:
        return z.copy(),None
    sol=solve_ivp(gauss_rhs(use_j2,direction),(t0,t1),z,
        **integrator_options(tight),max_step=60. if direction is not None else 1800.,dense_output=record)
    if not sol.success:raise RuntimeError(sol.message)
    if record:
        spacing=20. if direction is not None else 600.
        tt=np.linspace(t0,t1,max(2,int(np.ceil((t1-t0)/spacing))+1))
        rec=(tt,sol.sol(tt),np.full(len(tt),direction is not None))
    else:rec=None
    return sol.y[:,-1],rec


def impulsive_seed():
    vp0=np.sqrt(MU*(2/RP0-1/AI));vpt=np.sqrt(MU*(2/RP0-1/ATR))
    vat=np.sqrt(MU*(2/RAF-1/ATR));vaf=np.sqrt(MU*(2/RAF-1/AF))
    def vectors(di):
        it=II-di
        q0=np.array([-np.sin(OMI)*np.cos(II),np.cos(OMI)*np.cos(II),np.sin(II)])
        qt=np.array([-np.sin(OMI)*np.cos(it),np.cos(OMI)*np.cos(it),np.sin(it)])
        qf=np.array([-np.sin(OMF)*np.cos(IF),np.cos(OMF)*np.cos(IF),np.sin(IF)])
        return np.array([vpt*qt-vp0*q0,vat*qt-vaf*qf])
    def derivative(di):
        d1=np.hypot(vpt-vp0,2*np.sqrt(vp0*vpt)*np.sin(di/2))
        d2=np.hypot(vaf-vat,2*np.sqrt(vat*vaf)*np.sin((II-IF-di)/2))
        return vp0*vpt*np.sin(di)/d1-vat*vaf*np.sin(II-IF-di)/d2
    optimum=brentq(derivative,0.,II-IF,xtol=1e-15)
    return vectors(optimum),float(optimum)
DVS,DI1=impulsive_seed()


def two_impulses(record=False):
    """Exact scalar apsidal jumps, passive propagation by Gauss' equations."""
    t1=2*2*np.pi*np.sqrt(AI**3/MU)
    t2=t1+np.pi*np.sqrt(ATR**3/MU)
    z=Z0.copy();recs=[]
    z,rec=propagate(z,0,t1,tight=True,record=record)
    if rec is not None:recs.append(rec)
    z[:6]=[ATR,ETR,II-DI1,OMI,0.,4*np.pi]
    z[6]*=np.exp(-np.linalg.norm(DVS[0])/VE)
    z,rec=propagate(z,t1,t2,tight=True,record=record)
    if rec is not None:recs.append(rec)
    z[:6]=[AF,EF,IF,OMF,WF,5*np.pi]
    z[6]*=np.exp(-np.linalg.norm(DVS[1])/VE)
    z,rec=propagate(z,t2,TF,tight=True,record=record)
    if rec is not None:recs.append(rec)
    return z,join_records(recs)


def burn_parameters(dvs):
    mass=M0;plan=[]
    for group,weights in enumerate(WEIGHTS):
        mag=np.linalg.norm(dvs[group])
        if not np.isfinite(mag) or mag<=0:raise ValueError('Invalid group magnitude')
        for index,w in enumerate(weights):
            dv=w*mag;after=mass*np.exp(-dv/VE);tau=(mass-after)/MDOT
            if not CONSTRAINTS['min_duration_s']<=tau<=CONSTRAINTS['max_duration_s']:
                raise ValueError('Duration outside [60,3600] seconds')
            plan.append({'group':group,'index':index,'weight':float(w),
                'delta_v_mps':float(dv*1000),'duration_s':float(tau),
                'direction':(dvs[group]/mag).tolist(),'mass_before_kg':float(mass),
                'mass_after_kg':float(after)})
            mass=after
    if mass<MDRY:raise ValueError('Planned mass below dry mass')
    return plan


def coast_to_center(z,t,group,previous_group,tau,use_j2,offset,tight,record):
    period=2*np.pi*np.sqrt(z[0]**3/MU)
    tmin=t+(2. if previous_group is not None and group!=previous_group else 1.5)*period
    if tmin>=TF:raise ValueError('Waiting interval extends beyond final epoch')
    zm,rec1=propagate(z,t,tmin,use_j2,tight=tight,record=record)
    base=0. if group==0 else np.pi+offset
    target=base+2*np.pi*np.ceil((zm[5]-base)/(2*np.pi)+1e-12)
    def event(te,ze):return ze[5]-target
    event.direction=1;event.terminal=True
    sol=solve_ivp(gauss_rhs(use_j2),(tmin,min(tmin+1.4*period,TF)),zm,
        **integrator_options(tight),max_step=1800.,events=event,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    if len(sol.t_events[0])!=1:raise RuntimeError('True-anomaly crossing not found')
    tc=float(sol.t_events[0][0]);start=tc-tau/2
    if start<tmin:raise ValueError('Predicted centre has insufficient preceding coast')
    zs=sol.sol(start)
    if record:
        tt=np.linspace(tmin,start,max(2,int(np.ceil((start-tmin)/600.))+1))
        rec2=(tt,sol.sol(tt),np.zeros(len(tt),dtype=bool))
        rec=(np.r_[rec1[0],rec2[0]],np.c_[rec1[1],rec2[1]],np.r_[rec1[2],rec2[2]])
    else:rec=None
    return zs,tc,rec


def join_records(recs):
    if not recs:return None
    return {'t':np.concatenate([r[0] for r in recs]),
            'z':np.concatenate([r[1] for r in recs],axis=1),
            'active':np.concatenate([r[2] for r in recs])}


def finite(dvs,use_j2=False,offset=0.,tight=False,record=False):
    plan=burn_parameters(dvs);z=Z0.copy();t=0.;previous=None;recs=[]
    for b in plan:
        zs,tc,rec=coast_to_center(z,t,b['group'],previous,b['duration_s'],use_j2,offset,tight,record)
        b.update({'center_s':tc,'start_s':tc-b['duration_s']/2,'end_s':tc+b['duration_s']/2,
                  'true_anomaly_offset_rad':float(offset if b['group']==1 else 0.)})
        if b['start_s']<0 or b['end_s']>TF:raise ValueError('Manoeuvre outside time interval')
        if previous is not None and b['start_s']-t<CONSTRAINTS['minimum_pause_s']:
            raise ValueError('Pause shorter than 12 hours')
        if rec is not None:recs.append(rec)
        z,rec=propagate(zs,b['start_s'],b['end_s'],use_j2,np.array(b['direction']),tight,record)
        if rec is not None:recs.append(rec)
        t=b['end_s'];previous=b['group']
    if z[6]<MDRY:raise ValueError('Integrated mass below dry mass')
    z,rec=propagate(z,t,TF,use_j2,tight=tight,record=record)
    if rec is not None:recs.append(rec)
    return z,plan,join_records(recs)


def save_json(result):
    temporary=BASE/'kepler-results-writing.json'
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(BASE/'kepler-results.json')


def target(result):
    # Internal units only: 10m/s per group-vector component, 0.05rad site offset.
    q=np.zeros(7);history=[];step=1e-3
    def evaluate(qp):return finite(DVS+.010*qp[:6].reshape(2,3),True,.05*qp[6])
    for k in range(12):
        z,plan,_=evaluate(q);f=residual(z);norm=float(np.linalg.norm(f,ord=2))
        row={'iteration':k,'q':q.tolist(),'norm':norm,'residual':f.tolist(),'final':summary(z)}
        history.append(row);result['targeting_history']=history;save_json(result)
        print(f'Iteration {k}: norm={norm:.10g}, a={z[0]:.8f}, e={z[1]:.8f}, mass={z[6]:.6f}',flush=True)
        if norm<=EPSILON:return DVS+.010*q[:6].reshape(2,3),.05*q[6]
        derivatives=np.empty((5,7))
        for j in range(7):
            plus=q.copy();minus=q.copy();plus[j]+=step;minus[j]-=step
            zp,_,_=evaluate(plus);zm,_,_=evaluate(minus)
            derivatives[:,j]=(residual(zp)-residual(zm))/(2*step)
        dq,_,rank,singular=np.linalg.lstsq(derivatives,-f,rcond=1e-10)
        row['derivative_rank']=int(rank);row['singular_values']=singular.tolist()
        if rank<5:raise RuntimeError('Derivative matrix has rank below five')
        accepted=False
        for factor in [1.,.5,.25,.125,.0625,.03125,.015625]:
            candidate=q+factor*dq
            try:zt,pt,_=evaluate(candidate)
            except (ValueError,RuntimeError) as exc:
                row.setdefault('rejected_trials',[]).append({'factor':factor,'reason':str(exc)});continue
            if np.linalg.norm(residual(zt),ord=2)<norm:
                q=candidate;row['step_factor']=factor;accepted=True;break
        if not accepted:save_json(result);raise RuntimeError('No feasible decreasing Newton step')
    raise RuntimeError('Targeting iteration limit reached')


def compute():
    result={'description':'Spatial elliptic-orbit transfer; all state propagation by Gauss equations in classical Keplerian elements.',
      'state_order':['a_km','e','i_rad','Omega_rad','omega_rad','nu_unwrapped_rad','m_kg'],
      'constants':{'mu_km3_s2':MU,'earth_radius_km':RE,'J2':J2,'g0_m_s2':G0,'Isp_s':ISP,'force_N':FORCE,'mass0_kg':M0,'dry_mass_kg':MDRY},
      'initial_classical_state':Z0.tolist(),'target_classical_elements':TARGET.tolist(),
      'target_epoch_s':TF,'target_apogee_radius_km':RAF,'target_perigee_radius_km':RPF,
      'transfer_semimajor_axis_km':ATR,'transfer_eccentricity':ETR,
      'plane_change_first_deg':float(np.rad2deg(DI1)),
      'nominal_group_dv_kms':DVS.tolist(),'nominal_total_dv_mps':float(1000*np.linalg.norm(DVS,axis=1).sum()),
      'weights':[w.tolist() for w in WEIGHTS],'constraints':CONSTRAINTS,
      'residual_definition':['(a-af)/ai','e-ef','i-if','wrap(Omega-Omegaf)','wrap(omega-omegaf)'],
      'residual_angle_unit':'rad','epsilon':EPSILON,
      'derivatives':{'scheme':'componentwise central differences','internal_step':1e-3},
      'schedule':'After preceding manoeuvre coast >=1.5 osculating periods (2 for group transition), find next unwrapped true-anomaly threshold 0 mod2pi for group1, pi+offset mod2pi for group2. Burn starts tau/2 before the passively predicted event. Events are recomputed in every forecast.',
      'direction_law':'One fixed inertial unit vector per group, projected analytically onto S,T,W by the trigonometric orbital basis.',
      'integrator':{'method':'DOP853','primary':{'rtol':2e-11,'atol':[1e-7]+[1e-12]*5+[1e-8]},
                    'tight':{'rtol':2e-13,'atol':[1e-9]+[1e-14]*5+[1e-10]},
                    'max_step_passive_s':1800.,'max_step_thrust_s':60.},
      'targeting':'7 parameters, 5 classical-element residuals; minimum-norm Newton correction with backtracking, not fuel optimization.',
      'stages':{}}
    traces={}
    z,tr=two_impulses(True);result['stages']['two_impulses']=summary(z);traces['two_impulses']=tr
    for name,j2 in [('finite_central',False),('finite_j2',True)]:
        z,plan,tr=finite(DVS,j2,record=True)
        result['stages'][name]=summary(z);traces[name]=tr
        result['nominal_burns' if not j2 else 'initial_j2_burns']=plan
        save_json(result);print(name,summary(z),flush=True)
    dvs,offset=target(result)
    z,plan,tr=finite(dvs,True,offset,record=True)
    result['stages']['targeted_j2']=summary(z);result['targeted_burns']=plan;traces['targeted_j2']=tr
    result['targeted_group_dv_kms']=dvs.tolist()
    inertial_unit=dvs/np.linalg.norm(dvs,axis=1)[:,None]
    result['targeted_group_unit_directions']=inertial_unit.tolist()
    reference_states=[Z0.copy(),np.array([ATR,ETR,II-DI1,OMI,0.,np.pi,M0])]
    reference_bases=np.array([rtn_basis(ref) for ref in reference_states])
    result['reference_orbital_elements']=[ref[:6].tolist() for ref in reference_states]
    result['reference_basis_rows_STW_inertial']=reference_bases.tolist()
    result['targeted_group_unit_directions_reference_STW']=[(basis@unit).tolist() for basis,unit in zip(reference_bases,inertial_unit)]
    result['targeted_group_dv_reference_STW_kms']=[(basis@dv).tolist() for basis,dv in zip(reference_bases,dvs)]
    result['direction_macro_semantics']='KeplerDirectionOne/Two are constant components of the final inertial unit thrust vectors in two fixed reference orbital bases: initial perigee before impulse1; transfer apogee before impulse2. They are not a law of continuously fixed RTN direction.'
    result['targeted_total_dv_mps']=float(1000*np.linalg.norm(dvs,axis=1).sum())
    result['targeted_second_group_true_anomaly_offset_deg']=float(np.rad2deg(offset))
    save_json(result)
    zt,pt,_=finite(dvs,True,offset,tight=True)
    result['verification']={'tight_final':summary(zt),'primary_criterion_satisfied':bool(np.linalg.norm(residual(z))<=EPSILON),
      'tight_criterion_satisfied':bool(np.linalg.norm(residual(zt))<=EPSILON),
      'minimum_pause_s':min(b['start_s']-a['end_s'] for a,b in zip(plan[:-1],plan[1:])),
      'min_duration_s':min(b['duration_s'] for b in plan),'max_duration_s':max(b['duration_s'] for b in plan),
      'last_end_s':plan[-1]['end_s'],'mass_reserve_kg':float(z[6]-MDRY)}
    assert np.linalg.norm(residual(z),ord=2)<=EPSILON
    assert np.linalg.norm(residual(zt),ord=2)<=EPSILON
    assert z[6]>=MDRY and zt[6]>=MDRY
    assert result['verification']['minimum_pause_s']>=CONSTRAINTS['minimum_pause_s']
    assert all(60<=b['duration_s']<=3600 and b['end_s']<=TF for b in plan)
    save_json(result)
    np.savez_compressed(BASE/'kepler-results.npz',**{name+'_'+key:value for name,tr in traces.items() for key,value in tr.items()})
    return result,traces



def fmt(value,digits=3):
    if abs(value)<.5*10**(-digits):value=0.
    return f'{value:.{digits}f}'.replace('.',r'{,}')


def sci(value,digits=3):
    if value==0:return '0'
    mantissa,exponent=f'{value:.{digits}e}'.split('e')
    return mantissa.replace('.',r'{,}')+r'\cdot10^{'+str(int(exponent))+'}'


def tex_outputs(result):
    nominal=np.linalg.norm(np.array(result['nominal_group_dv_kms']),axis=1)*1000
    final=np.linalg.norm(np.array(result['targeted_group_dv_kms']),axis=1)*1000
    primary=result['stages']['targeted_j2'];strict=result['verification']['tight_final']
    fields={
       'KeplerDVOne':(nominal[0],3),'KeplerDVTwo':(nominal[1],3),
       'KeplerDVTotal':(result['nominal_total_dv_mps'],3),
       'KeplerPlaneOne':(result['plane_change_first_deg'],7),
       'KeplerTransferA':(result['transfer_semimajor_axis_km'],3),
       'KeplerTransferE':(result['transfer_eccentricity'],9),
       'KeplerTargetRa':(result['target_apogee_radius_km'],3),
       'KeplerTargetRp':(result['target_perigee_radius_km'],3),
       'KeplerTargetA':(AF,0),'KeplerTargetE':(EF,2),
       'KeplerTargetI':(np.rad2deg(IF),0),'KeplerTargetOmega':(np.rad2deg(OMF),0),
       'KeplerTargetomega':(np.rad2deg(WF),0),
       'KeplerDurationDays':(TF/86400,0),'KeplerFinalEpochHours':(TF/3600,0),
       'KeplerTargetDVTotal':(result['targeted_total_dv_mps'],3),
       'KeplerDVIncrease':(result['targeted_total_dv_mps']-result['nominal_total_dv_mps'],3),
       'KeplerFinalMass':(primary['mass_kg'],3),'KeplerFuel':(primary['fuel_kg'],3),
       'KeplerDryReserve':(primary['mass_kg']-MDRY,3),
       'KeplerNominalMass':(result['stages']['finite_central']['mass_kg'],3),
       'KeplerIterations':(len(result['targeting_history'])-1,0),
       'KeplerNominalGroupOne':(nominal[0],3),'KeplerNominalGroupTwo':(nominal[1],3),
       'KeplerTargetGroupOne':(final[0],3),'KeplerTargetGroupTwo':(final[1],3),
       'KeplerSiteShift':(result['targeted_second_group_true_anomaly_offset_deg'],7),
       'KeplerNuShift':(result['targeted_second_group_true_anomaly_offset_deg'],7)}
    for stage,prefix in [('finite_central','KeplerCentral'),('finite_j2','KeplerJTwo')]:
        for suffix,key,precision in [('A','a_km',3),('E','e',9),('I','i_deg',7),
                                      ('Omega','Omega_deg',7),('omega','omega_deg',7)]:
            fields[prefix+suffix]=(result['stages'][stage][key],precision)
    for suffix,key,precision in [('A','a_km',6),('E','e',11),('I','i_deg',9),
                                 ('Omega','Omega_deg',9),('omega','omega_deg',9),('Nu','nu_deg',9)]:
        fields['KeplerStrict'+suffix]=(strict[key],precision)
    lines=['% Generated by kepler_transfer.py; all propagation in classical elements.']
    for name,(value,precision) in fields.items():
        lines.append('\\newcommand{\\'+name+'}{'+fmt(value,precision)+'}')
    for name,value in {'KeplerEpsilon':EPSILON,
         'KeplerNormFiniteCentral':result['stages']['finite_central']['residual_norm'],
         'KeplerNormFiniteJTwo':result['stages']['finite_j2']['residual_norm'],
         'KeplerNormFinal':primary['residual_norm'],
         'KeplerNormTight':strict['residual_norm']}.items():
        lines.append('\\newcommand{\\'+name+'}{'+sci(value)+'}')
    for k,name in enumerate(['KeplerDirectionOne','KeplerDirectionTwo']):
        vector=result['targeted_group_unit_directions_reference_STW'][k]
        matrix=r'\begin{pmatrix}'+r'\\'.join(fmt(v,8) for v in vector)+r'\end{pmatrix}'
        lines.append('\\newcommand{\\'+name+'}{'+matrix+'}')
    (BASE/'kepler-values.tex').write_text('\n'.join(lines)+'\n')
    # Rows are the five actual classical target elements plus free true anomaly.
    lines=[r'% Target and attained classical elements; tighter Gauss forecast.',
       r'\begin{tabular}{@{}lrr@{}}',r'\toprule',
       r'Элемент & Цель & Повторный прогноз \\',r'\midrule']
    rows=[('$a$, км',AF,strict['a_km'],6),('$e$',EF,strict['e'],11),
          ('$i$, град',np.rad2deg(IF),strict['i_deg'],9),
          (r'$\Omega$, град',np.rad2deg(OMF),strict['Omega_deg'],9),
          (r'$\omega$, град',np.rad2deg(WF),strict['omega_deg'],9)]
    for label,target_value,value,precision in rows:
        attained=('$'+sci(value,4)+'$') if label.startswith(r'$\omega$') and 0<abs(value)<1e-5 else fmt(value,precision)
        lines.append(label+' & '+fmt(target_value,precision)+' & '+attained+r' \\')
    lines.append(r'$\nu$, град & Свободна & '+fmt(strict['nu_deg'],6)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (BASE/'kepler-final-elements-table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'% Model comparison; terminal norm is dimensionless.',
       r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
       r'Модель & $\|\mathbf F\|_2$ & $\Delta V_\Sigma$, м/с & $m_f$, кг \\',r'\midrule']
    labels={'two_impulses':'Два импульса','finite_central':'Протяжённые, центр. поле',
            'finite_j2':'Протяжённые, $J_2$','targeted_j2':'Уточнённые, $J_2$'}
    for key,label in labels.items():
        v=result['stages'][key]
        norm=strict['residual_norm'] if key=='targeted_j2' else v['residual_norm']
        dv=result['targeted_total_dv_mps'] if key=='targeted_j2' else result['nominal_total_dv_mps']
        lines.append(label+' & $'+sci(norm)+'$ & '+fmt(dv,3)+' & '+fmt(v['mass_kg'],3)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (BASE/'kepler-model-table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'% Accepted Newton iterates.',r'\begin{tabular}{@{}rrr@{}}',r'\toprule',
          r'$k$ & $\|\mathbf F\|_2$ & $m_f$, кг \\',r'\midrule']
    for row in result['targeting_history']:
        lines.append(str(row['iteration'])+' & $'+sci(row['norm'],4)+'$ & '+fmt(row['final']['mass_kg'],3)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (BASE/'kepler-history-table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'% Seven final finite-thrust manoeuvres.',r'\begin{tabular}{@{}rrrrr@{}}',r'\toprule',
       r'№ & $t_{b,j}$, сут & $\tau_j$, мин & $\Delta V_j$, м/с & $m_{e,j}$, кг \\',r'\midrule']
    for k,b in enumerate(result['targeted_burns']):
        lines.append(str(k+1)+' & '+fmt(b['start_s']/86400,6)+' & '+fmt(b['duration_s']/60,4)+' & '+fmt(b['delta_v_mps'],3)+' & '+fmt(b['mass_after_kg'],3)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (BASE/'kepler-burn-table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'% Final fixed inertial directions, expressed in two fixed reference S,T,W bases.',
       r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',
       r'Группа & $u_S$ & $u_T$ & $u_W$ \\',r'\midrule']
    for k,label in enumerate(['Перигейная','Апогейная']):
        lines.append(label+' & '+' & '.join(fmt(v,8) for v in result['targeted_group_unit_directions_reference_STW'][k])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (BASE/'kepler-direction-table.tex').write_text('\n'.join(lines)+'\n')


def save_figure(fig,name):
    fig.savefig(BASE/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    fig.savefig(BASE/(name+'.png'),dpi=160,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def figures(result,traces):
    # All plotted ordinates are integrated classical elements, thrust, or mass.
    fig,axes=plt.subplots(5,1,figsize=(10.3,9.6),sharex=True)
    stages=[('finite_central','Протяжённые манёвры, центральное поле',PALETTE[0],'-'),
            ('finite_j2','Исходная последовательность с $J_2$',PALETTE[2],'--'),
            ('targeted_j2','Уточнённая последовательность с $J_2$',PALETTE[1],'-')]
    for name,label,color,style in stages:
        tr=traces[name];ix=np.arange(0,len(tr['t']),max(1,len(tr['t'])//3000))
        data=tr['z'][:,ix].copy()
        data[0]/=1000.;data[2:5]=np.rad2deg(data[2:5])
        for row,ax in enumerate(axes):
            ax.plot(tr['t'][ix]/86400,data[row],style,color=color,lw=1.1,label=label)
    target_values=[AF/1000,EF,np.rad2deg(IF),np.rad2deg(OMF),np.rad2deg(WF)]
    for ax,value,label in zip(axes,target_values,['$a$, тыс. км','$e$','$i$, град',r'$\Omega$, град',r'$\omega$, град']):
        ax.axhline(value,color='#777777',lw=.7,ls=':')
        ax.set_ylabel(label);ax.yaxis.set_major_locator(MaxNLocator(4))
    axes[1].set_ylim(0,.8);axes[-1].set_xlabel('Время, сут');axes[-1].set_xlim(0,TF/86400)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.005),frameon=False,ncol=1,fontsize=12.5)
    fig.tight_layout(h_pad=.2,rect=(0,0,1,.90));save_figure(fig,'kepler-elements')
    # A single norm/tolerance is used for convergence and its validation.
    hist=result['targeting_history'];fig,ax=plt.subplots(figsize=(8.3,4.1))
    ax.semilogy([h['iteration'] for h in hist],[h['norm'] for h in hist],'o-',color=PALETTE[0],ms=6,lw=1.4)
    ax.axhline(EPSILON,color='#777777',ls='--',lw=1.1,label=r'$\varepsilon=10^{-8}$')
    ax.set_ylabel(r'$\|\mathbf{F}\|_2$');ax.set_xlabel('Номер итерации')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(loc='upper right',fontsize=13)
    ax.set_ylim(1e-15,1.);fig.tight_layout();save_figure(fig,'kepler-convergence')
    # The thrust and mass profiles come directly from the Gauss integration.
    tr=traces['targeted_j2'];fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10.1,4.3),sharex=True,gridspec_kw={'height_ratios':[.8,1.4]})
    ax1.plot(tr['t']/86400,FORCE*tr['active'],color=PALETTE[0],lw=1.2)
    ax1.set_ylabel('Тяга, Н');ax1.set_ylim(-25,575);ax1.set_yticks([0,450])
    for k,b in enumerate(result['targeted_burns']):
        ax1.annotate(str(k+1),(b['center_s']/86400,450),xytext=(0,6),textcoords='offset points',ha='center',fontsize=12.5)
    ax2.plot(tr['t']/86400,tr['z'][6],color=PALETTE[1],lw=1.5,label='Масса КА')
    ax2.axhline(MDRY,color=PALETTE[4],ls='--',lw=1.,label=f'Сухая масса {MDRY:.0f} кг')
    ax2.set_ylabel('Масса, кг');ax2.set_ylim(MDRY-40,M0+50);ax2.set_xlabel('Время, сут');ax2.set_xlim(0,TF/86400);ax2.legend(loc='upper right')
    fig.tight_layout(h_pad=.25);save_figure(fig,'kepler-timeline')
    # Nominal group magnitudes, weights, and corresponding duration variation.
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.1,3.3));nom=result['nominal_burns']
    for ax,start,end,color in [(ax1,0,3,PALETTE[0]),(ax2,3,7,PALETTE[3])]:
        subset=nom[start:end];x=np.arange(len(subset));vals=[b['delta_v_mps'] for b in subset]
        ax.bar(x,vals,color=color,width=.56)
        for k,b in enumerate(subset):
            label=f"{b['delta_v_mps']:.1f} м/с\n{b['duration_s']/60:.2f} мин".replace('.',',')
            ax.annotate(label,(k,vals[k]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=12.5)
        ax.set_xticks(x,[str(k) for k in range(start+1,end+1)]);ax.set_xlabel('Номер манёвра')
        ax.set_ylabel(r'$\Delta V_j$, м/с');ax.set_ylim(0,max(vals)*1.42);ax.grid(axis='x',visible=False)
    ax1.set_title('Перигейная группа:\nравные '+r'$\Delta V_j$')
    ax2.set_title('Апогейная группа:\nубывающие '+r'$\Delta V_j$')
    fig.tight_layout(w_pad=2);save_figure(fig,'kepler-groups')


def redraw():
    result=json.loads((BASE/'kepler-results.json').read_text())
    names=['two_impulses','finite_central','finite_j2','targeted_j2']
    with np.load(BASE/'kepler-results.npz') as data:
        traces={name:{key:data[name+'_'+key] for key in ['t','z','active']} for name in names}
    tex_outputs(result);figures(result,traces)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only',action='store_true',help='Regenerate four figures and TeX outputs from saved classical-element data')
    args=parser.parse_args()
    if args.figures_only:redraw()
    else:
        result,traces=compute();tex_outputs(result);figures(result,traces)
        print(json.dumps(result['verification'],ensure_ascii=False,indent=2))
