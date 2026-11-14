#!/usr/bin/env python3
"""Educational two-stage ascent, two upper-stage burns, a 650-km circular target.

Run: python3 10/images/ascent_numerics.py
Redraw: python3 10/images/ascent_numerics.py --figures-only
Dynamics: planar Cartesian state in km, km/s, kg; inertial non-rotating Earth.
Loss integrals are in km/s and include all active and passive phases.
Stage separation drops 16000 kg without a velocity jump.
Useful outputs: JSON parameters/history/checks, three CSV/NPZ trajectories,
vector PDF figures and ascent-values.tex. Logs and raster previews: ../build.
This is a prescribed-family targeting calculation, not an ascent optimization.
"""
from pathlib import Path
import os, json, argparse
BASE=Path(__file__).resolve().parent
BUILD=BASE.parent/'build'; BUILD.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(BUILD/'mplconfig'))
import numpy as np
from scipy.integrate import solve_ivp
MU=398600.4418; RE=6378.137; G0=9.80665
M1P=160000.; M1D=16000.; M2P=45000.; M2D=4000.; PAYLOAD=5000.
F1=3500000.; ISP1=300.; F2=800000.; ISP2=450.
MD1=F1/(G0*ISP1); MD2=F2/(G0*ISP2)
T1=M1P/MD1; T2MAX=M2P/MD2; M0=M1P+M1D+M2P+M2D+PAYLOAD
TARGET_H=650.; RF=RE+TARGET_H; VC=np.sqrt(MU/RF)
TV=10.; TP=25.; PITCH_ANGLE=20.; RAISE_ANGLE=45.
CD=0.3; AREA=10.; RHO0=1.225; H=7.2

def parameters(z):
 r=np.linalg.norm(z[:2]); er=z[:2]/r; et=np.array([-er[1],er[0]])
 vr=z[2:4]@er; vt=z[2:4]@et; v=np.hypot(vr,vt)
 return r,er,et,vr,vt,v,np.arctan2(vr,vt)

def rhs(mode):
 def fun(t,z):
  r,er,et,vr,vt,v,gamma=parameters(z)
  if mode=='vertical': theta=np.pi/2; force=F1; md=MD1
  elif mode=='pitch': theta=np.pi/2-np.deg2rad(PITCH_ANGLE)*(t-TV)/TP;force=F1;md=MD1
  elif mode=='turn': theta=gamma;force=F1;md=MD1
  elif mode=='raise': theta=np.deg2rad(RAISE_ANGLE);force=F2;md=MD2
  elif mode=='circularize': theta=0.;force=F2;md=MD2
  else: theta=gamma;force=0.;md=0.
  u=np.sin(theta)*er+np.cos(theta)*et
  rho=RHO0*np.exp(-max(-1.,r-RE)/H)
  drag=.5*rho*(1000*v)**2*CD*AREA
  vh=z[2:4]/v if v>1e-10 else er
  acc=-MU/r**2*er+force/(1000*z[4])*u-drag/(1000*z[4])*vh
  cosalpha=np.clip(u@vh,-1,1)
  return np.r_[z[2:4],acc,-md,force/(1000*z[4]),MU/r**2*(vr/v if v>1e-10 else 1),drag/(1000*z[4]),force/(1000*z[4])*(1-cosalpha)]
 return fun

def seg(z,t0,t1,mode,tight=False,record=False):
 if t1<=t0:return z,None
 sol=solve_ivp(rhs(mode),(t0,t1),z,method='DOP853',rtol=2e-13 if tight else 2e-11,atol=1e-12 if tight else 1e-10,max_step=.4 if tight else 2.,dense_output=record)
 if not sol.success:raise RuntimeError(sol.message)
 if record:
  tt=np.linspace(t0,t1,max(3,int(np.ceil(t1-t0))+1))
  return sol.y[:,-1],(tt,sol.sol(tt),np.full(len(tt),mode,dtype='U12'))
 return sol.y[:,-1],None

def propagate(q,tight=False,record=False):
 tb1,tc,tb2=q
 z=np.r_[RE,0.,0.,0.,M0,np.zeros(4)]
 recs=[]
 for a,b,m in [(0,TV,'vertical'),(TV,TV+TP,'pitch'),(TV+TP,T1,'turn')]:
  z,rec=seg(z,a,b,m,tight,record)
  if rec:recs.append(rec)
 z[4]-=M1D
 for a,b,m in [(T1,T1+tb1,'raise'),(T1+tb1,T1+tb1+tc,'coast'),(T1+tb1+tc,T1+tb1+tc+tb2,'circularize')]:
  z,rec=seg(z,a,b,m,tight,record)
  if rec:recs.append(rec)
 return z,recs

def resid(q):
 z,_=propagate(q)
 r,er,et,vr,vt,v,gamma=parameters(z)
 return np.array([(r-RF)/RF,vr/VC,(vt-VC)/VC])


PALETTE=['#187AA5','#087F78','#B57714','#79539C','#C44847']
CONSTRAINTS={'max_dynamic_pressure_Pa':50000.,'max_thrust_load_g0':8.,
 'minimum_altitude_m':0.,'minimum_final_mass_kg':M2D+PAYLOAD,
 'maximum_upper_total_burn_s':T2MAX,
 'note':'Illustrative acceptance limits; not optimization constraints.'}

def correct(initial):
 """Damped Newton with a central-difference derivative and one norm tolerance."""
 p=np.array(initial,dtype=float); hist=[]
 steps=np.array([.02,.05,.02]); tolerance=1e-11
 for k in range(12):
  rr=resid(p); nr=float(np.linalg.norm(rr))
  row={'iteration':k,'parameters_s':p.tolist(),'residual_dimensionless':rr.tolist(),'norm':nr}
  hist.append(row)
  if nr<tolerance:break
  derivative=np.column_stack([(resid(p+np.eye(3)[j]*steps[j])-resid(p-np.eye(3)[j]*steps[j]))/(2*steps[j]) for j in range(3)])
  delta=np.linalg.solve(derivative,-rr)
  row['derivative_per_s']=derivative.tolist();row['condition_number']=float(np.linalg.cond(derivative))
  for half in range(15):
   factor=2.**(-half); trial=p+factor*delta
   if (trial>0).all() and trial[0]+trial[2]<T2MAX and np.linalg.norm(resid(trial))<nr:
    p=trial;row['step_fraction']=factor;break
  else:raise RuntimeError('No accepted Newton step')
 else:raise RuntimeError('Newton iteration failed')
 return p,hist,tolerance

def diagnostics(recs,pitch):
 tt=np.concatenate([r[0] for r in recs]); zz=np.hstack([r[1] for r in recs]); modes=np.concatenate([r[2] for r in recs])
 rr=np.linalg.norm(zz[:2],axis=0); er=zz[:2]/rr;et=np.vstack([-er[1],er[0]])
 vr=np.sum(zz[2:4]*er,axis=0);vt=np.sum(zz[2:4]*et,axis=0);vv=np.hypot(vr,vt)
 gamma=np.where(vv>1e-10,np.arctan2(vr,vt),np.pi/2);theta=gamma.copy();force=np.zeros(len(tt))
 for mode in ['vertical','pitch','turn','raise','circularize']:
  idx=modes==mode;force[idx]=F1 if mode in ['vertical','pitch','turn'] else F2
  if mode=='vertical':theta[idx]=np.pi/2
  elif mode=='pitch':theta[idx]=np.pi/2-np.deg2rad(pitch)*(tt[idx]-TV)/TP
  elif mode=='raise':theta[idx]=np.deg2rad(RAISE_ANGLE)
  elif mode=='circularize':theta[idx]=0.
 beta=(theta-gamma+np.pi)%(2*np.pi)-np.pi
 rho=RHO0*np.exp(-np.maximum(-1.,rr-RE)/H)
 q=.5*rho*(1000*vv)**2
 eps=vv**2/2-MU/rr;hh=rr*vt
 aa=-MU/(2*eps);ecc=np.sqrt(np.maximum(0.,1+2*eps*hh**2/MU**2))
 hp=aa*(1-ecc)-RE;ha=aa*(1+ecc)-RE
 return dict(t=tt,z=zz,mode=modes,height=rr-RE,radius=rr,vr=vr,vt=vt,speed=vv,gamma=np.rad2deg(gamma),
  theta=np.rad2deg(theta),beta=np.rad2deg(beta),force=force,dynamic_pressure=q,qalpha=q*np.abs(beta),
  load=force/(zz[4]*G0),energy=eps,moment=hh,perigee=hp,apogee=ha,eccentricity=ecc,
  phi=np.arctan2(zz[1],zz[0]))

def state_summary(z):
 r,er,et,vr,vt,v,gamma=parameters(z)
 eps=v*v/2-MU/r;hh=r*vt;a=-MU/(2*eps)
 eccentricity=np.hypot(r*vt*vt/MU-1.,-r*vr*vt/MU)
 return {'height_km':float(r-RE),'radial_speed_km_s':float(vr),'transverse_speed_km_s':float(vt),
  'speed_km_s':float(v),'gamma_deg':float(np.rad2deg(gamma)),'mass_kg':float(z[4]),
  'upper_remaining_propellant_kg':float(min(M2P,z[4]-M2D-PAYLOAD)),'semimajor_axis_km':float(a),
  'eccentricity':float(eccentricity),'perigee_height_km':float(a*(1-eccentricity)-RE),
  'apogee_height_km':float(a*(1+eccentricity)-RE),
  'characteristic_speed_km_s':float(z[5]),'gravity_loss_km_s':float(z[6]),
  'drag_loss_km_s':float(z[7]),'steering_loss_km_s':float(z[8]),
  'speed_balance_error_m_s':float(1000*(z[5]-sum(z[6:9])-v)),
  'residual_dimensionless':[(r-RF)/RF,vr/VC,(vt-VC)/VC]}

def save_case(name,diag):
 import csv
 headers=['time_s','x_km','y_km','vx_km_s','vy_km_s','mass_kg','characteristic_speed_km_s','gravity_loss_km_s','drag_loss_km_s','steering_loss_km_s',
 'phase','height_km','radial_speed_km_s','transverse_speed_km_s','gamma_deg','thrust_angle_deg','beta_deg','thrust_N','dynamic_pressure_Pa','qalpha_Pa_rad','thrust_load_g0','osculating_perigee_km','osculating_apogee_km']
 with (BASE/f'ascent-{name}.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(headers)
  for i,t in enumerate(diag['t']):
   w.writerow([t,*diag['z'][:,i],diag['mode'][i],diag['height'][i],diag['vr'][i],diag['vt'][i],diag['gamma'][i],diag['theta'][i],diag['beta'][i],diag['force'][i],diag['dynamic_pressure'][i],diag['qalpha'][i],diag['load'][i],diag['perigee'][i],diag['apogee'][i]])
 np.savez_compressed(BASE/f'ascent-{name}.npz',**diag)

def prepare():
 global PITCH_ANGLE
 PITCH_ANGLE=20.
 initial=[150.,300.,80.]
 p,hist,tolerance=correct(initial)
 z,recs=propagate(p,record=True)
 zcheck,recscheck=propagate(p,tight=True,record=True)
 nominal=diagnostics(recscheck,20.)
 cases={'nominal':nominal};results={}
 for name,angle in [('steep',16.),('shallow',24.)]:
  PITCH_ANGLE=angle
  zz,rec=propagate(p,tight=True,record=True)
  d=diagnostics(rec,angle);cases[name]=d;results[name]=state_summary(zz)
 PITCH_ANGLE=20.
 results['nominal']=state_summary(zcheck)
 events=[]
 for tt,zz,mode in recscheck:
  events.append({'phase':str(mode[0]),'time_begin_s':float(tt[0]),'time_end_s':float(tt[-1]),
                 'state_end':state_summary(zz[:,-1])})
 pressure=float(nominal['dynamic_pressure'].max()); load=float(nominal['load'].max())
 valid={'fuel_mass':bool(zcheck[4]>=M2D+PAYLOAD),'upper_burn':bool(p[0]+p[2]<=T2MAX),
 'dynamic_pressure':bool(pressure<=CONSTRAINTS['max_dynamic_pressure_Pa']),
 'thrust_load':bool(load<=CONSTRAINTS['max_thrust_load_g0']),
 'altitude':bool(nominal['height'].min()>=-1e-8)}
 report={'model':'planar Cartesian non-rotating spherical Earth; exponential stationary atmosphere; no lift; two burns of restartable upper-stage engine',
 'constants':{'mu_km3_s2':MU,'earth_radius_km':RE,'g0_m_s2':G0,'target_height_km':TARGET_H,'target_speed_km_s':VC,
  'initial_mass_kg':M0,'payload_kg':PAYLOAD,'stage1_propellant_kg':M1P,'stage1_dry_kg':M1D,'stage2_propellant_kg':M2P,'stage2_dry_kg':M2D,
  'stage1_thrust_N':F1,'stage2_thrust_N':F2,'stage1_isp_s':ISP1,'stage2_isp_s':ISP2,
  'Cd':CD,'area_m2':AREA,'density_sea_level_kg_m3':RHO0,'density_scale_height_km':H,
  'vertical_duration_s':TV,'pitch_duration_s':TP,'pitch_change_deg':20.,'raise_thrust_angle_deg':RAISE_ANGLE},
 'parameter_order':['upper_raise_duration_s','coast_duration_s','upper_circularization_duration_s'],
 'initial_parameters_s':initial,'solved_parameters_s':p.tolist(),'residual_tolerance':tolerance,'history':hist,
 'phases':events,'cases':results,'constraints':CONSTRAINTS,'checks':valid,
 'extrema_note':'Maxima over the recorded grid with spacing no larger than one second, not formal continuous-time guarantees.',
 'max_dynamic_pressure_Pa':pressure,'max_qalpha_Pa_rad':float(nominal['qalpha'].max()),'max_thrust_load_g0':load,
 'verification':{'nominal_final':state_summary(z),'tight_final':state_summary(zcheck),
  'position_difference_m':float(np.linalg.norm(zcheck[:2]-z[:2])*1000),
  'velocity_difference_m_s':float(np.linalg.norm(zcheck[2:4]-z[2:4])*1000),
  'nominal_rtol':2e-11,'tight_rtol':2e-13,'nominal_max_step_s':2.,'tight_max_step_s':.4}}
 (BASE/'ascent-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 for name,d in cases.items():save_case(name,d)
 return report,cases

def figures(report,cases):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 for font in Path('/usr/share/fonts/truetype/paratype').glob('PTF*.ttf'):
  font_manager.fontManager.addfont(str(font))
 from matplotlib.ticker import MaxNLocator
 plt.rcParams.update({'font.family':'PT Serif','mathtext.fontset':'stix','mathtext.rm':'PT Serif','mathtext.it':'PT Serif:italic','mathtext.bf':'PT Serif:bold','font.size':13,'axes.labelsize':13.,'axes.titlesize':14.,'legend.fontsize':11.,
 'xtick.labelsize':11.5,'ytick.labelsize':11.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42})
 d=cases['nominal'];t=d['t']; p=report['solved_parameters_s']; tf=float(t[-1]); edges=[TV,TV+TP,T1,T1+p[0],T1+p[0]+p[1]]
 labels=['Вертикальный','Разворот','Гравитационный','Набор апогея','Пассивный','Довыведение']
 colors=[PALETTE[0],PALETTE[3],PALETTE[1],PALETTE[2],'#8A9298',PALETTE[4]]
 def save(fig,stem):
  fig.savefig(BASE/(stem+'.pdf'),bbox_inches='tight',pad_inches=.1)
  fig.savefig(BUILD/(stem+'.png'),dpi=135,bbox_inches='tight',pad_inches=.1)
  plt.close(fig)
 def shade(ax):
  for x in [T1,T1+p[0],T1+p[0]+p[1]]:ax.axvline(x,color='#8A9298',lw=.7,ls='--',alpha=.7)
  ax.set_xlim(0,tf)
 def stacked(n,height=8.):
  fig,axs=plt.subplots(n,1,figsize=(9.3,height),sharex=True,layout='constrained')
  axs=np.atleast_1d(axs)
  for ax in axs:shade(ax)
  axs[-1].set_xlabel('Время от старта, с')
  return fig,axs
 fig,ax=plt.subplots(figsize=(9.3,5.5),layout='constrained')
 for mode,label,c in zip(['vertical','pitch','turn','raise','coast','circularize'],labels,colors):
  m=d['mode']==mode;ax.plot(RE*d['phi'][m],d['height'][m],color=c,lw=2.8,label=label)
 ax.axhline(TARGET_H,color=PALETTE[0],lw=1,ls=':');ax.set(xlabel='Дальность по поверхности, км',ylabel='Высота, км')
 ax.legend(loc='lower right',ncol=2);save(fig,'ascent-trajectory')
 fig,axs=stacked(3,9.)
 axs[0].plot(t,d['height'],lw=2,color=PALETTE[0]);axs[0].axhline(TARGET_H,lw=1,ls=':',color=PALETTE[4]);axs[0].set_ylabel('Высота, км')
 axs[1].plot(t,d['vr'],lw=2,color=PALETTE[2],label=r'$V_r$');axs[1].plot(t,d['vt'],lw=2,color=PALETTE[1],label=r'$V_t$');axs[1].axhline(VC,lw=1,ls=':',color=PALETTE[4]);axs[1].set_ylabel('Скорость, км/с');axs[1].legend(loc='upper left',ncol=2)
 axs[2].plot(t,d['gamma'],lw=2,color=PALETTE[0],label=r'$\gamma$ — траектория')
 th=d['theta'].copy();th[d['mode']=='coast']=np.nan
 axs[2].plot(t,th,lw=1.6,color=PALETTE[2],label=r'$\theta$ — тяга');axs[2].set_ylabel('Угол, град');axs[2].legend(loc='upper right')
 save(fig,'ascent-state-history')
 fig,axs=stacked(3,8.5)
 axs[0].plot(t,d['z'][4]/1000,lw=2,color=PALETTE[0]);axs[0].set_ylabel('Масса, т')
 axs[1].plot(t,d['force']/1e6,lw=2,color=PALETTE[1]);axs[1].set_ylabel('Тяга, МН')
 axs[2].plot(t,d['load'],lw=2,color=PALETTE[2]);axs[2].axhline(CONSTRAINTS['max_thrust_load_g0'],ls=':',color=PALETTE[4],label='Учебное ограничение');axs[2].set_ylabel(r'$F/(mg_0)$');axs[2].legend(loc='upper left')
 save(fig,'ascent-propulsion')
 fig,axs=plt.subplots(2,1,figsize=(9.3,6.5),sharex=True,layout='constrained')
 early=t<=T1
 axs[0].plot(t[early],d['dynamic_pressure'][early]/1000,lw=2,color=PALETTE[0]);axs[0].axhline(50.,color=PALETTE[4],ls=':',label='50 кПа');axs[0].set_ylabel(r'$q_{\rm dyn}$, кПа');axs[0].legend(loc='upper right')
 axs[1].plot(t[early],d['qalpha'][early]/1000,lw=2,color=PALETTE[2]);axs[1].set_ylabel(r'$q_{\rm dyn}|\alpha|$, кПа·рад');axs[1].set_xlabel('Время от старта, с')
 for ax in axs:ax.set_xlim(0,T1);ax.axvline(TV,color='#8A9298',ls='--',lw=.7);ax.axvline(TV+TP,color='#8A9298',ls='--',lw=.7)
 save(fig,'ascent-aeroloads')
 fig,axs=stacked(2,6.8)
 axs[0].plot(t,d['perigee'],lw=2,color=PALETTE[0],label='Перигей');axs[0].plot(t,d['apogee'],lw=2,color=PALETTE[2],label='Апогей');axs[0].axhline(0,color='#777',lw=.8);axs[0].axhline(TARGET_H,ls=':',color=PALETTE[4],lw=.9);axs[0].set_ylabel('Высота апсиды, км');axs[0].legend(loc='lower right',ncol=2)
 axs[1].plot(t,d['energy'],lw=2,color=PALETTE[1]);axs[1].set_ylabel(r'$\mathcal{E}$, км²/с²')
 save(fig,'ascent-osculating-orbit')
 fig,axs=stacked(2,6.5)
 for k,(name,label,col) in enumerate([('steep','Круче: разворот на 16°',PALETTE[3]),('nominal','Номинал: 20°',PALETTE[0]),('shallow','Пологее: разворот на 24°',PALETTE[2])]):
  c=cases[name];axs[0].plot(c['t'],c['height'],lw=2,color=col,label=label);axs[1].plot(c['t'],c['gamma'],lw=2,color=col)
 axs[0].axhline(TARGET_H,ls=':',color=PALETTE[4],lw=.9);axs[0].set_ylabel('Высота, км');axs[0].legend(loc='lower right')
 axs[1].set_ylabel(r'$\gamma$, град');axs[1].axhline(0,color='#777',lw=.8)
 save(fig,'ascent-program-comparison')
 fig,axs=stacked(2,7.)
 for j,label,col in [(6,'Гравитационная',PALETTE[0]),(7,'Аэродинамическая',PALETTE[1]),(8,'Рулевая',PALETTE[2])]:axs[0].plot(t,1000*d['z'][j],lw=2,color=col,label=label)
 axs[0].set_ylabel('Интеграл потерь, м/с');axs[0].legend(loc='upper left')
 axs[1].plot(t,d['z'][5],lw=2,color=PALETTE[0],label=r'$\Delta V_{\rm th}$');axs[1].plot(t,d['speed'],lw=2,color=PALETTE[1],label=r'$V$');axs[1].set_ylabel('Скорость, км/с');axs[1].legend(loc='upper left')
 save(fig,'ascent-losses')
 fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained')
 hist=report['history'];xx=[v['iteration'] for v in hist];yy=[v['norm'] for v in hist]
 ax.semilogy(xx,yy,'o-',color=PALETTE[0],lw=2);ax.axhline(report['residual_tolerance'],ls=':',color=PALETTE[4],label='Допуск')
 ax.set(xlabel='Номер итерации',ylabel=r'$\|\mathcal{R}\|_2$');ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(loc='upper right');save(fig,'ascent-convergence')
 # Resolve the final active arc on its own time scale.
 fig,axs=plt.subplots(2,1,figsize=(9.3,6.2),sharex=True,layout='constrained')
 m=d['mode']=='circularize'; local=t[m]-t[m][0]
 axs[0].plot(local,1000*d['vr'][m],lw=2.2,color=PALETTE[2]);axs[0].set_ylabel(r'$V_r$, м/с')
 deficit=1000*(d['vt'][m]-np.sqrt(MU/d['radius'][m]))
 axs[1].plot(local,deficit,lw=2.2,color=PALETTE[1]);axs[1].set_ylabel(r'$V_t-\sqrt{\mu/r}$, м/с');axs[1].set_xlabel('Время от начала довыведения, с')
 for ax in axs:ax.axhline(0,color='#8A9298',lw=.8);ax.set_xlim(0,local[-1])
 save(fig,'ascent-final-insertion')


def write_values(report):
 n=report['cases']['nominal'];p=report['solved_parameters_s'];check=report['verification']
 vals={'AscentTOne':T1,'AscentRaise':p[0],'AscentCoast':p[1],'AscentCircularize':p[2],
 'AscentFinalTime':T1+sum(p),'AscentFinalMass':n['mass_kg']/1000,'AscentReserve':n['upper_remaining_propellant_kg']/1000,
 'AscentCharSpeed':n['characteristic_speed_km_s'],'AscentGravityLoss':1000*n['gravity_loss_km_s'],
 'AscentDragLoss':1000*n['drag_loss_km_s'],'AscentSteeringLoss':1000*n['steering_loss_km_s'],
 'AscentMaxQ':report['max_dynamic_pressure_Pa']/1000,'AscentMaxQAlpha':report['max_qalpha_Pa_rad']/1000,
 'AscentMaxLoad':report['max_thrust_load_g0'],'AscentPositionCheck':check['position_difference_m'],
 'AscentVelocityCheck':check['velocity_difference_m_s'],'AscentEccentricity':n['eccentricity'],
 'AscentResidual':float(np.linalg.norm(n['residual_dimensionless'])),'AscentBalanceError':n['speed_balance_error_m_s']}
 lines=['% Generated by ascent_numerics.py; units documented in source.']
 for key,value in vals.items():
  text=f'{value:.3f}' if abs(value)>=.001 else f'{value:.2e}'
  if 'e' in text:
   mant,expo=text.split('e');text=mant+r'\cdot10^{'+str(int(expo))+'}'
  else:text=text.replace('.',r'{,}')
  lines.append('\\newcommand{\\'+key+'}{'+text+'}')
 (BASE/'ascent-values.tex').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--figures-only',action='store_true');args=parser.parse_args()
 if args.figures_only:
  report=json.loads((BASE/'ascent-results.json').read_text());cases={}
  for name in ['nominal','steep','shallow']:
   obj=np.load(BASE/f'ascent-{name}.npz',allow_pickle=False);cases[name]={k:obj[k] for k in obj.files}
 else:report,cases=prepare()
 figures(report,cases);write_values(report)
 print(json.dumps({'parameters_s':report['solved_parameters_s'],'final':report['cases']['nominal'],
  'verification':report['verification'],'constraints':report['checks'],'max_q_Pa':report['max_dynamic_pressure_Pa'],
  'max_load_g0':report['max_thrust_load_g0'],'history':[{'k':h['iteration'],'norm':h['norm']} for h in report['history']]},indent=2))
