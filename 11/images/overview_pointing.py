#!/usr/bin/env python3
"""Pointing estimates and exact plane changes for two apsidal manoeuvres.

Run from the course root: python3 11/images/overview_pointing.py
All angles are teaching assumptions. The estimate is not a sensor specification.
The left panel zooms the ends of vectors with a common origin outside the panel.
The right panel compares the angle BETWEEN orbital planes, not inclination alone.
"""
from pathlib import Path
import json
import os

BASE=Path(__file__).resolve().parent
BUILD=BASE.parent/'build'
PREVIEW=BUILD/'overview-preview'
PREVIEW.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(BUILD/'mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE,GREEN,ORANGE,PURPLE='#187AA5','#087F78','#B57714','#79539C'
GRAY,INK='#687582','#25323F'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':14,
    'axes.labelsize':14,'legend.fontsize':12,
    'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white',
    'pdf.fonttype':42,'ps.fonttype':42})

def main():
    mu,rp,ra,D=398600.4418,7000.,42164.,100.
    a=(rp+ra)/2
    vp,va=np.sqrt(mu*(2/np.array([rp,ra])-1/a))*1000
    theta=np.deg2rad(1.)
    alpha=np.rad2deg(np.arctan2(D*np.sin(theta),np.array([vp,va])+D*np.cos(theta)))
    d={'mu_km3_s2':mu,'rp_km':rp,'ra_km':ra,'D_m_s':D,
       'true_pointing_error_deg':1.,'estimated_pointing_error_deg':.7,
       'speeds_perigee_apogee_m_s':[float(vp),float(va)],
       'plane_angle_perigee_apogee_deg':alpha.tolist(),
       'note':'Separate 100 m/s manoeuvres; same initial orbit and normal pointing error. All vectors on left share the same origin. Sensor estimate is illustrative.'}
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.2),gridspec_kw={'width_ratios':[1.1,1.]})
    ax=axes[0]
    for deg,color,label,pos in [(0,BLUE,'Требуемое направление',(98.05,-.28)),
        (.7,PURPLE,'Оценённое: 0,7°',(98.05,.72)),
        (1.,ORANGE,'Фактическое: 1°',(98.05,2.10))]:
        t=np.deg2rad(deg);v=D*np.array([np.cos(t),np.sin(t)])
        visible_start=(97.95,97.95*np.tan(t))
        ax.annotate('',xy=v,xytext=visible_start,
                    arrowprops={'arrowstyle':'-|>','lw':2,'color':color})
        ax.plot(*v,'o',ms=4,color=color)
        ax.annotate(label,xy=v,xytext=pos,fontsize=11.5,color=color,
                    arrowprops={'arrowstyle':'-','lw':.8,'color':color},va='center')
    ax.set_xlim(97.9,100.35);ax.set_ylim(-.48,2.28);ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta V_t$, м/с');ax.set_ylabel(r'$\Delta V_n$, м/с')
    ax.set_title('а) Увеличение концов векторов',loc='left',fontsize=13.5,pad=14)
    ax.grid(alpha=.16)
    ax=axes[1]
    xs=np.arange(2)
    ax.bar(xs,alpha,color=[BLUE,GREEN],width=.52,alpha=.8)
    for x,y in zip(xs,alpha):
        ax.text(x,y+.002,format(y,'.4f').replace('.',','),ha='center',fontsize=12)
    ax.set_xticks(xs,['Манёвр\nв перигее','Манёвр\nв апогее'],fontsize=12)
    ax.set_ylabel(r'Угол между плоскостями $\alpha$, град')
    ax.set_ylim(0,.07);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.set_title('б) Результат одной ошибки 1°',loc='left',fontsize=13.5,pad=14)
    ax.text(.5,.91,'Одинаковый модуль: 100 м/с',transform=ax.transAxes,
            ha='center',fontsize=11.5,color=INK)
    fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.87,wspace=.43)
    fig.savefig(BASE/'overview-pointing.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(PREVIEW/'overview-pointing.png',dpi=170,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)
    (BASE/'overview-pointing-data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
