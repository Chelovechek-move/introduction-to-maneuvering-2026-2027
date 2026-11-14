"""Explanatory geometry for lecture 10 (no ascent integration).

Run from repository root: python3 10/images/geometry_diagrams.py
Vector PDFs and example data remain in images, PNG QA in ../build.
Orbit curves are exact two-body osculating conics. The sequence, mass
blocks, and coast-insertion drawing are explicitly qualitative diagrams.
The launch-window drawing is the exact projection along Earth's axis
of one spherical parallel and a selected fixed orbit plane.
"""
from pathlib import Path
import os
import json

OUT = Path(__file__).resolve().parent
QA = OUT.parent / "build" / "geometry-qa"
QA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(QA / "mplconfig"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Arc, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path as MplPath

for font in Path("/usr/share/fonts/truetype/paratype").glob("PTF*.ttf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    "font.family": "PT Serif", "font.size": 13,
    "mathtext.fontset": "custom", "mathtext.rm": "PT Serif",
    "mathtext.it": "PT Serif:italic", "mathtext.bf": "PT Serif:bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": "#25323F", "axes.labelcolor": "#25323F",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13,
})
C = {"blue": "#187AA5", "ink": "#25323F", "teal": "#087F78",
     "orange": "#B57714", "purple": "#79539C", "grey": "#7B858B",
     "light": "#D8E0E5", "paper": "#F3F6F7", "earth": "#E6EEF0"}

def arrow(ax, p, v, color="ink", lw=2, scale=13, zorder=5, style="-|>"):
    p, v = np.asarray(p), np.asarray(v)
    ax.add_patch(FancyArrowPatch(p, p+v, arrowstyle=style,
                 mutation_scale=scale, color=C.get(color,color), lw=lw,
                 shrinkA=0, shrinkB=0, zorder=zorder))

def off(ax, xlim, ylim):
    ax.set_aspect("equal")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.axis("off")

def save(fig, stem):
    fig.savefig(OUT / f"geometry-{stem}.pdf", bbox_inches="tight", pad_inches=.08,
                metadata={"Title": "Lecture 10: "+stem, "Author": "Course materials"})
    fig.savefig(QA / f"geometry-{stem}.png", dpi=175, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)

def arc(ax, r, a0, a1, col, p=(0,0), lw=1.3):
    ax.add_patch(Arc(p,2*r,2*r,theta1=a0,theta2=a1,color=C[col],lw=lw))

def orbits():
    re, alt, mu = 6378.137, 650., 398600.4418
    r = re+alt; v = np.sqrt(mu/r)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.3,5.05),gridspec_kw={"width_ratios":[1.2,1]})
    fig.subplots_adjust(left=.02,right=.99,bottom=.23,top=.94,wspace=.25)
    ax.add_patch(Circle((0,0),re,fc=C["earth"],ec=C["grey"],lw=.8))
    ax.add_patch(Circle((0,0),re+100,fill=False,ec=C["grey"],lw=1.1,ls="--"))
    phi=np.linspace(-np.pi,np.pi,1501)
    rows=[]
    for gamma,col in [(0,"blue"),(3,"teal"),(5,"orange")]:
        ga=np.deg2rad(gamma)
        ervec=np.array([-np.sin(ga)**2,-np.sin(ga)*np.cos(ga)])
        e=np.linalg.norm(ervec)
        p=r*np.cos(ga)**2
        rr=p/(1+ervec[0]*np.cos(phi)+ervec[1]*np.sin(phi))
        ax.plot(rr*np.cos(phi),rr*np.sin(phi),color=C[col],lw=1.9,label=f"$\\gamma={gamma}^\\circ$")
        rows.append({"gamma_deg":gamma,"speed_km_s":float(v),"a_km":r,
                     "e":float(e),"perigee_altitude_km":float(r*(1-e)-re),
                     "apogee_altitude_km":float(r*(1+e)-re)})
    ax.plot(r,0,"o",color=C["ink"],ms=4.5,zorder=6)
    for ga,col in [(0,"blue"),(3,"teal"),(5,"orange")]:
        ga=np.deg2rad(ga)
        arrow(ax,(r,0),1800*np.array([np.sin(ga),np.cos(ga)]),col,lw=1.1,scale=8)
    ax.text(r+220,-260,"$P$",fontsize=14)
    ax.text(-650,-200,"Земля",ha="center",fontsize=13,color=C["grey"])
    ax.legend(loc="upper left",frameon=False,fontsize=11,borderpad=0)
    off(ax,(-8700,8800),(-7900,8700))
    angles=np.linspace(0,6,301)
    hp=alt-r*np.sin(np.deg2rad(angles))
    bx.axhspan(0,100,color=C["light"],alpha=.65)
    bx.axhline(0,color=C["grey"],lw=.8)
    bx.axhline(100,color=C["grey"],lw=.9,ls="--")
    bx.plot(angles,hp,color=C["blue"],lw=2.2)
    for row,col in zip(rows,["blue","teal","orange"]):
        x,y=row["gamma_deg"],row["perigee_altitude_km"]
        bx.plot(x,y,"o",color=C[col],ms=5)
        if x==0: bx.annotate("650 км",(x,y),xytext=(12,-5),textcoords="offset points",color=C[col])
        elif x==3: bx.annotate("282 км",(x,y),xytext=(9,10),textcoords="offset points",color=C[col])
        else: bx.annotate("37,5 км",(x,y),xytext=(-7,-24),textcoords="offset points",ha="right",color=C[col])
    bx.set_xlim(-.15,6);bx.set_ylim(-110,710)
    bx.set_xlabel("Угол над горизонтом $\\gamma$, град")
    bx.set_ylabel("Высота перигея $H_p$, км")
    bx.grid(alpha=.2);bx.text(5.9,117,"100 км",ha="right",fontsize=10,color=C["grey"])
    fig.text(.5,.035,r"$H=650\ \mathrm{км},\quad V=\sqrt{\mu/r}=7{,}531\ \mathrm{км/с},\quad e=|\sin\gamma|$",ha="center",fontsize=13)
    save(fig,"orbits")
    (OUT/"geometry-example.json").write_text(json.dumps({"mu_km3_s2":mu,"earth_radius_km":re,"altitude_km":alt,"model":"two-body; same point and circular speed magnitude; outward radial velocity for gamma>0","cases":rows},ensure_ascii=False,indent=2)+"\n")

def local_frame():
    fig,ax=plt.subplots(figsize=(8.2,4.75))
    fig.subplots_adjust(left=.025,right=.99,bottom=.20,top=.97)
    gamma,theta=np.deg2rad([25,52]);speed=3.3
    V=speed*np.array([np.cos(gamma),np.sin(gamma)])
    U=2.85*np.array([np.cos(theta),np.sin(theta)])
    arrow(ax,(0,0),(4.15,0),"grey",lw=1.2)
    arrow(ax,(0,0),(0,3.25),"grey",lw=1.2)
    ax.text(4.1,-.13,r"$\mathbf{e}_t$",ha="right",va="top",fontsize=15)
    ax.text(-.12,3.20,r"$\mathbf{e}_r$",ha="right",fontsize=15)
    ax.plot([V[0],V[0]],[0,V[1]],ls="--",lw=1,color=C["light"])
    ax.plot([0,V[0]],[V[1],V[1]],ls="--",lw=1,color=C["light"])
    arrow(ax,(0,0),V,"orange",lw=2.6)
    arrow(ax,(0,0),U,"blue",lw=2.2)
    ax.text(V[0]+.12,V[1]-.02,r"$\mathbf{V}$",color=C["orange"],fontsize=16)
    ax.text(U[0]-.12,U[1]+.10,r"$\mathbf{u}$",ha="right",color=C["blue"],fontsize=16)
    arrow(ax,(0,-.045),(V[0],0),"teal",lw=1.7,scale=11)
    arrow(ax,(-.045,0),(0,V[1]),"purple",lw=1.7,scale=11)
    ax.text(V[0]/2,-.23,r"$V_t$",ha="center",va="top",color=C["teal"],fontsize=15)
    ax.text(-.16,V[1]*.72,r"$V_r$",ha="right",color=C["purple"],fontsize=15)
    arc(ax,.95,0,52,"blue");ax.text(.69,.67,r"$\theta$",color=C["blue"],fontsize=15)
    arc(ax,1.55,0,25,"orange");ax.text(1.62,.25,r"$\gamma$",color=C["orange"],fontsize=15)
    arc(ax,2.10,25,52,"purple");ax.text(1.68,1.48,r"$\beta$",color=C["purple"],fontsize=15)
    ax.plot(0,0,"o",color=C["ink"],ms=4)
    off(ax,(-.7,4.4),(-.5,3.4))
    fig.text(.5,.075,r"$\beta=\theta-\gamma,\qquad V_r=V\sin\gamma,\qquad V_t=V\cos\gamma$",ha="center",fontsize=16)
    save(fig,"local-frame")

def bezier(ax,verts,col,lw=3,ls="-"):
    path=MplPath(verts,[MplPath.MOVETO]+[MplPath.CURVE4]*(len(verts)-1))
    ax.add_patch(PathPatch(path,fill=False,lw=lw,ec=C[col],ls=ls,capstyle="round"))

def sequence():
    fig,ax=plt.subplots(figsize=(9.5,3.9))
    fig.subplots_adjust(left=.01,right=.99,bottom=.10,top=.97)
    x=np.linspace(-.4,8.8,501);ground=-.02-.015*x*x
    ax.fill_between(x,-.6,ground,color=C["earth"])
    ax.plot(x,ground,color=C["grey"],lw=1)
    bezier(ax,[(0,0),(0,.20),(0,.39),(0,.60)],"blue")
    bezier(ax,[(0,.60),(0,.90),(.21,1.12),(.48,1.33)],"teal")
    bezier(ax,[(.48,1.33),(1.3,1.98),(2.4,2.37),(3.65,2.55)],"blue")
    bezier(ax,[(3.65,2.55),(4.5,2.68),(5.6,2.95),(6.65,2.95)],"grey",lw=2.3,ls="--")
    bezier(ax,[(6.65,2.95),(7.13,2.95),(7.55,2.97),(8.0,2.96)],"orange")
    ax.plot([8,8.8],[2.96,2.94],color=C["blue"],lw=1.2)
    for p in [(0,0),(.48,1.33),(3.65,2.55),(6.65,2.95),(8,2.96)]:
        ax.plot(*p,"o",ms=4,color=C["ink"])
    labels=[((0,.28),(-.12,1.2),"Вертикальный\nучасток"),
            ((.22,1.08),(1.6,.63),"Начальный\nразворот"),
            ((2.0,2.16),(2.8,1.12),"Гравитационный\nразворот"),
            ((5.2,2.84),(5.25,1.9),"Пассивный\nучасток"),
            ((7.3,2.95),(7.55,1.85),"Довыведение")]
    for p,t,label in labels:
        ax.annotate(label,xy=p,xytext=t,ha="center",va="center",fontsize=12,
                    arrowprops={"arrowstyle":"-","color":C["grey"],"lw":.8})
    ax.text(8.5,3.18,"Орбита",ha="center",color=C["blue"],fontsize=12)
    ax.text(5.0,-.34,"Схема последовательности; масштабы условные",ha="center",fontsize=11,color=C["grey"])
    off(ax,(-.9,9.0),(-.6,3.5));save(fig,"sequence")

def gravity_turn():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.0,4.4),gridspec_kw={"width_ratios":[1.15,1]})
    fig.subplots_adjust(left=.01,right=.99,bottom=.15,top=.88,wspace=.20)
    ax.set_title("Сложение приращений скорости",pad=10)
    v=np.array([2.,1.65]);dv_t=.38*v;dv_g=np.array([0,-1.05]);vnew=v+dv_t+dv_g
    arrow(ax,(0,0),(3.5,0),"grey",lw=.8)
    arrow(ax,(0,0),v,"orange",lw=2.2)
    arrow(ax,v,dv_t,"teal",lw=2)
    arrow(ax,v+dv_t,dv_g,"purple",lw=2)
    arrow(ax,(0,0),vnew,"blue",lw=2.2)
    ax.text(1.28,1.35,r"$\mathbf{V}$",color=C["orange"],fontsize=15)
    ax.text(2.05,2.23,r"$\Delta\mathbf{V}_F\parallel\mathbf{V}$",color=C["teal"],fontsize=13,ha="center")
    ax.text(2.91,1.85,r"$\Delta\mathbf{V}_g$",color=C["purple"],fontsize=14)
    ax.text(vnew[0]+.10,vnew[1]-.13,r"$\mathbf{V}+\Delta\mathbf{V}$",color=C["blue"],fontsize=14)
    a=np.rad2deg(np.arctan2(v[1],v[0]));b=np.rad2deg(np.arctan2(vnew[1],vnew[0]))
    arc(ax,.75,0,b,"blue");arc(ax,1.13,0,a,"orange")
    ax.text(.93,.28,r"$\gamma+\Delta\gamma$",color=C["blue"],fontsize=12)
    ax.text(.80,.83,r"$\gamma$",color=C["orange"],fontsize=14)
    off(ax,(-.15,4.1),(-.2,2.65))
    bx.set_title("Вращение местного горизонта",pad=10)
    ph=np.linspace(-20,75,200)*np.pi/180
    bx.plot(2.7*np.cos(ph),2.7*np.sin(ph),color=C["light"],lw=1.6)
    P0=np.array([2.7,0.]);ang=np.deg2rad(30);P1=2.7*np.array([np.cos(ang),np.sin(ang)])
    for P,phi,col in [(P0,0,"orange"),(P1,ang,"blue")]:
        bx.plot([0,P[0]],[0,P[1]],color=C["grey"],ls="--",lw=.8)
        et=np.array([-np.sin(phi),np.cos(phi)])
        arrow(bx,P,1.1*et,col,lw=2)
        bx.plot(*P,"o",ms=4,color=C[col])
    bx.text(2.8,.58,r"$\mathbf{e}_t(t)$",color=C["orange"],fontsize=13)
    bx.text(1.23,2.26,r"$\mathbf{e}_t(t+\Delta t)$",color=C["blue"],fontsize=13)
    arc(bx,.80,0,30,"purple");bx.text(.91,.18,r"$\Delta\varphi$",color=C["purple"],fontsize=14)
    bx.text(-.1,-.12,"$O$",ha="right",fontsize=13)
    off(bx,(-.25,3.7),(-.30,2.8))
    fig.text(.5,.03,r"$\dot\gamma=\left(\frac{V}{r}-\frac{\mu}{V r^2}\right)\cos\gamma\quad$ при тяге вдоль скорости",ha="center",fontsize=14)
    save(fig,"gravity-turn")

def staging():
    fig,ax=plt.subplots(figsize=(9.3,4.8))
    fig.subplots_adjust(left=.025,right=.99,bottom=.10,top=.92)
    xs=[.0,2.25,4.5,6.75];width=1.25
    blocks=[[(.5,"orange",r"$m_L$"),(.38,"grey",r"$m_{s,2}$"),(1.6,"blue",r"$m_{p,2}$"),(.55,"purple",r"$m_{s,1}$"),(3.2,"teal",r"$m_{p,1}$")],
            [(.5,"orange",r"$m_L$"),(.38,"grey",r"$m_{s,2}$"),(1.6,"blue",r"$m_{p,2}$"),(.55,"purple",r"$m_{s,1}$")],
            [(.5,"orange",r"$m_L$"),(.38,"grey",r"$m_{s,2}$"),(1.6,"blue",r"$m_{p,2}$")],
            [(.5,"orange",r"$m_L$"),(.38,"grey",r"$m_{s,2}$"),(.55,"blue",r"$m_{p,2}^{\rm ост}$")]]
    # Put payload at top and propellant stage 1 at bottom, with common scale.
    titles=["Старт","Выгорание\nступени 1","Сброс\nступени 1","Выключение\nступени 2"]
    for x,bs,title in zip(xs,blocks,titles):
        y=0
        for h,col,label in reversed(bs):
            ax.add_patch(Rectangle((x,y),width,h,fc=C[col],ec="white",lw=1.2,alpha=.9))
            ax.text(x+width/2,y+h/2,label,color="white",ha="center",va="center",fontsize=13)
            y+=h
        ax.text(x+width/2,-.28,title,ha="center",va="top",fontsize=12)
    arrow(ax,(3.75,.55),(.44,-.48),"purple",lw=1.3,scale=10)
    ax.text(4.04,-.17,r"$-m_{s,1}$",ha="center",fontsize=12,color=C["purple"])
    for x in [1.52,3.78,6.03]: arrow(ax,(x,2.15),(.45,0),"grey",lw=1.2,scale=10)
    ax.text(6.9,5.80,"$p$ — топливо\n$s$ — сухая масса\n$L$ — нагрузка",ha="center",va="top",fontsize=12,linespacing=1.5)
    ax.text(4,6.55,"Массовые блоки показаны условно",ha="center",fontsize=11,color=C["grey"])
    off(ax,(-.25,8.2),(-1.0,6.9));save(fig,"staging")

def coast_insertion():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9,4.4),gridspec_kw={"width_ratios":[1.45,1]})
    fig.subplots_adjust(left=.01,right=.99,bottom=.16,top=.94,wspace=.10)
    ra,rp,re=1.65,1.0,.76;a=(ra+rp)/2;e=(ra-rp)/(ra+rp);b=a*np.sqrt(1-e*e)
    ph=np.linspace(-np.pi,np.pi,501)
    ax.add_patch(Circle((0,0),re,fc=C["earth"],ec=C["grey"],lw=.8))
    ax.plot(ra*np.cos(ph),ra*np.sin(ph),color=C["blue"],lw=1.6)
    E=np.linspace(-np.pi,np.pi,600)
    ax.plot(a*e+a*np.cos(E),b*np.sin(E),color=C["light"],lw=1.0)
    Ec=np.linspace(-2.0,0,160)
    xx=a*e+a*np.cos(Ec);yy=b*np.sin(Ec)
    ax.plot(xx,yy,color=C["orange"],lw=2.7)
    j=90;arrow(ax,(xx[j],yy[j]),np.array([xx[j+8]-xx[j],yy[j+8]-yy[j]]),"orange",lw=2.4,scale=12)
    ax.plot(ra,0,"o",color=C["ink"],ms=4.5)
    ax.plot([0,ra],[0,0],color=C["grey"],lw=.9,ls="--")
    ax.text(ra+.08,-.18,"Апогей",ha="left",va="top",fontsize=12)
    arrow(ax,(ra,0),(0,.7),"orange",lw=2)
    ax.text(ra+.08,.58,r"$\mathbf{V}^-$",color=C["orange"],fontsize=14)
    ax.text(.55,-1.39,"Пассивная дуга",color=C["orange"],fontsize=12,ha="center")
    ax.text(-1.58,1.38,"Целевая окружность",color=C["blue"],fontsize=12)
    ax.text(.83,.12,r"$r_a$",color=C["grey"],fontsize=13)
    off(ax,(-1.95,2.6),(-1.82,1.9))
    arrow(bx,(0,0),(0,2.5),"blue",lw=4,scale=16)
    arrow(bx,(0,0),(0,1.7),"orange",lw=2.6,scale=14)
    arrow(bx,(.28,1.7),(0,.8),"teal",lw=2,scale=12)
    bx.plot([0,.40],[1.7,1.7],color=C["grey"],ls="--",lw=.7)
    bx.plot([0,.40],[2.5,2.5],color=C["grey"],ls="--",lw=.7)
    bx.text(-.18,1.12,r"$V^-_t$",ha="right",color=C["orange"],fontsize=15)
    bx.text(-.18,2.36,r"$V^+_t=V_{\rm circ}$",ha="right",color=C["blue"],fontsize=14)
    bx.text(.45,2.03,r"$\Delta V_t$",color=C["teal"],fontsize=15)
    bx.text(0,-.15,r"$V_r^-=V_r^+=0$",ha="center",fontsize=13)
    off(bx,(-1.9,1.8),(-.45,2.9))
    fig.text(.5,.025,r"$ \Delta V_t=\sqrt{\frac{\mu}{r_a}}-\sqrt{\mu\left(\frac{2}{r_a}-\frac{1}{a}\right)}$",ha="center",fontsize=17)
    save(fig,"coast-insertion")

def launch_plane():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.1,4.65))
    fig.subplots_adjust(left=.01,right=.99,bottom=.18,top=.90,wspace=.12)
    ax.set_title("Меридиональная плоскость",pad=8)
    ax.add_patch(Circle((0,0),1,fc=C["earth"],ec=C["grey"],lw=.8))
    arrow(ax,(0,-1.17),(0,2.65),"grey",lw=1,scale=11)
    arrow(ax,(-1.18,0),(2.65,0),"grey",lw=1,scale=11)
    ax.text(.06,1.46,"$z$",fontsize=14);ax.text(1.40,-.12,"Экватор",ha="right",fontsize=11)
    phi=np.deg2rad(37);er=np.array([np.cos(phi),np.sin(phi)]);en=np.array([-np.sin(phi),np.cos(phi)])
    P=er
    ax.plot([0,P[0]],[0,P[1]],color=C["ink"],lw=1.3)
    ax.plot(*P,"o",color=C["ink"],ms=4)
    arrow(ax,P,.55*er,"blue",lw=1.8)
    arrow(ax,P,.67*en,"teal",lw=1.8)
    ax.text(*(P+.59*er),r"$\mathbf{e}_r$",fontsize=14,color=C["blue"])
    ax.text(*(P+.71*en),r"$\mathbf{e}_N$",ha="center",fontsize=14,color=C["teal"])
    ax.add_patch(Circle(P,.045,fc="white",ec=C["purple"],lw=1.2,zorder=8));ax.plot(*P,".",color=C["purple"],ms=3,zorder=9)
    ax.text(P[0]+.08,P[1]-.17,r"$\mathbf{e}_E$",color=C["purple"],fontsize=14)
    arc(ax,.42,0,37,"orange");ax.text(.47,.12,r"$\phi$",color=C["orange"],fontsize=14)
    ax.text(-.1,-.12,"$O$",ha="right",fontsize=13)
    off(ax,(-1.25,1.72),(-1.22,1.67))
    bx.set_title("Местный горизонт",pad=8)
    arrow(bx,(-1.6,0),(3.95,0),"grey",lw=1,scale=11)
    arrow(bx,(0,-.30),(0,2.8),"grey",lw=1,scale=11)
    bx.text(2.27,-.14,"Восток",ha="right",fontsize=12)
    bx.text(.10,2.38,"Север",fontsize=12)
    A=np.deg2rad(58);eh=np.array([np.sin(A),np.cos(A)]);normal=np.array([-np.cos(A),np.sin(A)])
    arrow(bx,(0,0),2.1*eh,"blue",lw=2.4)
    arrow(bx,(0,0),1.95*normal,"purple",lw=2)
    bx.text(*(2.20*eh),r"$\mathbf{e}_A$",color=C["blue"],fontsize=15)
    bx.text(*(2.05*normal),r"$\mathbf{n}$",color=C["purple"],ha="right",fontsize=15)
    bx.plot([normal[0]*1.95,0],[normal[1]*1.95,normal[1]*1.95],ls="--",lw=.9,color=C["light"])
    bx.text(.1,normal[1]*1.95,r"$\sin A$",color=C["purple"],fontsize=12)
    arc(bx,.78,90-58,90,"orange");bx.text(.45,.75,"$A$",color=C["orange"],fontsize=15)
    bx.add_patch(Circle((0,0),.055,fc="white",ec=C["ink"],lw=1));bx.plot(0,0,".",color=C["ink"],ms=3)
    bx.text(-.11,-.22,r"$\mathbf{e}_r$",ha="right",fontsize=13)
    off(bx,(-1.6,2.4),(-.4,2.65))
    fig.text(.5,.04,r"$\mathbf{n}=\mathbf{e}_r\times\mathbf{e}_A=\sin A\,\mathbf{e}_N-\cos A\,\mathbf{e}_E,\qquad \cos i=\cos\phi\,\sin A$",ha="center",fontsize=14)
    save(fig,"launch-plane")

def rotation():
    fig,ax=plt.subplots(figsize=(8.0,4.7))
    fig.subplots_adjust(left=.035,right=.98,bottom=.17,top=.96)
    vair=np.array([2.7,3.7]);vatm=np.array([1.20,0]);v=vair+vatm
    arrow(ax,(-.20,0),(5.4,0),"grey",lw=1,scale=11)
    arrow(ax,(0,-.20),(0,4.85),"grey",lw=1,scale=11)
    ax.text(5.05,-.18,"Восток",ha="right",fontsize=12)
    ax.text(-.08,4.65,"Север",ha="right",fontsize=12)
    arrow(ax,(0,0),vair,"teal",lw=2.6)
    arrow(ax,vair,vatm,"purple",lw=2.2)
    arrow(ax,(0,0),v,"blue",lw=2.6)
    ax.text(1.35,2.32,r"$\mathbf{V}_{\rm air}$",ha="right",color=C["teal"],fontsize=16)
    ax.text(3.32,3.92,r"$\mathbf{V}_{\rm atm}=\mathbf{\omega}_E\times\mathbf{R}$",ha="center",color=C["purple"],fontsize=14)
    ax.text(3.45,2.55,r"$\mathbf{V}$",color=C["blue"],fontsize=17)
    airang=np.rad2deg(np.arctan2(vair[1],vair[0]));ang=np.rad2deg(np.arctan2(v[1],v[0]))
    arc(ax,1.05,airang,90,"teal");arc(ax,1.60,ang,90,"blue")
    ax.text(.44,1.04,r"$A_{\rm air}$",color=C["teal"],fontsize=13)
    ax.text(1.08,1.42,"$A$",color=C["blue"],fontsize=15)
    ax.plot(0,0,"o",color=C["ink"],ms=4)
    ax.text(-.10,-.17,"$P$",ha="right",fontsize=13)
    ax.text(3.1,.27,"Ветер отсутствует;\nмасштаб векторов условный",ha="center",fontsize=11,color=C["grey"])
    off(ax,(-.75,5.3),(-.40,4.85))
    fig.text(.5,.035,r"$\mathbf{V}=\mathbf{V}_{\rm air}+\mathbf{\omega}_E\times\mathbf{R}$",ha="center",fontsize=17)
    save(fig,"rotation")

def launch_window():
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    fig.subplots_adjust(left=.025,right=.99,bottom=.12,top=.96)
    phi=np.deg2rad(30);incl=np.deg2rad(60);rho=np.cos(phi);z=np.sin(phi)
    yy=z/np.tan(incl);xx=np.sqrt(rho*rho-yy*yy)
    ax.add_patch(Circle((0,0),rho,fc="white",ec=C["orange"],lw=2.3))
    ax.plot([-1.18,1.12],[yy,yy],color=C["blue"],lw=2)
    arrow(ax,(-1.16,0),(2.28,0),"grey",lw=.8,scale=10)
    arrow(ax,(0,-1.03),(0,2.25),"grey",lw=.8,scale=10)
    ax.text(1.13,-.10,"$X$",fontsize=13);ax.text(.06,1.17,"$Y$",fontsize=13)
    ax.plot(0,0,".",color=C["ink"],ms=6)
    ax.text(-.06,-.1,"$O'$",ha="right",fontsize=12)
    for P,lab in [(np.array([xx,yy]),"A"),(np.array([-xx,yy]),"B")]:
        ax.plot(*P,"o",ms=6,color=C["blue"])
        ax.plot([0,P[0]],[0,P[1]],ls="--",lw=.9,color=C["light"])
        ax.text(P[0]+(.05 if lab=="A" else -.06),P[1]+.08,f"${lab}$",fontsize=14,ha="left" if lab=="A" else "right",color=C["blue"])
    t=np.deg2rad(-40);P=rho*np.array([np.cos(t),np.sin(t)])
    ax.plot(*P,"o",ms=5,color=C["orange"])
    ax.text(P[0]+.08,P[1]-.06,"Старт вне плоскости",color=C["orange"],fontsize=12,va="top")
    t0,t1=np.deg2rad([75,105]);p0=rho*np.array([np.cos(t0),np.sin(t0)]);p1=rho*np.array([np.cos(t1),np.sin(t1)])
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=14,
        connectionstyle="arc3,rad=.1",lw=2,color=C["orange"]))
    ax.text(-.15,1.02,r"$\omega_E$",color=C["orange"],ha="center",fontsize=15)
    ax.annotate("След целевой плоскости\nна плоскости параллели",xy=(-.23,yy),xytext=(-.35,-.42),ha="center",fontsize=12,color=C["blue"],arrowprops={"arrowstyle":"-","color":C["blue"],"lw":.8})
    ax.text(2.04,.85,"Вид вдоль оси Земли",ha="center",fontsize=13)
    ax.text(2.04,.30,"Точка старта движется\nпо параллели",ha="center",fontsize=12,color=C["orange"],linespacing=1.4)
    ax.text(2.04,-.25,"Целевая плоскость\nфиксирована в ИСК",ha="center",fontsize=12,color=C["blue"],linespacing=1.4)
    off(ax,(-1.25,2.9),(-1.10,1.25))
    fig.text(.5,.025,r"$z=R_E\sin\phi=\mathrm{const},\qquad \Lambda(t)=\Lambda_0+\omega_E t$",ha="center",fontsize=14)
    save(fig,"launch-window")

def geometry_reference_numbers():
    """Reproduce the numerical local-velocity/window example in the text."""
    p=OUT/"geometry-example.json"
    data=json.loads(p.read_text())
    mu,re,omega=398600.4418,6378.137,7.2921159e-5
    phi,i=np.deg2rad([28.5,51.6]);alt=200.;r=re+alt
    speed=np.sqrt(mu/r);A=np.arcsin(np.cos(i)/np.cos(phi))
    vn=speed*np.cos(A);ve=speed*np.sin(A)
    vatm=omega*r*np.cos(phi);vair=np.hypot(vn,ve-vatm)
    aair=np.arctan2(ve-vatm,vn)
    delta=np.arcsin(np.tan(phi)/np.tan(i))
    dt=(np.pi-2*delta)/omega/3600
    sidereal=2*np.pi/omega/3600
    data["launch_geometry_example"]={
        "mu_km3_s2":mu,"earth_radius_km":re,"omega_rad_s":omega,
        "latitude_deg":28.5,"inclination_deg":51.6,"altitude_km":alt,
        "circular_inertial_speed_km_s":float(speed),
        "inertial_azimuth_deg":float(np.rad2deg(A)),
        "inertial_north_speed_km_s":float(vn),
        "inertial_east_speed_km_s":float(ve),
        "atmosphere_east_speed_at_altitude_km_s":float(vatm),
        "air_speed_km_s":float(vair),"air_azimuth_deg":float(np.rad2deg(aair)),
        "initial_pad_speed_km_s":float(omega*re*np.cos(phi)),
        "launch_longitude_offset_deg":float(np.rad2deg(delta)),
        "successive_plane_crossing_intervals_h":[float(dt),float(sidereal-dt)],
        "model":"spherical Earth, co-rotating atmosphere without wind; local horizontal circular state; no claim about constant heading during ascent"
    }
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")

if __name__ == "__main__":
    orbits();local_frame();sequence();gravity_turn();staging()
    coast_insertion();launch_plane();rotation();launch_window();geometry_reference_numbers()
    print("Wrote 9 geometric PDF figures and geometry-example.json")
