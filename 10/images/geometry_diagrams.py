"""Explanatory geometry for lecture 10 (no ascent integration).

Run from repository root: python3 10/images/geometry_diagrams.py
Vector PDFs and example data remain in images, PNG QA in ../build.
Orbit curves are exact two-body osculating conics. The sequence, mass
blocks, and coast-insertion drawing are explicitly qualitative diagrams.
Launch geometry uses orthographic projections of spherical-Earth
vectors. The launch-window drawing shows a fixed orbit plane and a
daily latitude plane in 3D, then their intersection viewed along +Z.
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
    """Global polar angle and the same local orbital frame in one geometry.

    Positive X points right and varphi grows counterclockwise.
    The requested local view has e_r up-left and e_t up-right:
    e_r=(cos(varphi),sin(varphi)), e_t=(sin(varphi),-cos(varphi)).
    Thus d e_r/d varphi=-e_t in this particular view.
    """
    fig,ax=plt.subplots(figsize=(8.4,6.1))
    fig.subplots_adjust(left=.025,right=.99,bottom=.14,top=.98)
    polar=np.deg2rad(122.)
    er=np.array([np.cos(polar),np.sin(polar)])
    et=np.array([np.sin(polar),-np.cos(polar)])
    P=4.3*er
    gamma,theta=np.deg2rad([25.,52.]);speed=3.3
    V=speed*(np.cos(gamma)*et+np.sin(gamma)*er)
    U=2.85*(np.cos(theta)*et+np.sin(theta)*er)

    # Inertial origin and polar position: varphi has its vertex at O.
    arrow(ax,(0,0),(3.25,0),"grey",lw=1.1,scale=11)
    ax.text(3.30,-.08,"$X$",ha="left",va="top",fontsize=17)
    arrow(ax,(0,0),P,"ink",lw=1.65,scale=12)
    midpoint=.54*P-.32*et
    ax.text(*midpoint,r"$\mathbf{R}$",ha="center",va="center",color=C["ink"],fontsize=18)
    phi_points=1.12*np.c_[np.cos(np.linspace(0,polar,90)),
                             np.sin(np.linspace(0,polar,90))]
    ax.plot(phi_points[:,0],phi_points[:,1],color=C["ink"],lw=1.25)
    arrow(ax,phi_points[-5],phi_points[-1]-phi_points[-5],"ink",lw=1.25,scale=10)
    ax.text(1.40*np.cos(polar/2),1.40*np.sin(polar/2),
            r"$\varphi$",fontsize=18,ha="center",va="center")
    ax.plot(0,0,"o",color=C["ink"],ms=4.3)
    ax.text(.08,-.14,"$O$",ha="left",va="top",fontsize=17)
    ax.text(-.02,-.37,"Центр Земли",ha="right",va="top",fontsize=13,color=C["grey"])

    # The radial continuation and its perpendicular local tangent.
    arrow(ax,P,3.10*er,"grey",lw=1.15,scale=12)
    arrow(ax,P,3.55*et,"grey",lw=1.15,scale=12)
    ax.text(*(P+3.27*er),r"$\mathbf{e}_r$",ha="center",va="center",fontsize=18)
    ax.text(*(P+3.73*et),r"$\mathbf{e}_t$",ha="center",va="center",fontsize=18)
    # Small square makes the orthogonality explicit, away from the angles.
    sq=np.array([P+.19*er,P+.19*(er+et),P+.19*et])
    ax.plot(sq[:,0],sq[:,1],color=C["grey"],lw=.8)

    # Components and their projection rectangle remain tied to this basis.
    radial=speed*np.sin(gamma)*er;transverse=speed*np.cos(gamma)*et
    radial_end,transverse_end,velocity_end=P+radial,P+transverse,P+V
    ax.plot([radial_end[0],velocity_end[0]],
            [radial_end[1],velocity_end[1]],ls="--",lw=1,color=C["light"])
    ax.plot([transverse_end[0],velocity_end[0]],
            [transverse_end[1],velocity_end[1]],ls="--",lw=1,color=C["light"])
    arrow(ax,P,radial,"purple",lw=1.8,scale=11)
    arrow(ax,P,transverse,"teal",lw=1.8,scale=11)
    ax.text(*(P+.66*radial-.21*et),r"$V_r$",ha="center",va="center",color=C["purple"],fontsize=17)
    ax.text(*(P+.67*transverse-.22*er),r"$V_t$",ha="center",va="center",color=C["teal"],fontsize=17)
    arrow(ax,P,V,"orange",lw=2.6,scale=14)
    arrow(ax,P,U,"blue",lw=2.3,scale=14)
    ax.text(*(P+V+np.array([.13,.10])),r"$\mathbf{V}$",ha="left",va="bottom",color=C["orange"],fontsize=19)
    ax.text(*(P+U+np.array([-.11,.04])),r"$\mathbf{u}$",ha="right",va="bottom",color=C["blue"],fontsize=19)

    tangent_angle=np.rad2deg(polar)-90
    velocity_angle=tangent_angle+np.rad2deg(gamma)
    thrust_angle=tangent_angle+np.rad2deg(theta)
    arc(ax,.88,tangent_angle,thrust_angle,"blue",p=P)
    arc(ax,1.57,tangent_angle,velocity_angle,"orange",p=P)
    arc(ax,2.17,velocity_angle,thrust_angle,"purple",p=P)
    def label_at(radius,angle,label,col):
        direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        ax.text(*(P+radius*direction),label,color=C[col],fontsize=18,ha="center",va="center")
    label_at(1.06,70,r"$\theta$","blue")
    label_at(1.83,44,r"$\gamma$","orange")
    label_at(2.36,70,r"$\beta$","purple")
    ax.plot(*P,"o",color=C["ink"],ms=4.5)
    ax.text(*(P+np.array([-.20,-.12])),"$P$",ha="right",va="top",fontsize=16)
    off(ax,(-4.6,3.9),(-.65,7.15))
    fig.text(.5,.035,r"$\beta=\theta-\gamma,\qquad V_r=V\sin\gamma,\qquad V_t=V\cos\gamma$",ha="center",fontsize=17)
    save(fig,"local-frame")

def bezier(ax,verts,col,lw=3,ls="-"):
    path=MplPath(verts,[MplPath.MOVETO]+[MplPath.CURVE4]*(len(verts)-1))
    ax.add_patch(PathPatch(path,fill=False,lw=lw,ec=C[col],ls=ls,capstyle="round"))

def sequence():
    """Overall qualitative sequence and five comparable local directions.

    The lower panels are independent local orbital frames. Their angles
    are illustrative, not samples from the integrated numerical example.
    No direction u or angle beta is assigned during the coasting phase.
    """
    fig=plt.figure(figsize=(9.8,6.7))
    grid=fig.add_gridspec(2,5,height_ratios=[.92,1.05],left=.025,right=.985,
                         bottom=.065,top=.985,hspace=.22,wspace=.25)
    ax=fig.add_subplot(grid[0,:])
    x=np.linspace(-.4,8.8,501);ground=-.02-.015*x*x
    ax.fill_between(x,-1.3,ground,color=C["earth"])
    ax.plot(x,ground,color=C["grey"],lw=1)
    bezier(ax,[(0,0),(0,.20),(0,.39),(0,.60)],"blue")
    bezier(ax,[(0,.60),(0,.90),(.21,1.12),(.48,1.33)],"teal")
    bezier(ax,[(.48,1.33),(1.3,1.98),(2.4,2.37),(3.65,2.55)],"blue")
    bezier(ax,[(3.65,2.55),(4.5,2.68),(5.6,2.95),(6.65,2.95)],"grey",lw=2.3,ls="--")
    bezier(ax,[(6.65,2.95),(7.13,2.95),(7.55,2.97),(8.0,2.96)],"orange")
    ax.plot([8,8.8],[2.96,2.94],color=C["blue"],lw=1.2)
    for p in [(0,0),(.48,1.33),(3.65,2.55),(6.65,2.95),(8,2.96)]:
        ax.plot(*p,"o",ms=4,color=C["ink"])
    labels=[((0,.28),(-.28,1.20),"1"),
            ((.22,1.08),(1.45,.65),"2"),
            ((2.0,2.16),(2.8,1.20),"3"),
            ((5.2,2.84),(5.25,1.90),"4"),
            ((7.3,2.95),(7.55,1.90),"5")]
    for p,t,label in labels:
        ax.annotate(label,xy=p,xytext=t,ha="center",va="center",fontsize=14,
                    bbox={"boxstyle":"circle,pad=.20","fc":"white","ec":C["grey"],"lw":.8},
                    arrowprops={"arrowstyle":"-","color":C["grey"],"lw":.8})
    ax.text(8.5,3.18,"Орбита",ha="center",color=C["blue"],fontsize=14)
    off(ax,(-.9,9.0),(-.95,3.5))
    ax.set_aspect("auto")

    stages=[("1. Вертикальный\nподъём",90.,90.,r"$\beta=0$"),
            ("2. Начальный\nразворот",70.,48.,r"$\beta<0$"),
            ("3. Гравитационный\nразворот",34.,34.,r"$\beta=0$"),
            ("4. Пассивный\nучасток",23.,None,"$F=0$\n$\\beta$ не задан"),
            ("5. Довыведение",18.,0.,r"$\beta<0\ \to\ 0$")]
    for j,(title,gamma,theta,foot) in enumerate(stages):
        panel=fig.add_subplot(grid[1,j])
        panel.set_title(title,fontsize=14,pad=11,linespacing=1.25)
        arrow(panel,(0,0),(1.43,0),"grey",lw=1.0,scale=10)
        arrow(panel,(0,0),(0,1.43),"grey",lw=1.0,scale=10)
        panel.text(1.48,-.04,r"$\mathbf{e}_t$",ha="center",va="top",fontsize=16)
        panel.text(-.07,1.50,r"$\mathbf{e}_r$",ha="center",va="bottom",fontsize=16)
        ga=np.deg2rad(gamma);velocity=np.array([np.cos(ga),np.sin(ga)])
        if theta is not None and abs(theta-gamma)<1e-8:
            # Both vectors have the same origin and direction. The longer
            # thrust arrow remains visible beyond the orange V arrow.
            arrow(panel,(0,0),1.19*velocity,"blue",lw=2.4,scale=12)
            arrow(panel,(0,0),.88*velocity,"orange",lw=2.4,scale=11)
            if gamma==90:
                panel.text(-.12,.77,r"$\mathbf{V}$",ha="right",va="center",color=C["orange"],fontsize=17)
                panel.text(.13,1.16,r"$\mathbf{u}$",ha="left",va="center",color=C["blue"],fontsize=17)
            else:
                panel.text(*(1.24*velocity+np.array([.02,.10])),r"$\mathbf{u}$",ha="left",va="bottom",color=C["blue"],fontsize=17)
                panel.text(*(.88*velocity+np.array([-.03,.12])),r"$\mathbf{V}$",ha="right",va="bottom",color=C["orange"],fontsize=17)
        else:
            arrow(panel,(0,0),1.22*velocity,"orange",lw=2.4,scale=12)
            vlab=1.22*velocity+np.array([.03,.12])
            panel.text(*vlab,r"$\mathbf{V}$",ha="left",va="bottom",color=C["orange"],fontsize=17)
            if theta is not None:
                th=np.deg2rad(theta);direction=np.array([np.cos(th),np.sin(th)])
                arrow(panel,(0,0),1.13*direction,"blue",lw=2.1,scale=12)
                if theta==0:
                    panel.text(.78,-.18,r"$\mathbf{u}$",ha="center",va="top",color=C["blue"],fontsize=17)
                else:
                    panel.text(*(1.13*direction+np.array([.10,-.04])),r"$\mathbf{u}$",ha="left",va="center",color=C["blue"],fontsize=17)
                arc(panel,.63,min(gamma,theta),max(gamma,theta),"purple",lw=1.25)
                if theta==0:
                    panel.text(.88,.10,r"$\beta$",ha="left",va="center",color=C["purple"],fontsize=16)
                else:
                    panel.text(.32,.81,r"$\beta$",ha="center",va="center",color=C["purple"],fontsize=16)
        panel.plot(0,0,"o",ms=3,color=C["ink"])
        off(panel,(-.31,1.71),(-.38,1.78))
        panel.text(.70,-.56,foot,ha="center",va="top",fontsize=14 if theta is None else 15,linespacing=1.25)
    save(fig,"sequence")

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
    """Global axis and local basis, with the original planar horizon view."""
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.2,4.65))
    fig.subplots_adjust(left=.015,right=.99,bottom=.14,top=.90,wspace=.08)
    az,el=np.deg2rad([-35,20])
    view=np.array([[-np.sin(az),np.cos(az),0],
                   [-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)]])
    pr=lambda a:np.asarray(a)@view.T
    phi=np.deg2rad(35)
    er=np.array([np.cos(phi),0,np.sin(phi)])
    en=np.array([-np.sin(phi),0,np.cos(phi)])
    ee=np.array([0.,1.,0.]);k=np.array([0.,0.,1.])
    def line(axis,points,**kw):
        p=pr(points);axis.plot(p[:,0],p[:,1],**kw)
    def vector(axis,p,v,col,lw=1.8):
        arrow(axis,pr(p),pr(v),col,lw=lw,scale=12)
    def label(axis,p,s,col="ink",offset=(0,0),ha="left",size=15):
        q=pr(p)+np.asarray(offset)
        axis.text(*q,s,color=C[col],fontsize=size,ha=ha,va="center",
                  bbox=dict(facecolor="white",edgecolor="none",pad=.25,alpha=.8),zorder=20)
    def axis_z(axis,o,lo,hi):
        vector(axis,o+lo*k,(hi-lo)*k,"grey",lw=1)
        label(axis,o+hi*k,"$Z$",offset=(.05,.03),size=16)
    ax.set_title("Земная ось и местный базис",pad=10,fontsize=15)
    ax.add_patch(Circle((0,0),1,fc=C["earth"],ec=C["grey"],lw=.8))
    t=np.linspace(0,2*np.pi,361)
    line(ax,np.c_[np.cos(t),np.sin(t),np.zeros_like(t)],color=C["grey"],lw=.8)
    line(ax,np.c_[np.cos(phi)*np.cos(t),np.cos(phi)*np.sin(t),
                  np.full_like(t,np.sin(phi))],color=C["light"],lw=.9)
    line(ax,np.c_[np.cos(t),np.zeros_like(t),np.sin(t)],color=C["light"],lw=.8)
    origin=np.zeros(3);P=er
    axis_z(ax,origin,-1.17,1.57)
    line(ax,[origin,P],color=C["ink"],lw=1.1)
    line(ax,[origin,[1.12,0,0]],color=C["grey"],lw=.8)
    label(ax,[1.14,0,0],"экватор","grey",offset=(.02,-.06),size=12)
    ax.plot(*pr(P),"o",color=C["ink"],ms=4,zorder=8)
    for v,length,col,name,offset in [
        (er,.64,"blue",r"$\mathbf{e}_r$",(.04,.03)),
        (en,.72,"teal",r"$\mathbf{e}_N$",(-.07,.04)),
        (ee,.72,"orange",r"$\mathbf{e}_E$",(.04,-.04)),
        (k,.91,"ink",r"$\mathbf{k}$",(.06,.04))]:
        vector(ax,P,length*v,col);label(ax,P+length*v,name,col,offset=offset)
    label(ax,P,"$P$",offset=(-.06,-.10),ha="right",size=14)
    label(ax,origin,"$O$",offset=(-.07,-.07),ha="right",size=14)
    arcpoints=np.array([[.34*np.cos(q),0,.34*np.sin(q)] for q in np.linspace(0,phi,45)])
    line(ax,arcpoints,color=C["purple"],lw=1.2)
    label(ax,[.44*np.cos(phi/2),0,.44*np.sin(phi/2)],r"$\phi$","purple",offset=(.025,.02))
    off(ax,(-1.18,1.42),(-1.15,1.60))
    bx.set_title("Местный горизонт",pad=8)
    arrow(bx,(-1.6,0),(3.95,0),"grey",lw=1,scale=11)
    arrow(bx,(0,-.30),(0,2.8),"grey",lw=1,scale=11)
    bx.text(2.27,-.14,"Восток",ha="right",fontsize=12)
    bx.text(.10,2.38,"Север",fontsize=12)
    arrow(bx,(0,0),(0,1.95),"teal",lw=2.2,scale=13)
    bx.text(.12,2.00,r"$\mathbf{e}_N$",color=C["teal"],fontsize=15)
    arrow(bx,(0,0),(1.95,0),"orange",lw=2.2,scale=13)
    bx.text(2.00,.12,r"$\mathbf{e}_E$",color=C["orange"],fontsize=15)
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
    fig.text(.5,.035,r"$\mathbf{k}=\sin\phi\,\mathbf{e}_r+\cos\phi\,\mathbf{e}_N"
             r"\qquad\quad \cos i=\mathbf{n}\cdot\mathbf{k}=\cos\phi\,\sin A$",
             ha="center",fontsize=15)
    save(fig,"launch-plane")

def rotation():
    """Initial eastward pad velocity and the air/inertial velocity triangle."""
    fig,(gx,ax)=plt.subplots(1,2,figsize=(10.0,4.55))
    fig.subplots_adjust(left=.015,right=.99,bottom=.15,top=.90,wspace=.13)
    az,el=np.deg2rad([-35,20])
    view=np.array([[-np.sin(az),np.cos(az),0],
                   [-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)]])
    pr=lambda a:np.asarray(a)@view.T
    phi=np.deg2rad(35);k=np.array([0.,0.,1.]);ee=np.array([0.,1.,0.])
    P=np.array([np.cos(phi),0,np.sin(phi)]);Q=np.array([0.,0.,np.sin(phi)])
    gx.set_title("Начальная скорость на площадке",pad=10,fontsize=15)
    gx.add_patch(Circle((0,0),1,fc=C["earth"],ec=C["grey"],lw=.8))
    t=np.linspace(0,2*np.pi,361)
    eq=pr(np.c_[np.cos(t),np.sin(t),np.zeros_like(t)])
    lat=pr(np.c_[np.cos(phi)*np.cos(t),np.cos(phi)*np.sin(t),np.full_like(t,np.sin(phi))])
    gx.plot(eq[:,0],eq[:,1],color=C["grey"],lw=.9)
    gx.plot(lat[:,0],lat[:,1],color=C["light"],lw=1.2)
    arrow(gx,pr(-1.15*k),pr(2.72*k),"grey",lw=1,scale=11)
    gx.text(*pr(1.60*k),"$Z$",fontsize=16,ha="left",va="center")
    arrow(gx,pr(.35*k),pr(.85*k),"purple",lw=2,scale=13)
    gx.text(-.12,1.06,r"$\mathbf{\omega}_E$",fontsize=16,color=C["purple"],ha="right")
    arrow(gx,(0,0),pr(P),"ink",lw=1.35,scale=10)
    qp=pr(np.array([Q,P]));gx.plot(qp[:,0],qp[:,1],color=C["grey"],lw=1.1,ls="--")
    gx.plot(*pr(P),"o",color=C["ink"],ms=5,zorder=10)
    arrow(gx,pr(P),pr(.79*ee),"blue",lw=2.5,scale=15)
    label=pr(P+.82*ee)
    gx.text(label[0]-.04,label[1]+.13,r"$\mathbf{V}_0$",color=C["blue"],fontsize=17,ha="center")
    gx.text(label[0]+.06,label[1]-.18,"восток",color=C["blue"],fontsize=13,ha="center")
    mid=pr((Q+P)/2)
    gx.text(mid[0]+.13,mid[1]+.11,r"$R_E\cos\phi$",color=C["grey"],fontsize=14,ha="center")
    midpoint=pr(P/2)
    gx.text(midpoint[0]-.02,midpoint[1]+.09,r"$\mathbf{R}_0$",fontsize=15,ha="right")
    ray=pr(np.array([[0.,0.,0.],[1.10,0.,0.]]))
    gx.plot(ray[:,0],ray[:,1],color=C["grey"],lw=.8)
    tt=np.linspace(0,phi,60);pp=pr(np.c_[.35*np.cos(tt),np.zeros_like(tt),.35*np.sin(tt)])
    gx.plot(pp[:,0],pp[:,1],color=C["orange"],lw=1.3)
    pp=pr([.45*np.cos(phi/2),0,.45*np.sin(phi/2)])
    gx.text(pp[0]+.025,pp[1]-.03,r"$\phi$",color=C["orange"],fontsize=15)
    gx.text(-.04,-.09,"$O$",fontsize=14,ha="right")
    off(gx,(-1.15,1.62),(-1.16,1.62))
    ax.set_title("Инерциальная и воздушная скорости",pad=10,fontsize=15)
    vair=np.array([2.7,3.7]);vatm=np.array([1.20,0]);v=vair+vatm
    arrow(ax,(-.20,0),(5.4,0),"grey",lw=1,scale=11)
    arrow(ax,(0,-.20),(0,4.85),"grey",lw=1,scale=11)
    ax.text(5.05,-.18,"Восток",ha="right",fontsize=14)
    ax.text(-.08,4.65,"Север",ha="right",fontsize=14)
    arrow(ax,(0,0),vair,"teal",lw=2.6);arrow(ax,vair,vatm,"purple",lw=2.2)
    arrow(ax,(0,0),v,"blue",lw=2.6)
    ax.text(1.35,2.32,r"$\mathbf{V}_{\rm air}$",ha="right",color=C["teal"],fontsize=16)
    ax.text(3.32,3.98,r"$\mathbf{V}_{\rm atm}=\mathbf{\omega}_E\times\mathbf{R}$",
            ha="center",color=C["purple"],fontsize=15)
    ax.text(3.45,2.55,r"$\mathbf{V}$",color=C["blue"],fontsize=17)
    airang=np.rad2deg(np.arctan2(vair[1],vair[0]));ang=np.rad2deg(np.arctan2(v[1],v[0]))
    arc(ax,1.05,airang,90,"teal");arc(ax,1.60,ang,90,"blue")
    ax.text(.44,1.04,r"$A_{\rm air}$",color=C["teal"],fontsize=14)
    ax.text(1.08,1.42,"$A$",color=C["blue"],fontsize=16)
    ax.plot(0,0,"o",color=C["ink"],ms=4)
    ax.text(-.10,-.17,"$P$",ha="right",fontsize=14)
    ax.text(3.1,.27,"Без ветра;\nмасштабы условные",ha="center",fontsize=13,color=C["grey"])
    off(ax,(-.75,5.3),(-.40,4.85))
    fig.text(.25,.035,r"$V_{E,0}=|\mathbf{V}_0|=\omega_E R_E\cos\phi$",ha="center",fontsize=16)
    fig.text(.755,.035,r"$\mathbf{V}=\mathbf{V}_{\rm air}+\mathbf{\omega}_E\times\mathbf{R}$",
             ha="center",fontsize=16)
    save(fig,"rotation")

def launch_window():
    """The target plane in 3D and its trace in a top view along +Z."""
    from matplotlib.patches import Polygon
    fig,(gx,ax)=plt.subplots(1,2,figsize=(10.1,4.8))
    fig.subplots_adjust(left=.01,right=.99,bottom=.14,top=.91,wspace=.14)
    phi=np.deg2rad(30);incl=np.deg2rad(60);rho=np.cos(phi);z=np.sin(phi)
    yy=z/np.tan(incl);xx=np.sqrt(rho*rho-yy*yy)
    az,el=np.deg2rad([-45,25])
    view=np.array([[-np.sin(az),np.cos(az),0],
                   [-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)]])
    pr=lambda a:np.asarray(a)@view.T
    k=np.array([0.,0.,1.]);ex=np.array([1.,0.,0.]);ey=np.array([0.,1.,0.])
    vplane=np.array([0.,np.cos(incl),np.sin(incl)])
    gx.set_title("Орбитальная плоскость и параллель",pad=10,fontsize=15)
    gx.add_patch(Circle((0,0),1,fc=C["earth"],ec=C["grey"],lw=.7,alpha=.5))
    orbit=np.array([-1.12*ex-1.12*vplane,1.12*ex-1.12*vplane,
                     1.12*ex+1.12*vplane,-1.12*ex+1.12*vplane])
    parallel=np.array([[-1.06,-1.00,z],[1.06,-1.00,z],[1.06,1.00,z],[-1.06,1.00,z]])
    gx.add_patch(Polygon(pr(orbit),fc=C["blue"],ec=C["blue"],alpha=.18,lw=1.1))
    gx.add_patch(Polygon(pr(parallel),fc=C["orange"],ec=C["orange"],alpha=.10,lw=.9))
    tt=np.linspace(0,2*np.pi,361)
    pp=pr(np.c_[rho*np.cos(tt),rho*np.sin(tt),np.full_like(tt,z)])
    gx.plot(pp[:,0],pp[:,1],color=C["orange"],lw=2)
    eq=pr(np.c_[np.cos(tt),np.sin(tt),np.zeros_like(tt)])
    gx.plot(eq[:,0],eq[:,1],color=C["grey"],lw=.7)
    for vec,name in [(ex,"X"),(ey,"Y"),(k,"Z")]:
        arrow(gx,pr(-.15*vec),pr(1.66*vec),"grey",lw=.85,scale=10)
        p=pr(1.55*vec);gx.text(p[0]+.03,p[1]+.015,"$"+name+"$",fontsize=15)
    tr=pr(np.array([[-1.15,yy,z],[1.15,yy,z]]))
    gx.plot(tr[:,0],tr[:,1],color=C["blue"],lw=2.4)
    for p,name in [(np.array([xx,yy,z]),"A"),(np.array([-xx,yy,z]),"B")]:
        q=pr(p);gx.plot(*q,"o",color=C["blue"],ms=5,zorder=9)
        gx.text(q[0]+.04,q[1]-.08,"$"+name+"$",fontsize=14,color=C["blue"])
    lam=np.deg2rad(60);P=np.array([rho*np.cos(lam),rho*np.sin(lam),z]);P0=P-z*k
    p=pr(P);gx.plot(*p,"o",color=C["orange"],ms=5,zorder=10)
    gx.text(p[0]+.06,p[1]+.06,r"$P(t)$",fontsize=15,color=C["orange"])
    ray=pr(np.array([np.zeros(3),P0,P]))
    gx.plot(ray[:2,0],ray[:2,1],color=C["orange"],lw=1.0)
    gx.plot(ray[1:,0],ray[1:,1],color=C["orange"],ls="--",lw=.8)
    qa=np.linspace(0,lam,70);ap=pr(np.c_[.40*np.cos(qa),.40*np.sin(qa),np.zeros_like(qa)])
    gx.plot(ap[:,0],ap[:,1],color=C["orange"],lw=1.3)
    pp=pr([.53*np.cos(lam/2),.53*np.sin(lam/2),0])
    gx.text(pp[0]+.03,pp[1]-.06,r"$\Lambda(t)$",fontsize=14,color=C["orange"])
    gx.annotate("Целевая\nорбитальная плоскость",xy=pr(.8*vplane+.5*ex),
                xytext=(1.02,1.10),ha="center",fontsize=13,color=C["blue"],
                arrowprops=dict(arrowstyle="-",color=C["blue"],lw=.8))
    gx.annotate("Плоскость\nсуточной параллели",xy=pr([-.85,-.60,z]),
                xytext=(-1.20,-.94),ha="center",fontsize=13,color=C["orange"],
                arrowprops=dict(arrowstyle="-",color=C["orange"],lw=.8))
    off(gx,(-1.62,1.66),(-1.48,1.62))
    ax.set_title("Вид со стороны $+Z$",pad=10,fontsize=15)
    ax.add_patch(Circle((0,0),rho,fc="white",ec=C["orange"],lw=2.3))
    ax.plot([-1.18,1.12],[yy,yy],color=C["blue"],lw=2)
    arrow(ax,(-1.16,0),(2.28,0),"grey",lw=.8,scale=10)
    arrow(ax,(0,-1.03),(0,2.25),"grey",lw=.8,scale=10)
    ax.text(1.13,-.10,"$X$",fontsize=14);ax.text(.06,1.17,"$Y$",fontsize=14)
    ax.add_patch(Circle((0,0),.045,fc="white",ec=C["ink"],lw=1.1,zorder=10))
    ax.plot(0,0,".",color=C["ink"],ms=4,zorder=11)
    ax.text(-.12,-.12,"$Z$",ha="right",fontsize=15)
    ax.text(.10,-.14,"к читателю",fontsize=11,color=C["grey"])
    for P,lab in [(np.array([xx,yy]),"A"),(np.array([-xx,yy]),"B")]:
        ax.plot(*P,"o",ms=6,color=C["blue"])
        ax.text(P[0]+(.05 if lab=="A" else -.06),P[1]+.08,"$"+lab+"$",
                fontsize=14,ha="left" if lab=="A" else "right",color=C["blue"])
    P=rho*np.array([np.cos(lam),np.sin(lam)])
    ax.plot([0,P[0]],[0,P[1]],color=C["orange"],lw=1.1)
    ax.plot(*P,"o",ms=5,color=C["orange"])
    ax.text(P[0]+.06,P[1]+.05,r"$P(t)$",color=C["orange"],fontsize=15)
    qa=np.linspace(0,lam,70);ap=.35*np.c_[np.cos(qa),np.sin(qa)]
    ax.plot(ap[:,0],ap[:,1],color=C["orange"],lw=1.4)
    arrow(ax,ap[-5],ap[-1]-ap[-5],"orange",lw=1.1,scale=9)
    ax.text(.44,.12,r"$\Lambda(t)$",fontsize=15,color=C["orange"])
    t0,t1=np.deg2rad([112,138]);p0=rho*np.array([np.cos(t0),np.sin(t0)]);p1=rho*np.array([np.cos(t1),np.sin(t1)])
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=12,
                                connectionstyle="arc3,rad=.1",lw=1.6,color=C["orange"]))
    ax.text(-.54,.94,r"$\omega_E$",color=C["orange"],fontsize=15)
    ax.annotate("След целевой плоскости\nна плоскости параллели",xy=(-.35,yy),
                xytext=(-.02,-1.05),ha="center",fontsize=13,color=C["blue"],
                arrowprops=dict(arrowstyle="-",color=C["blue"],lw=.8))
    off(ax,(-1.23,1.30),(-1.30,1.25))
    fig.text(.5,.035,r"$Z=R_E\sin\phi=\mathrm{const},\qquad"
             r" \Lambda(t)=\Lambda_0+\omega_Et$",ha="center",fontsize=15)
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
