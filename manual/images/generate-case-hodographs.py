"""Generate the case figures from the same impulses that solve the boundary equations.
Run from any directory with Python 3 and NumPy. Generated .tex assets need only pdflatex.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
V0 = np.sqrt(398600.4418 / 7000) * 1000
PI = np.pi

def primer(lam, phi):
    a,b,c,d,e = lam
    return np.array([c*np.sin(phi)-b*np.cos(phi),
                     2*a+2*c*np.cos(phi)+2*b*np.sin(phi),
                     d*np.sin(phi)+e*np.cos(phi)]).T

def boundary(phi, dv):
    r,t,z=dv.T
    return np.array([2*t.sum(), (r*np.sin(phi)+2*t*np.cos(phi)).sum(),
                     (-r*np.cos(phi)+2*t*np.sin(phi)).sum(),
                     (z*np.sin(phi)).sum(), (z*np.cos(phi)).sum()])

cases = {}
def add(key, data, lam, phi, dv, spatial=False, note=''):
    phi=np.asarray(phi)%(2*PI); dv=np.asarray(dv)
    order=np.argsort(phi); phi=phi[order]; dv=dv[order]
    mags=np.linalg.norm(dv,axis=1)
    target=np.array([*data[:3],0,data[3]])
    grid=np.linspace(0,2*PI,3601)
    norm=np.linalg.norm(primer(lam,grid),axis=1)
    assert np.max(abs(boundary(phi,dv)-target))<1e-12, key
    assert norm.max()<=1+1e-12, key
    assert np.max(abs(primer(lam,phi)-dv/mags[:,None]))<1e-12, key
    a,b,c,d,e=lam
    lower=a*data[0]+c*data[1]+b*data[2]+e*data[3]
    assert abs(mags.sum()-lower)<1e-12, key
    cases[key]=dict(data=list(data),lam=list(lam),phi=phi.tolist(),dv=dv.tolist(),
                    J=float(mags.sum()),dimensional_cost=float(V0*mags.sum()),
                    spatial=spatial,note=note,
                    max_boundary_residual=float(np.max(abs(boundary(phi,dv)-target))))

# Tangent orbits: one contact, center at s_t=1/4, semiaxes 3/4 and 3/8.
data=(.002,.0016,.0012,0)
phase=np.arctan2(data[2],data[1])
add('tangent',data,(.125,.375*np.sin(phase),.375*np.cos(phase),0,0),
    [phase],[[0,data[0]/2,0]])
# Disjoint orbits: demonstrate that the first phase need not be an apsidal phase.
data=(.004,.001,.001,0); a,x,y,i=data; phase=PI/6
v1=(x*x+y*y-a*a)/(4*(x*np.cos(phase)+y*np.sin(phase)-a)); v2=a/2-v1
phase2=np.arctan2((y-2*v1*np.sin(phase))/(2*v2),(x-2*v1*np.cos(phase))/(2*v2))
add('disjoint',data,(.5,0,0,0,0),[phase,phase2],[[0,v1,0],[0,v2,0]],
    note=r'Для примера выбран угол первого импульса $\varphi_1=30^\circ$.')
# Intersecting coplanar orbits.
data=(.001,.002,.001,0); a,x,y,i=data; ecc=np.hypot(x,y); phase=np.arctan2(y,x)
add('crossing',data,(0,.5*np.sin(phase),.5*np.cos(phase),0,0),
    [phase,phase+PI],[[0,(a+ecc)/4,0],[0,(a-ecc)/4,0]])
# NC.
data=(.001,.002,.0005,.002); a,x,y,i=data; J=np.sqrt(x*x/4+y*y+i*i)
direction=np.array([-y,x/2,i])/J
lam=np.array([0,-direction[0],direction[1]/2,0,direction[2]])
lam[3]=-3*lam[2]*lam[1]/lam[4]
add('nc',data,lam,[0,PI],[J/2*(1+a/x)*direction,-J/2*(1-a/x)*direction],True)
# NDC.
data=(.004,.001,.001,.002); a,x,y,i=data
k=a*a+i*i-x*x-y*y; eta=(k+np.sqrt(k*k+4*i*i*y*y))/(2*i*abs(y))
sigma=np.sign(y); L=np.hypot(x,abs(y)+i*eta)
alpha=np.arctan2(sigma*(abs(y)+i*eta),x); beta=np.arccos(a/L)
J=np.sqrt(a*a/4+(L*L-a*a)/(1+eta*eta))
dr=L*np.sin(beta)/((1+eta*eta)*J); dt=a/(2*J); dz=-sigma*eta*dr
delta=-sigma*x*np.sqrt(L*L-a*a)/(a*(abs(y)+i*eta))
amp=L/((1+eta*eta)*J)
lam=(a*(eta*eta-3)/(4*(1+eta*eta)*J),amp*np.sin(alpha),amp*np.cos(alpha),
     -sigma*eta*amp*np.cos(alpha),sigma*eta*amp*np.sin(alpha))
add('ndc',data,lam,[alpha+beta,alpha-beta],
    [J/2*(1+delta)*np.array([dr,dt,dz]),J/2*(1-delta)*np.array([-dr,dt,-dz])],True)
# Segment limit.
data=(.004,.001,0,.002); a,x,y,i=data; J=np.sqrt(a*a/4+i*i)
add('line',data,(a/(4*J),0,0,0,i/J),[0,PI],
    [(1+x/a)/2*np.array([0,a/2,i]),(1-x/a)/2*np.array([0,a/2,-i])],True)
# Singular case: the three-impulse construction used in the text.
data=(.001,0,.001,.0002); a,x,y,i=data; sigma=np.sign(y)
J=.5*np.hypot(x,abs(y)+np.sqrt(3)*i); alpha=np.arctan2(sigma*(abs(y)+np.sqrt(3)*i),x)
ss=i*(abs(y)+np.sqrt(3)*i)/(np.sqrt(3)*J); sc=-sigma*i*x/(np.sqrt(3)*J)
s=np.sqrt(ss/J); c=np.sqrt(1-s*s); beta=np.arctan2(s,c)
psi=np.array([PI-beta,-beta,beta]); mags=np.array([J/2-a/(4*c),a/(4*c)-sc/(2*s*c),J/2+sc/(2*s*c)])
lam=(0,.5*np.sin(alpha),.5*np.cos(alpha),-sigma*np.sqrt(3)/2*np.cos(alpha),sigma*np.sqrt(3)/2*np.sin(alpha))
add('sc',data,lam,alpha+psi,primer(lam,alpha+psi)*mags[:,None],True)

def num(v, digits=6):
    if abs(v)<5*10**(-digits): v=0
    return f'{v:.{digits}f}'.rstrip('0').rstrip('.').replace('.', '{,}')

def path(points):
    return ' '.join(f'({x:.5f},{y:.5f})' for x,y in points)

az,el=np.deg2rad([50,25])
proj=np.array([[-np.sin(az),np.cos(az),0],[-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)]])
view=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])

def draw_face_on_hodograph(out, key, center, radius, numbered):
    """Preserve distances in the hodograph plane; draw all three axis rays."""
    case=cases[key]; lam=case['lam']; center=np.asarray(center)
    horizontal=np.array([0.,1.,0.])
    if key=='line':
        vertical=np.array([0.,0.,1.]); normal=np.array([1.,0.,0.])
    else:
        p=np.array([-lam[1],2*lam[2],lam[4]])
        q=np.array([lam[2],2*lam[1],lam[3]])
        normal=np.cross(p,q); normal/=np.linalg.norm(normal)
        vertical=np.cross(normal,horizontal)
        if vertical[2]<0: normal=-normal; vertical=-vertical
    plane_projection=np.stack([horizontal,vertical])
    # This parallel oblique projection acts as an isometry on the chosen plane.
    # Its normal is drawn down and left so the r and z axis rays stay distinct.
    axis_projection=plane_projection+np.outer(np.array([-.65,-.45]),normal)
    def project(v): return np.asarray(v)@plane_projection.T*radius+center
    def draw(points,style):
        out.append(r'\draw['+style+r'] plot coordinates {'+path(points)+'};')
    curve=primer(lam,np.linspace(0,2*PI,361))
    contacts=primer(lam,np.array(case['phi']))
    assert np.max(abs(curve@normal))<1e-12, key
    assert np.max(abs(np.linalg.norm(curve@plane_projection.T,axis=1)-
                      np.linalg.norm(curve,axis=1)))<1e-12, key
    assert np.max(abs(np.linalg.norm(contacts@plane_projection.T,axis=1)-1))<1e-12, key
    out.append(f'\\filldraw[fill=figplane!45!white,draw=figgray!65,line width=.65pt] '
               f'({center[0]:.5f},{center[1]:.5f}) circle[radius={radius}cm];')
    for ax in range(3):
        direction=axis_projection[:,ax].copy()
        direction/=np.linalg.norm(direction)
        endpoint=center+radius*1.32*direction
        draw([center,endpoint],'->,figink!75,thin')
        anchor=['below left','right','above'][ax]
        out.append(f'\\node[{anchor},inner sep=2pt] '
                   f'at ({endpoint[0]:.5f},{endpoint[1]:.5f}) '
                   f'{{$s_{["r","t","z"][ax]}$}};')
    if key=='ndc':
        draw(project(contacts),'figgray!60,thin,densely dashed')
    draw(project(curve),'accent,line width=1.2pt')
    x,y=project([0,2*lam[0],0])
    out.append(f'\\node[accent,inner sep=0pt] at ({x:.5f},{y:.5f}) {{$\\times$}};')
    for j,point in enumerate(project(contacts),1):
        draw([center,point],'->,accent,line width=.7pt')
        out.append(f'\\filldraw[fill=fignormal,draw=white,line width=.4pt] '
                   f'({point[0]:.5f},{point[1]:.5f}) circle[radius=2.3pt];')
        if numbered:
            anchor='above right' if point[1]>center[1] else 'below right'
            out.append(f'\\node[{anchor},inner sep=3pt] '
                       f'at ({point[0]:.5f},{point[1]:.5f}) {{{j}}};')

for key,case in cases.items():
    data=case['data']; lam=case['lam']; phi=np.array(case['phi']); dv=np.array(case['dv']); mags=np.linalg.norm(dv,axis=1)
    out=[]
    placement='!htbp' if key in ('tangent','disjoint','crossing','line') else 'H'
    out.append(r'\begin{figure}['+placement+r']\centering')
    out.append(r'\begin{tikzpicture}[>=Latex,line cap=round,line join=round]')
    out.append(r'\node at (2.65,6.85) {Годограф};')
    out.append(r'\node at (11.6,6.85) {Модули импульсов};')
    out.append(r'\node at (11.6,4.1) {Модуль базис-вектора};')
    center=np.array([2.65,3.6]); scale=1.7
    def project(v): return np.asarray(v)@proj.T*scale+center
    def draw(points,style): out.append(r'\draw['+style+r'] plot coordinates {'+path(points)+'};')
    curve=primer(lam,np.linspace(0,2*PI,361))
    if key in ('ndc','line'):
        draw_face_on_hodograph(out,key,center,scale,numbered=True)
    elif case['spatial']:
        out.append(r'\fill[figplane!45!white] (2.65,3.6) circle[radius=1.7cm];')
        p=np.array([-lam[1],2*lam[2],lam[4]]); q=np.array([lam[2],2*lam[1],lam[3]])
        if np.linalg.norm(np.cross(p,q))>1e-9:
            u=p/np.linalg.norm(p); v=q-np.dot(q,u)*u; v/=np.linalg.norm(v)
        else: u=np.array([0,1,0]); v=np.array([0,0,1])
        corners=project(1.06*np.array([u+v,u-v,-u-v,-u+v]))
        out.append(r'\filldraw[fill=accent,fill opacity=.09,draw=accent!30] '+ '--'.join(f'({x:.5f},{y:.5f})' for x,y in corners)+r'--cycle;')
        tt=np.linspace(0,2*PI,181)
        for aa,bb in [(0,1),(0,2),(1,2)]:
            vec=np.zeros((len(tt),3)); vec[:,aa]=np.cos(tt); vec[:,bb]=np.sin(tt)
            draw(project(vec),'figgray!40,thin,densely dashed')
            phase=np.arctan2(view[bb],view[aa]); front=np.linspace(phase-PI/2,phase+PI/2,91)
            vec=np.zeros((len(front),3)); vec[:,aa]=np.cos(front); vec[:,bb]=np.sin(front)
            draw(project(vec),'figgray!55,thin')
        for ax in range(3):
            v=np.eye(3)[ax]
            draw(project([-1.15*v,np.zeros(3)]),'figgray!50,thin,densely dashed')
            a,b=project([np.zeros(3),1.4*v]); draw([a,b],'->,figink!75,thin')
            anchor=['above right','below right','above'][ax]
            out.append(f'\\node[{anchor},inner sep=1pt] at ({b[0]:.5f},{b[1]:.5f}) {{$s_{["r","t","z"][ax]}$}};')
        draw(project(curve),'accent,line width=1.15pt')
        if key=='ndc':
            x,y=project([0,2*lam[0],0]); out.append(f'\\node[accent] at ({x:.5f},{y:.5f}) {{$\\times$}};')
        for j,v in enumerate(primer(lam,phi)):
            a,b=project([np.zeros(3),v]); draw([a,b],'->,accent,line width=.55pt')
            out.append(f'\\filldraw[fill=fignormal,draw=white,line width=.4pt] ({b[0]:.5f},{b[1]:.5f}) circle[radius=2pt];')
            offset=np.array([.13 if b[0]>center[0] else -.13,.18 if b[1]>center[1] else -.18])
            xy=b+offset; out.append(f'\\node[fill=white,fill opacity=.85,text opacity=1,inner sep=.6pt] at ({xy[0]:.5f},{xy[1]:.5f}) {{{j+1}}};')
        out.append(r'\draw[figgray!65,thin] (2.65,3.6) circle[radius=1.7cm];')
    else:
        out.append(r'\draw[figgray!60,dashed] (2.65,3.6) circle[radius=1.7cm];')
        out.append(r'\draw[->,figgray] (.6,3.6)--(4.85,3.6) node[right] {$s_t$};')
        out.append(r'\draw[->,figgray] (2.65,1.55)--(2.65,5.8) node[above] {$s_r$};')
        draw(center+scale*curve[:,[1,0]],'accent,line width=1.15pt')
        pts=primer(lam,phi)
        for j,v in enumerate(pts):
            xy=center+scale*v[[1,0]]
            out.append(f'\\filldraw[fill=fignormal,draw=white,line width=.4pt] ({xy[0]:.5f},{xy[1]:.5f}) circle[radius=2pt];')
            if key!='disjoint':
                out.append(f'\\node[above,inner sep=3pt] at ({xy[0]:.5f},{xy[1]:.5f}) {{{j+1}}};')
        if key=='disjoint': out.append(r'\node[above,align=center] at (4.35,3.6) {1, 2};')
    out.append(r'\node[below left,inner sep=2pt] at (2.65,3.6) {$0$};')
    # Right-hand plots share their phase axis.
    x0,x1=7.1,16.5
    xx=lambda ph:x0+(x1-x0)*ph/(2*PI)
    y0,ys=.85,2.6
    for y in [0,.5,1]:
        yy=y0+ys*y
        draw([[x0,yy],[x1,yy]],'figgray!25,thin' if y!=1 else 'figink!65,dashed')
        out.append(f'\\node[left,inner sep=3pt] at ({x0},{yy}) {{${num(y,1)}$}};')
    draw([[x0,y0],[x1+.1,y0]],'->,figgray');draw([[x0,y0],[x0,y0+ys*1.15]],'->,figgray')
    out.append(r'\node[rotate=90] at (6.25,2.2) {$\|\boldsymbol s(\varphi)\|$};')
    tt=np.linspace(0,2*PI,721); nn=np.linalg.norm(primer(lam,tt),axis=1)
    draw(np.array([xx(tt),y0+ys*nn]).T,'accent,line width=1.15pt')
    maxmag=max(mags)*V0; ymax=np.ceil(maxmag*1.2)
    topbase,topscale=4.65,1.6/ymax
    draw([[x0,topbase],[x1+.1,topbase]],'->,figgray');draw([[x0,topbase],[x0,6.5]],'->,figgray')
    for val in [0,ymax/2,ymax]:
        yy=topbase+topscale*val
        draw([[x0,yy],[x1,yy]],'figgray!25,thin')
        out.append(f'\\node[left,inner sep=3pt] at ({x0},{yy:.5f}) {{${num(val,1)}$}};')
    out.append(r'\node[rotate=90] at (6.25,5.55) {$\Delta V_j$, м/с};')
    for phase,label in zip(np.linspace(0,2*PI,5),['0',r'\pi/2',r'\pi',r'3\pi/2',r'2\pi']):
        x=xx(phase)
        draw([[x,y0-.05],[x,topbase+1.6]],'figgray!16,thin')
        out.append(f'\\node[below,inner sep=3pt] at ({x:.5f},{y0}) {{${label}$}};')
    out.append(r'\node at (11.8,.15) {Угол $\varphi$, рад};')
    for j,(phase,mag) in enumerate(zip(phi,mags)):
        x=xx(phase); yy=topbase+topscale*mag*V0
        draw([[x,y0],[x,y0+ys]],'fignormal!70,densely dashed,thin')
        draw([[x,topbase],[x,yy]],'fignormal,line width=1pt')
        for y in [yy,y0+ys]:
            out.append(f'\\fill[fignormal] ({x:.5f},{y:.5f}) circle[radius=1.8pt];')
        anch='above right' if phase<.05 else 'above'
        out.append(f'\\node[{anch},inner sep=2pt] at ({x:.5f},{yy:.5f}) {{{j+1}}};')
    out.append(r'\end{tikzpicture}')
    caption=r'Пример перехода: годограф базис-вектора, модули импульсов и зависимость модуля базис-вектора от угла. '
    if key in ('ndc','line'):
        caption+=(r'Годограф показан в своей плоскости. ' if key=='ndc' else
                  r'Отрезок показан в плоскости $(s_t,s_z)$. ')
        caption+=r'Серая окружность имеет единичный радиус; оси показаны в косоугольной проекции. '
    caption+=r'Номера точек соответствуют номерам импульсов в численном результате. Пунктир отмечает уровень $\|\boldsymbol s\|=1$; золотистые отметки показывают выбранные углы приложения импульсов.'
    out.append(r'\caption{'+caption+'}')
    out.append(r'\label{fig:case-'+key+r'}')
    out.append(r'\begin{tcolorbox}[colback=accent!3!white,colframe=accent!65!black,boxrule=.5pt,arc=1mm,left=3mm,right=3mm,top=1.5mm,bottom=1.5mm,before skip=5pt,after skip=0pt,fontupper=\normalsize]')
    out.append(r'Начальная орбита круговая: $a_i=r_0=7000$ км; $V_0=7546{,}053$ м/с.')
    out.append('\\[ '+r'\Delta a='+num(data[0],7)+r',\quad\Delta e_x='+num(data[1],7)+r',\quad\Delta e_y='+num(data[2],7)+r',\quad\Delta i='+num(data[3],7)+r'\text{ рад}.\]')
    if case['note']: out.append(case['note'])
    out.append(r'Множители для годографа (округлённо):')
    out.append(r'\[ (\lambda_1,\lambda_2,\lambda_3,\lambda_4,\lambda_5)=('+r',\;'.join(num(v) for v in lam)+r').\]')
    out.append(r'\end{tcolorbox}\end{figure}')
    (HERE/f'case-{key}.tex').write_text('\n'.join(out)+'\n')
    result=[r'\begin{explanation}{Численный результат}',r'Для данных рис.~\ref{fig:case-'+key+r'} получаем:',r'\begin{center}\begin{tabular}{cc|rrr}',r'$j$ & $\varphi_j$, град & $\Delta V_{rj}$, м/с & $\Delta V_{tj}$, м/с & $\Delta V_{zj}$, м/с\\\hline']
    for j,(ph,vel) in enumerate(zip(phi,dv),1): result.append(str(j)+' & $'+num(ph*180/PI,5)+'$ & '+' & '.join('$'+num(v*V0,3)+'$' for v in vel)+r'\\')
    result.extend([r'\end{tabular}\end{center}',r'\[ J='+num(case['J'],9)+r',\qquad \Delta V_\Sigma='+num(case['dimensional_cost'],3)+r'\text{ м/с}.\]',r'\end{explanation}'])
    (HERE/f'case-{key}-result.tex').write_text('\n'.join(result)+'\n')

# Overview: reuse the same three planar hodographs without the impulse/norm plots.
overview=[r'\begin{figure}[H]\centering',
          r'\begin{tikzpicture}[>=Latex,line cap=round,line join=round]']
panels=[('tangent',r'а) Смещённый эллипс\\Касающиеся орбиты'),
        ('disjoint',r'б) Точка\\Непересекающиеся орбиты'),
        ('crossing',r'в) Центральный эллипс\\Пересекающиеся орбиты')]
for index,(key,title) in enumerate(panels):
    overview.append(f'\\begin{{scope}}[shift={{({5.5*index},0)}}]')
    overview.extend([r'\draw[figgray!60,dashed] (0,0) circle[radius=1.4cm];',
                     r'\draw[->,figgray] (-1.7,0)--(1.85,0) node[right] {$s_t$};',
                     r'\draw[->,figgray] (0,-1.65)--(0,1.75) node[above] {$s_r$};'])
    curve=primer(cases[key]['lam'],np.linspace(0,2*PI,361))
    if key!='disjoint':
        overview.append(r'\draw[accent,line width=1.15pt] plot coordinates {'+
                        path(1.4*curve[:,[1,0]])+'};')
    for point in primer(cases[key]['lam'],np.array(cases[key]['phi'])):
        x,y=1.4*point[[1,0]]
        overview.append(f'\\filldraw[fill=fignormal,draw=white,line width=.4pt] '
                        f'({x:.5f},{y:.5f}) circle[radius=2.3pt];')
    overview.extend([r'\node[below left,inner sep=2pt] at (0,0) {$0$};',
                     r'\node[align=center,text width=5.3cm,anchor=north] at (0,-1.9) {'+title+'};',
                     r'\end{scope}'])
overview.extend([r'\end{tikzpicture}',
                 r'\caption{Три типа годографа базис-вектора в компланарной задаче. '
                 r'Пунктиром показана единичная окружность, золотистым цветом --- '
                 r'точки касания; в случае (б) весь годограф состоит из одной точки. '
                 r'Для (а) и (б) показан разгонный вариант.}',
                 r'\label{fig:planar-hodographs-overview}',r'\end{figure}'])
(HERE/'planar-hodographs-overview.tex').write_text('\n'.join(overview)+'\n')

# Spatial overview: the three main types and the segment limit of NDC.
# Reuse the example multipliers, viewing direction and unit-sphere scale.
spatial_overview=[r'\begin{figure}[H]\centering',
                  r'\begin{tikzpicture}[>=Latex,line cap=round,line join=round]']
spatial_panels=[
    ('nc', r'а) Узловое решение (NC)\\Эллипс с центром в начале координат'),
    ('ndc', r'б) Невырожденное решение (NDC)\\Смещённый эллипс'),
    ('sc', r'в) Особое решение (SC)\\Окружность на сфере'),
    ('line', r'г) Предельный случай NDC\\Отрезок'),
]
for index,(key,title) in enumerate(spatial_panels):
    lam=cases[key]['lam']
    spatial_overview.append(
        f'\\begin{{scope}}[shift={{({8.3*(index%2)},{-6.5*(index//2)})}}]')
    radius=1.9
    if key in ('ndc','line'):
        draw_face_on_hodograph(spatial_overview,key,[0.,0.],radius,numbered=False)
        spatial_overview.extend([
            r'\node[below left,inner sep=2pt] at (0,0) {$0$};',
            r'\node[align=center,text width=7.7cm,anchor=north] at (0,-2.6) {'+title+'};',
            r'\end{scope}',
        ])
        continue
    def project_overview(v): return np.asarray(v)@proj.T*radius
    def draw_overview(points,style):
        spatial_overview.append(r'\draw['+style+r'] plot coordinates {'+path(points)+'};')
    spatial_overview.append(
        rf'\fill[figplane!45!white] (0,0) circle[radius={radius}cm];')
    if key!='line':
        p=np.array([-lam[1],2*lam[2],lam[4]])
        q=np.array([lam[2],2*lam[1],lam[3]])
        u=p/np.linalg.norm(p)
        v=q-np.dot(q,u)*u; v/=np.linalg.norm(v)
        corners=project_overview(1.06*np.array([u+v,u-v,-u-v,-u+v]))
        spatial_overview.append(
            r'\filldraw[fill=accent,fill opacity=.09,draw=accent!30] '+
            '--'.join(f'({x:.5f},{y:.5f})' for x,y in corners)+r'--cycle;')
    tt=np.linspace(0,2*PI,181)
    for aa,bb in [(0,1),(0,2),(1,2)]:
        vec=np.zeros((len(tt),3)); vec[:,aa]=np.cos(tt); vec[:,bb]=np.sin(tt)
        draw_overview(project_overview(vec),'figgray!40,thin,densely dashed')
        angle=np.arctan2(view[bb],view[aa])
        front=np.linspace(angle-PI/2,angle+PI/2,91)
        vec=np.zeros((len(front),3))
        vec[:,aa]=np.cos(front); vec[:,bb]=np.sin(front)
        draw_overview(project_overview(vec),'figgray!55,thin')
    for ax in range(3):
        v=np.eye(3)[ax]
        draw_overview(project_overview([-1.15*v,np.zeros(3)]),
                      'figgray!50,thin,densely dashed')
        start,end=project_overview([np.zeros(3),1.35*v])
        draw_overview([start,end],'->,figink!75,thin')
        anchor=['above left','below right','above'][ax]
        spatial_overview.append(
            f'\\node[{anchor},inner sep=2pt] at ({end[0]:.5f},{end[1]:.5f}) '
            f'{{$s_{["r","t","z"][ax]}$}};')
    curve=primer(lam,np.linspace(0,2*PI,361))
    draw_overview(project_overview(curve),'accent,line width=1.2pt')
    if key in ('ndc','line'):
        x,y=project_overview([0,2*lam[0],0])
        spatial_overview.append(
            f'\\node[accent,inner sep=0pt] at ({x:.5f},{y:.5f}) {{$\\times$}};')
    if key!='sc':
        for point in primer(lam,np.array(cases[key]['phi'])):
            start,end=project_overview([np.zeros(3),point])
            draw_overview([start,end],'->,accent,line width=.55pt')
            spatial_overview.append(
                f'\\filldraw[fill=fignormal,draw=white,line width=.4pt] '
                f'({end[0]:.5f},{end[1]:.5f}) circle[radius=2.3pt];')
    spatial_overview.extend([
        rf'\draw[figgray!65,thin] (0,0) circle[radius={radius}cm];',
        r'\node[below left,inner sep=2pt] at (0,0) {$0$};',
        r'\node[align=center,text width=7.7cm,anchor=north] at (0,-2.6) {'+title+'};',
        r'\end{scope}',
    ])
spatial_overview.extend([
    r'\end{tikzpicture}',
    r'\caption{Годографы базис-вектора в некомпланарной задаче: '
    r'три основных типа решений и предельный случай NDC. '
    r'Серым показана единичная сфера, голубым --- плоскость эллипса, '
    r'крестиком --- центр смещённого годографа. '
    r'В (б) годограф показан в своей плоскости, в (г) --- в плоскости $(s_t,s_z)$; '
    r'серые окружности имеют единичный радиус, оси показаны в косоугольной проекции. '
    r'Золотистые точки отмечают изолированные касания; '
    r'в случае (в) вся окружность лежит на сфере.}',
    r'\label{fig:spatial-hodographs-overview}',r'\end{figure}',
])
(HERE/'spatial-hodographs-overview.tex').write_text('\n'.join(spatial_overview)+'\n')

build=HERE.parent/'build'; build.mkdir(exist_ok=True)
(build/'chapter3-case-data.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
for key,c in cases.items():
 print(key, 'J=', c['J'], 'phases=',np.rad2deg(c['phi']).tolist(), 'residual=',c['max_boundary_residual'])
