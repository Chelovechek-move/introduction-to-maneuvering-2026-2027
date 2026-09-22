#!/usr/bin/env python3
"""Recompute lecture 3 figures. Run: python3 3/images/generate.py.

Two-body states use km, s and radians. Figures are editable TikZ/PGFPlots,
compiled with the lecture by pdflatex. Examples and sampling match
one_impulse_maneuvers_lecture.ipynb; plane changes recompute u in the new plane.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
MU = 398600.44158
EARTH = 6378.137
D = np.pi / 180


@dataclass(frozen=True)
class Orbit:
    a: float
    e: float
    w: float = 0.
    i: float = 0.
    node: float = 0.

    @property
    def p(self):
        return self.a * (1 - self.e) * (1 + self.e)

    def basis(self):
        O, i, w = self.node, self.i, self.w
        n = np.array([np.cos(O), np.sin(O), 0.])
        h = np.array([np.sin(i)*np.sin(O), -np.sin(i)*np.cos(O), np.cos(i)])
        t = np.cross(h, n)
        return n*np.cos(w)+t*np.sin(w), -n*np.sin(w)+t*np.cos(w), h

    def state(self, nu):
        P, Q, _ = self.basis()
        radius = self.p / (1 + self.e*np.cos(nu))
        r = radius*(np.cos(nu)*P + np.sin(nu)*Q)
        v = np.sqrt(MU/self.p)*(-np.sin(nu)*P+(self.e+np.cos(nu))*Q)
        return r, v


def unit(v):
    return v / np.linalg.norm(v)


def angle(a, b, normal):
    return np.arctan2(np.dot(normal, np.cross(a, b)), np.dot(a, b)) % (2*np.pi)


def ae_solutions(initial, nu1, a2, e2, plane=None):
    """Return branches with positive and negative final radial velocity."""
    r1, v1 = initial.state(nu1)
    radius = np.linalg.norm(r1)
    c = (a2*(1-e2**2)/radius-1)/e2
    if abs(c) > 1+1e-12:
        return [None, None]
    i2, O2 = plane if plane is not None else (initial.i, initial.node)
    base = Orbit(a2, e2, i=i2, node=O2)
    N = np.array([np.cos(O2), np.sin(O2), 0.])
    h = base.basis()[2]
    u2 = angle(N, r1, h)
    alpha = np.arccos(np.clip(c, -1, 1))
    out = []
    for nu2 in (alpha, (2*np.pi-alpha) % (2*np.pi)):
        orbit = replace(base, w=(u2-nu2) % (2*np.pi))
        r2, v2 = orbit.state(nu2)
        assert np.linalg.norm(r2-r1) < 1e-6, "Impulse must preserve position"
        out.append((float(np.linalg.norm(v2-v1)), orbit, nu2 % (2*np.pi)))
    return out


def aw_solutions(initial, nu1, a2, w2):
    r1, v1 = initial.state(nu1)
    # Extended precision limits cancellation close to the e=1 boundary.
    radius = np.longdouble(np.linalg.norm(r1))
    nu2 = (initial.w+nu1-w2) % (2*np.pi)
    b, c = radius*np.cos(nu2), radius-a2
    disc = b*b-4*a2*c
    if disc < 0:
        return [None, None]
    roots = [(-b+np.sqrt(disc))/(2*a2), (-b-np.sqrt(disc))/(2*a2)]
    out = []
    for e2 in roots:
        # e=0 has no defined perigee; e=1 is the rectilinear boundary here.
        if not 1e-10 < e2 < 1-1e-10:
            out.append(None)
            continue
        orbit = Orbit(a2, e2, w2, initial.i, initial.node)
        r2, v2 = orbit.state(nu2)
        assert np.linalg.norm(r2-r1) < 1e-5
        out.append((float(np.linalg.norm(v2-v1)), orbit, nu2))
    return out


def node_solutions(initial, nu1, a2, e2, O2):
    r1, _ = initial.state(nu1)
    A = r1[0]*np.sin(O2)-r1[1]*np.cos(O2)
    # Modulo pi selects the orbit normal with canonical 0 <= i <= pi.
    i2 = float(np.arctan2(-r1[2], A) % np.pi)
    if abs(np.sin(i2)) < 1e-9:
        return [None, None]  # RAAN undefined on an equatorial orbit.
    return ae_solutions(initial, nu1, a2, e2, (i2, O2))


def harmonic_roots(A, B, C):
    """A cos(x) + B sin(x) + C = 0, generic nondegenerate case."""
    rho = np.hypot(A, B)
    if rho < 1e-12 or abs(C) > rho+1e-10:
        return []
    phi, alpha = np.arctan2(B, A), np.arccos(np.clip(-C/rho, -1, 1))
    return sorted({float((phi+alpha) % (2*np.pi)), float((phi-alpha) % (2*np.pi))})


def worked_states(initial, nu1, solutions):
    """Full Cartesian calculations used in the worked examples on the slides."""
    r1, v1 = initial.state(nu1)
    rows = []
    for dv, target, nu2 in solutions:
        r2, v2 = target.state(nu2)
        rows.append(dict(
            a_km=float(target.a), e=float(target.e), i_deg=float(target.i/D),
            node_deg=float(target.node/D), w_deg=float(target.w/D), nu_deg=float(nu2/D),
            r2_km=np.asarray(r2,dtype=float).tolist(),
            v2_mps=np.asarray(v2*1000,dtype=float).tolist(),
            delta_v_mps=np.asarray((v2-v1)*1000,dtype=float).tolist(),
            delta_v_norm_mps=float(dv*1000),
            position_residual_m=float(np.linalg.norm(r2-r1)*1000)))
    return dict(r1_km=np.asarray(r1,dtype=float).tolist(),
                v1_mps=np.asarray(v1*1000,dtype=float).tolist(), solutions=rows)


def write(name, text):
    (HERE/name).write_text(text, encoding="utf-8")


def orbit_coordinates(orbit, projection, n=241):
    return np.array([projection @ orbit.state(x)[0] for x in np.linspace(0, 2*np.pi, n)])


def coords(points):
    return " ".join(f"({x:.5f},{y:.5f})" for x, y in points)


def orbit_picture(name, orbits, projection, point=None, extra_points=(), width=68, height=49):
    """Equal-scale physical orbit drawing, with Earth drawn to scale."""
    paths = [orbit_coordinates(o, projection) for o in orbits]
    cloud = np.vstack(paths)
    lo, hi = cloud.min(0), cloud.max(0)
    center = (lo+hi)/2
    scale = min((width-7)/(hi[0]-lo[0]), (height-6)/(hi[1]-lo[1]))
    xy = lambda q: (q-center)*scale
    styles = ["draw=courseink,line width=.75pt", "draw=courseaccent,line width=1pt",
              "draw=coursemuted,line width=.8pt,dashed"]
    lines = [r"\begin{tikzpicture}[course diagram,x=1mm,y=1mm]",
             f"\\path[use as bounding box] ({-width/2}, {-height/2}) rectangle ({width/2}, {height/2});"]
    ex, ey = xy(np.zeros(2))
    lines += [f"\\filldraw[fill=coursetint,draw=courserule,line width=.4pt] ({ex:.4f},{ey:.4f}) circle[radius={EARTH*scale:.4f}];",
              f"\\node[inner sep=0pt,font=\\fontsize{{8.5}}{{10}}\\selectfont] at ({ex:.4f},{ey:.4f}) {{$\\oplus$}};"]
    for path, style in zip(paths, styles):
        lines.append(f"\\draw[{style}] plot coordinates {{{coords(xy(path))}}};")
    if point is not None:
        x, y = xy(projection @ point)
        lines += [f"\\draw[aux line] ({ex:.4f},{ey:.4f})--({x:.4f},{y:.4f});",
                  f"\\fill[courseink] ({x:.4f},{y:.4f}) circle[radius=.85];",
                  f"\\node[anchor=south west,inner sep=1mm] at ({x:.4f},{y:.4f}) {{$P$}};"]
    for p, label, anchor in extra_points:
        x, y = xy(projection @ p)
        lines += [f"\\fill[courseaccent] ({x:.4f},{y:.4f}) circle[radius=.8];",
                  f"\\node[anchor={anchor},inner sep=1mm] at ({x:.4f},{y:.4f}) {{{label}}};"]
    lines.append(r"\end{tikzpicture}")
    write(name, "\n".join(lines)+"\n")


def node_orbit_picture(initial, nu1, solutions):
    """Actual 3D orbits for the a/e/node example, in orthographic projection.

    The camera is 30 degrees above the mean plane, at 45 degrees to OP.
    One physical scale is used in both screen directions. Earth occludes
    only orbit arcs behind it; neither inclinations nor distances are exaggerated.
    """
    orbits = [initial] + [s[1] for s in solutions]
    r, _ = initial.state(nu1)
    radial = unit(r)
    mean_normal = unit(orbits[0].basis()[2]+orbits[1].basis()[2])
    transverse = np.cross(mean_normal, radial)
    camera = (np.cos(30*D)*(np.cos(45*D)*radial+np.sin(45*D)*transverse)
              +np.sin(30*D)*mean_normal)
    right = unit(np.cross(mean_normal, camera))
    up = np.cross(camera, right)
    projection = np.array([right, up])
    assert np.allclose(projection @ projection.T, np.eye(2))
    for _, orbit, nu in solutions:
        assert np.linalg.norm(orbit.state(nu)[0]-r) < 1e-6
    assert np.allclose(orbits[1].basis()[2], orbits[2].basis()[2])

    paths = [np.array([o.state(nu)[0] for nu in np.linspace(0, 2*np.pi, 721)])
             for o in orbits]
    projected = [path @ projection.T for path in paths]
    cloud = np.vstack(projected)
    lo, hi = cloud.min(0), cloud.max(0)
    width, height = 68., 47.
    center = (lo+hi)/2
    scale = min((width-9)/(hi[0]-lo[0]), (height-7)/(hi[1]-lo[1]))
    xy = lambda position: (projection @ position-center)*scale
    point = lambda position: '('+','.join(f'{v:.4f}' for v in xy(position))+')'
    styles = ['courseink,line width=.75pt', 'courseaccent,line width=1pt',
              'coursemuted,line width=.8pt,dashed']
    arcs = []
    for path, style in zip(paths, styles):
        depth = path @ camera
        front = bool(depth[0] >= 0)
        arc = [path[0]]
        for j in range(len(path)-1):
            next_front = bool(depth[j+1] >= 0)
            if next_front != front:
                fraction = depth[j]/(depth[j]-depth[j+1])
                crossing = path[j]+fraction*(path[j+1]-path[j])
                arc.append(crossing)
                arcs.append((front, np.array(arc), style))
                arc = [crossing]
                front = next_front
            arc.append(path[j+1])
        arcs.append((front, np.array(arc), style))
    lines = [
        '% Actual orbit coordinates, uniform scale, orthographic camera; generated by node_orbit_picture().',
        r'\begin{tikzpicture}[course diagram,x=1mm,y=1mm]',
        f'\\path[use as bounding box] ({-width/2},{-height/2}) rectangle ({width/2},{height/2});',
    ]
    for foreground in (False, True):
        if foreground:
            lines.append(f'\\filldraw[fill=coursetint,draw=courserule,line width=.45pt] '
                         f'{point(np.zeros(3))} circle[radius={EARTH*scale:.4f}];')
        for front, arc, style in arcs:
            if front == foreground:
                screen = (arc @ projection.T-center)*scale
                lines.append(f'\\draw[{style}] plot coordinates {{{coords(screen)}}};')
    lines += [
        f'\\draw[aux line] {point(np.zeros(3))}--{point(r)};',
        f'\\node[inner sep=0pt,font=\\fontsize{{8.5}}{{10}}\\selectfont] at {point(np.zeros(3))} {{$\\oplus$}};',
        f'\\fill[courseink] {point(r)} circle[radius=.85];',
        f'\\node[anchor=north east,inner sep=1mm] at {point(r)} {{$P$}};',
    ]
    # Fixed-frame directions, translated to an empty corner for readability.
    triad = np.array([-width/2+10, -height/2+9])
    for axis, label, anchor in zip(np.eye(3), ('x', 'y', 'z'), ('east', 'north west', 'west')):
        endpoint = triad+7*(projection @ axis)
        a = '('+','.join(f'{v:.4f}' for v in triad)+')'
        b = '('+','.join(f'{v:.4f}' for v in endpoint)+')'
        lines.append(f'\\draw[coursemuted,line width=.45pt,->] {a}--{b} '
                     f'node[anchor={anchor},inner sep=.6mm] {{${label}$}};')
    lines.append(r'\end{tikzpicture}')
    write('ae-node-orbits.tex', '\n'.join(lines)+'\n')


def intersection_schematic():
    """Two actual ellipses in different planes, shown schematically in projection.

    Both focus at O and cross the common line L at the same two radii.
    P and Q are the orthonormal perifocal basis of orbit 1, with Q = h1 x P.
    This explanatory drawing does not use the numerical transfer example.
    """
    screen_angle = 125*D
    turn = np.array([[np.cos(screen_angle), -np.sin(screen_angle)],
                     [np.sin(screen_angle), np.cos(screen_angle)]])
    view = turn @ np.array([[.45, 0., np.sqrt(1-.45**2)], [0., 1., 0.]])
    L = np.array([1., 0., 0.])
    # Orient the schematic ellipse along the principal directions of the
    # projection of plane 1. P and Q then remain perpendicular on the page,
    # as well as in 3D, while the camera and intersection line stay unchanged.
    transverse1 = np.array([0., np.cos(-20*D), np.sin(-20*D)])
    projected_plane = view @ np.column_stack((L, transverse1))
    _, _, directions = np.linalg.svd(projected_plane)
    peri_direction = directions[0]
    if (projected_plane @ peri_direction)[1] < 0:
        peri_direction = -peri_direction
    phi = np.arctan2(peri_direction[1], peri_direction[0])
    eccentricity = .25
    bases, paths = [], []
    nus = np.linspace(0., 2*np.pi, 361)
    for tilt in (-20*D, 20*D):
        transverse = np.array([0., np.cos(tilt), np.sin(tilt)])
        P = np.cos(phi)*L + np.sin(phi)*transverse
        Q = -np.sin(phi)*L + np.cos(phi)*transverse
        h = np.cross(P, Q)
        assert np.allclose(np.cross(h, P), Q)
        bases.append((P, Q, h))
        radius = (1-eccentricity**2)/(1+eccentricity*np.cos(nus))
        paths.append((view @ ((P[:, None]*np.cos(nus) + Q[:, None]*np.sin(nus))*radius)).T)
    assert np.allclose(unit(np.cross(bases[0][2], bases[1][2])), L)
    cloud = np.vstack(paths)
    center = (cloud.min(0)+cloud.max(0))/2
    scale = 36/(cloud.max(0)[1]-cloud.min(0)[1])
    xy = lambda xyz: (view @ np.asarray(xyz)-center)*scale
    point = lambda xyz: "("+",".join(f"{v:.4f}" for v in xy(xyz))+")"
    P, Q, h1 = bases[0]
    pericenter = (1-eccentricity)*P
    assert np.isclose(np.dot(P, Q), 0.)
    assert np.isclose(np.dot(P, h1), 0.) and np.isclose(np.dot(Q, h1), 0.)
    assert np.isclose(np.dot(view @ P, view @ Q), 0.)
    assert np.allclose(paths[0][0], view @ pericenter)
    lines = [
        "% Schematic, orthographic view. Generated by intersection_schematic().",
        r"\begin{tikzpicture}[course diagram,x=1mm,y=1mm]",
        r"\path[use as bounding box] (-35,-21) rectangle (35,21);",
        f"\\draw[coursemuted,densely dashed,line width=.55pt] {point(-1.8*L)}--{point(1.7*L)};",
    ]
    for path, color in zip(paths, ('courseink', 'courseaccent')):
        lines.append(f"\\draw[{color},line width=.85pt] plot coordinates {{{coords((path-center)*scale)}}};")
    # Emphasize the two common points on the line, without confusing P with a point label.
    for sign in (1, -1):
        radius = (1-eccentricity**2)/(1+eccentricity*sign*np.cos(phi))
        lines.append(f"\\fill[courseink] {point(sign*radius*L)} circle[radius=.55];")
    lines += [
        f"\\draw[vector line,line width=.9pt] {point(np.zeros(3))}--{point(1.5*L)} node[above left,inner sep=.8mm] {{$\\mathbf L$}};",
        f"\\draw[vector line,line width=.9pt] {point(np.zeros(3))}--{point(pericenter)} node[pos=.32,right,inner sep=1mm] {{$\\mathbf P$}};",
        f"\\draw[vector line,line width=.9pt] {point(np.zeros(3))}--{point((1-eccentricity)*Q)} node[below left,inner sep=.7mm] {{$\\mathbf Q$}};",
        f"\\draw[courseink,line width=.55pt] {point(.13*P)}--{point(.13*P+.22*Q)}--{point(.22*Q)};",
        f"\\fill[courseink] {point(pericenter)} circle[radius=.65];",
        f"\\node[anchor=west,inner sep=1mm] at {point(pericenter)} {{Перицентр}};",
        f"\\fill[courseink] {point(np.zeros(3))} circle[radius=.65];",
        f"\\node[anchor=west,xshift=1mm,yshift=.5mm,inner sep=.5mm] at {point(np.zeros(3))} {{$O$}};",
        r"\node[courseink,anchor=west,inner sep=.5mm] at (6,-15) {Орбита 1};",
        r"\node[courseaccent,anchor=west,inner sep=.5mm] at (19,3) {Орбита 2};",
        r"\end{tikzpicture}",
    ]
    write('noncoplanar-orbits.tex', '\n'.join(lines)+'\n')


def plane_constraint_schematic():
    """Project two planes sharing OP, with their actual unit normals.

    Split the opaque plane patches along their common line so the front
    surface changes at the intersection. Normal 2 is translated along OP
    to separate the arrows and their right-angle markers.
    """
    radial = np.array([1., 0., 0.])
    transverse = [np.array([0., 1., 0.]),
                  np.array([0., np.cos(50*D), np.sin(50*D)])]
    normals = [unit(np.cross(radial, t)) for t in transverse]
    azimuth, elevation = -50*D, 22*D
    camera = np.array([np.cos(elevation)*np.cos(azimuth),
                       np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    view = np.array([[-np.sin(azimuth), np.cos(azimuth), 0.],
                     [-np.sin(elevation)*np.cos(azimuth),
                      -np.sin(elevation)*np.sin(azimuth), np.cos(elevation)]])
    assert np.allclose(view @ view.T, np.eye(2))
    assert np.allclose(unit(np.cross(*normals)), radial)
    for normal, tangent in zip(normals, transverse):
        assert np.isclose(np.dot(normal, radial), 0.)
        assert np.isclose(np.dot(normal, tangent), 0.)
    point = lambda xyz: '(' + ','.join(f'{v:.4f}' for v in .9*(view @ xyz)) + ')'
    origin = np.zeros(3)
    half_length = 24.
    widths = (15., 13.)
    colors = ('courseink', 'courseaccent')
    fills = ('courseink!7!coursepaper', 'courseaccent!13!coursepaper')
    lines = [
        '% Orthographic projection of two exact 3D planes; generated by plane_constraint_schematic().',
        r'\begin{tikzpicture}[course diagram,x=1mm,y=1mm]',
        r'\path[use as bounding box] (-34,-19) rectangle (34,35);',
    ]
    patches = []
    for j, (tangent, width) in enumerate(zip(transverse, widths)):
        for sign in (-1, 1):
            edge = sign*width*tangent
            vertices = [-half_length*radial, -half_length*radial+edge,
                        half_length*radial+edge, half_length*radial]
            depth = float(np.dot(camera, .5*edge))
            patches.append((depth, j, vertices))
    for _, j, vertices in sorted(patches, key=lambda p: p[0]):
        path = '--'.join(point(v) for v in vertices)
        lines.append(f'\\fill[{fills[j]}] {path}--cycle;')
        lines.append(f'\\draw[{colors[j]},line width=.65pt] {path};')
    # The same 3D line belongs to both patches, including O and the maneuver point P.
    lines += [
        f'\\draw[coursemuted,densely dashed,line width=.6pt] {point(-29*radial)}--{point(-half_length*radial)};',
        f'\\draw[coursemuted,densely dashed,line width=.6pt] {point(half_length*radial)}--{point(29*radial)};',
        f'\\draw[courseink,line width=.85pt] {point(-half_length*radial)}--{point(half_length*radial)};',
    ]
    for j, (normal, tangent, anchor) in enumerate(zip(normals, transverse, (origin, -10*radial))):
        color = colors[j]
        foot = point(anchor)
        # Two independent directions in the plane are perpendicular to its normal.
        lines.append(f'\\draw[{color},densely dashed,line width=.5pt] {foot}--{point(anchor+7*tangent)};')
        for direction, size in ((tangent, 3.), (radial, 2.5)):
            corner = [anchor+size*normal,
                      anchor+size*(normal+direction), anchor+size*direction]
            lines.append(f'\\draw[{color},line width=.5pt] ' + '--'.join(point(v) for v in corner) + ';')
        lines.append(f'\\draw[{color},line width=.95pt,->] {foot}--{point(anchor+35*normal)}'
                     f' node[above,inner sep=1mm] {{$\\widehat{{\\mathbf h}}_{j+1}$}};')
        if j:
            lines.append(f'\\fill[{color}] {foot} circle[radius=.45];')
    lines += [
        f'\\draw[vector line,line width=1pt] {point(origin)}--{point(19*radial)} node[pos=.58,below=1mm] {{$\\mathbf r$}};',
        f'\\fill[courseink] {point(origin)} circle[radius=.7];',
        f'\\node[below left,inner sep=1mm] at {point(origin)} {{$O$}};',
        f'\\fill[courseink] {point(19*radial)} circle[radius=.75];',
        f'\\node[below right,inner sep=1mm] at {point(19*radial)} {{$P$}};',
        f'\\node[courseink,inner sep=.5mm] at {point(10*radial-12*transverse[0])} {{1}};',
        f'\\node[courseaccent,inner sep=.5mm] at {point(13*radial+10*transverse[1])} {{2}};',
        r'\end{tikzpicture}',
    ]
    write('plane-constraint.tex', '\n'.join(lines)+'\n')


def sample(name, solve):
    rows=[]
    for deg in np.linspace(0,360,720,endpoint=False):
        solutions=solve(deg*D)
        row=[deg]
        for sol in solutions:
            if sol is None:
                row.extend([np.nan]*5)
            else:
                dv, o, _ = sol
                row.extend([dv*1000, o.w/D, o.e, o.a*(1-o.e), o.i/D])
        rows.append(row)
    arr=np.array(rows,dtype=float)
    np.savetxt(HERE/name,arr,fmt="%.9g",header="nu dvA wA eA rpA iA dvB wB eB rpB iB",comments="")
    # Independently refine every sampled local minimum on each continuous branch.
    minima=[]
    for branch,col in enumerate((1,6)):
        y=arr[:,col]
        for j in range(1,len(y)-1):
            if np.all(np.isfinite(y[j-1:j+2])) and y[j]<y[j-1] and y[j]<y[j+1]:
                def objective(deg):
                    sol=solve(deg*D)[branch]
                    return sol[0] if sol is not None else 1e9
                opt=minimize_scalar(objective,bounds=(arr[j-1,0],arr[j+1,0]),method="bounded",options={"xatol":1e-10})
                sol=solve(opt.x*D)[branch]
                minima.append(dict(nu=opt.x,dv=opt.fun*1000,branch=branch,
                                   w=sol[1].w/D,i=sol[1].i/D,e=float(sol[1].e)))
    return sorted(minima,key=lambda x:x['nu'])


def plot(name, data, ycols, ylabel, ymin, ymax, yticks, extra="", width="92mm", height="51mm", legend=None, legend_x=.98, legend_anchor="north east"):
    lines=[r"\begin{tikzpicture}",r"\begin{axis}[course plot,",
           f"width={width},height={height},ymin={ymin},ymax={ymax},ytick={{{yticks}}},",
           f"ylabel={{{ylabel}}},",r"xlabel={$\nu_1$, град},",
           r"xmin=0,xmax=360,xtick={0,60,120,180,240,300,360},",
           r"unbounded coords=jump,filter discard warning=false,",
           f"legend style={{at={{({legend_x},.98)}},anchor={legend_anchor},legend columns=2}},",
           r"]"]
    if extra:
        lines.append(extra)
    for j,ycol in enumerate(ycols):
        style="courseaccent,solid" if j==0 else "coursemuted,densely dashed"
        lines.append(f"\\addplot[{style},line width=.85pt,no marks] table[x=nu,y={ycol}] {{images/{data}}};")
    if legend:
        lines.append("\\legend{"+legend+"}")
    lines += [r"\end{axis}",r"\end{tikzpicture}"]
    write(name,"\n".join(lines)+"\n")


def main():
    BUILD.mkdir(exist_ok=True)
    report={}
    o1,o2=Orbit(10000,.1,20*D),Orbit(12000,.35,90*D)
    dw=o2.w-o1.w
    roots=harmonic_roots(o2.e*o1.p*np.cos(dw)-o1.e*o2.p,o2.e*o1.p*np.sin(dw),o1.p-o2.p)
    report['coplanar']=[]
    for nu in roots:
        r1,v1=o1.state(nu);r2,v2=o2.state(nu-dw)
        assert np.linalg.norm(r2-r1)<1e-8
        vr=lambda o,n: np.sqrt(MU/o.p)*o.e*np.sin(n)
        vt=lambda o,n: np.sqrt(MU/o.p)*(1+o.e*np.cos(n))
        report['coplanar'].append(dict(nu1=nu/D,nu2=((nu-dw) % (2*np.pi))/D,r=float(np.linalg.norm(r1)),
            vr1=vr(o1,nu),vt1=vt(o1,nu),vr2=vr(o2,nu-dw),vt2=vt(o2,nu-dw),
            dv=float(np.linalg.norm(v2-v1))*1000))
    projection=np.array([[1.,0,0],[0,1.,0]])
    orbit_picture('coplanar-orbits.tex',[o1,o2],projection,
                  extra_points=[(o1.state(n)[0],str(k+1),'south west' if k==0 else 'north east') for k,n in enumerate(roots)])
    n1=Orbit(18654.3640,.28969592,265.55428*D,35.62718*D,89.44357*D)
    n2=Orbit(20679.0085,.16932267,234.60634*D,38.46693*D,107.25881*D)
    line=unit(np.cross(n1.basis()[2],n2.basis()[2]))
    report['noncoplanar']=[]
    for sign in (1,-1):
        nu1=angle(n1.basis()[0],sign*line,n1.basis()[2])
        nu2=angle(n2.basis()[0],sign*line,n2.basis()[2])
        r1,v1=n1.state(nu1);r2,v2=n2.state(nu2)
        report['noncoplanar'].append(dict(nu1=nu1/D,nu2=nu2/D,r1=float(np.linalg.norm(r1)),r2=float(np.linalg.norm(r2)),
                                          mismatch_m=float(np.linalg.norm(r2-r1))*1000,dv=float(np.linalg.norm(v2-v1))*1000))
    intersection_schematic()
    plane_constraint_schematic()

    base=Orbit(26000,.6,250*D,64*D,0.)
    ae=ae_solutions(base,60*D,26554,.7)
    P,Q,h=base.basis()
    # In-plane view with the ascending node pointing to the right.
    inplane=np.array([[1.,0,0],[0,np.cos(base.i),np.sin(base.i)]])
    orbit_picture('ae-orbits.tex',[base]+[x[1] for x in ae],inplane,point=base.state(60*D)[0])
    report['ae']=[dict(dv=x[0]*1000,nu=x[2]/D,w=x[1].w/D) for x in ae]
    baseaw=Orbit(26222,.7,266*D,63.434*D,0.)
    aw=aw_solutions(baseaw,222*D,26554,270*D)
    inplaneaw=np.array([[1.,0,0],[0,np.cos(baseaw.i),np.sin(baseaw.i)]])
    orbit_picture('aw-orbits.tex',[baseaw]+[x[1] for x in aw],inplaneaw,point=baseaw.state(222*D)[0])
    report['aw']=[dict(dv=x[0]*1000,nu=x[2]/D,e=float(x[1].e),rp=float(x[1].a*(1-x[1].e))) for x in aw]

    # Same initial orbits, target elements and 0.5 degree grid as the notebook.
    report['opt_ae']=sample('opt-ae.dat',lambda nu:ae_solutions(base,nu,26554,.7))
    report['opt_aw']=sample('opt-aw.dat',lambda nu:aw_solutions(baseaw,nu,26554,270*D))
    report['opt_node']=sample('opt-node.dat',lambda nu:node_solutions(base,nu,26554,.7,3*D))
    plot('opt-ae-dv.tex','opt-ae.dat',['dvA','dvB'],r'$\Delta v$, м/с',0,7000,'0,1000,2000,3000,4000,5000,6000,7000',width='128mm',height='48mm',legend='A,B',legend_x=.5,legend_anchor='north')
    plot('opt-aw-dv.tex','opt-aw.dat',['dvA','dvB'],r'$\Delta v$, м/с',0,3000,'0,500,1000,1500,2000,2500,3000',width='128mm',height='48mm',legend='A,B')
    plot('opt-node-dv.tex','opt-node.dat',['dvA','dvB'],r'$\Delta v$, м/с',0,13000,'0,2000,4000,6000,8000,10000,12000',width='128mm',height='48mm',legend='A,B')

    fixed_node=node_solutions(base,60*D,26554,.7,3*D)
    assert abs(fixed_node[0][0]*1000-1435.7652)<.001
    node_orbit_picture(base,60*D,fixed_node)
    report['worked_examples']={
        'coplanar': [worked_states(o1,nu,[(np.linalg.norm(o2.state(nu-dw)[1]-o1.state(nu)[1]),o2,(nu-dw) % (2*np.pi))])
                     for nu in roots],
        'ae': worked_states(base,60*D,ae),
        'aw': worked_states(baseaw,222*D,aw),
        'ae_node': worked_states(base,60*D,fixed_node),
    }
    (BUILD/'calculations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
