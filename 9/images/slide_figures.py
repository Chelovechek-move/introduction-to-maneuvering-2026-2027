#!/usr/bin/env python3
"""Slide-only variants of the 13 figures used in collision-avoidance.tex.

Run from the course root: python3 9/images/slide_figures.py
Only *-slide.pdf files and build/slides-figures previews are written.
The manual's source files, figures and saved numerical data are read-only.
No orbit or collision-probability calculation is repeated. Saved numerical
arrays are used where available; otherwise original PDF vector paths are
preserved and only the paper colour and text font are replaced.
"""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BUILD = HERE.parent / 'build'
PREVIEWS = BUILD / 'slides-figures'
PREVIEWS.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(BUILD / 'mplconfig-slides')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.colors import LogNorm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# pypdf is a pure-Python package in the configured bundled PDF runtime.
BUNDLED = Path('/home/vladimir/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
if str(BUNDLED) not in sys.path:
    sys.path.append(str(BUNDLED))
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import ContentStream, DictionaryObject, FloatObject, NameObject, RectangleObject

PAPER, INK = '#FBF9F3', '#302B28'
PAPER_RGB = tuple(int(PAPER[j:j+2], 16)/255 for j in [1, 3, 5])
INK_RGB = tuple(int(INK[j:j+2], 16)/255 for j in [1, 3, 5])
BLUE, GREEN, ORANGE, PURPLE = '#187AA5', '#087F78', '#B57714', '#79539C'
MM = 1/25.4
PT = Path('/usr/share/fonts/truetype/paratype')
for font in sorted(PT.glob('PTF*.ttf')):
    font_manager.fontManager.addfont(str(font))
for name, fname in [('PTSerif', 'PTF55F.ttf'), ('PTSerifItalic', 'PTF56F.ttf'),
                    ('PTSerifBold', 'PTF75F.ttf'), ('PTSerifBoldItalic', 'PTF76F.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(PT/fname)))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def style():
    plt.rcParams.update({
        'font.family': 'PT Serif', 'font.size': 9.5,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'PT Serif',
        'mathtext.it': 'PT Serif:italic', 'mathtext.bf': 'PT Serif:bold',
        'mathtext.fallback': 'cm', 'axes.labelsize': 9.5,
        'axes.titlesize': 10., 'legend.fontsize': 9.,
        'xtick.labelsize': 9., 'ytick.labelsize': 9.,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
        'savefig.facecolor': PAPER, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.grid': True, 'grid.alpha': .16, 'grid.linewidth': .5,
    })


SAVED = []


def render_preview(path):
    subprocess.run(['pdftoppm', '-f', '1', '-l', '1', '-singlefile', '-scale-to', '1400',
                    '-png', str(path), str(PREVIEWS/path.stem)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pdf_record(path, source, method):
    render_preview(path)
    page = PdfReader(path).pages[0]
    SAVED[:] = [record for record in SAVED if record['file'] != path.name]
    SAVED.append({'file': path.name, 'source': source, 'method': method,
                  'width_mm': float(page.mediabox.width)*25.4/72,
                  'height_mm': float(page.mediabox.height)*25.4/72})


def finish(fig, name, panels=None, layout=True):
    """Restyle a figure without changing its data or coordinate geometry."""
    fig.patch.set_facecolor(PAPER)
    # Keep the source aspect ratio; fonts retain their physical point size.
    old = fig.get_size_inches()
    width = 148*MM
    fig.set_size_inches(width, old[1]/old[0]*width)
    for ax in fig.axes:
        ax.set_facecolor(PAPER)
        for spine in ax.spines.values():
            spine.set_edgecolor(INK)
        if hasattr(ax, 'zaxis'):
            for a in [ax.xaxis, ax.yaxis, ax.zaxis]:
                a.pane.set_facecolor(PAPER)
                a.pane.set_edgecolor(PAPER)
    for text in fig.findobj(Text):
        text.set_fontfamily('PT Serif')
        # Source drawings with explicit 12--15 pt labels were made wider.
        physical_font = {'design-impulse-map': 11.2, 'design-mahalanobis': 12.2,
                         'concept-cylinder': 12., 'concept-principal-axes': 12.,
                         'concept-risk': 11., 'concept-dilution': 11.,
                         'concept-geometry': 11.}.get(name)
        text.set_fontsize(physical_font or min(10., max(9., text.get_fontsize())))
        if text.get_bbox_patch() is not None:
            text.get_bbox_patch().set_facecolor(PAPER)
    if name in ['design-impulse-map', 'design-mahalanobis']:
        for text in fig.texts:
            if text.get_position()[1] < .1:
                text.set_y(.022)
    if name == 'design-mahalanobis':
        fig.axes[0].set_position([.14,.265,.33,.58])
    if name == 'concept-principal-axes':
        for text in fig.axes[0].texts:
            if text.get_text() == 'Главные оси':
                text.set_position((85.,65.)); text.set_fontsize(11.)
        for artist in fig.artists:
            if hasattr(artist,'set_positions'):
                artist.set_positions((.49,.53),(.55,.53))
    if name == 'concept-dilution':
        for text in fig.axes[1].texts:
            if '\\sigma=' in text.get_text():
                text.set_position((90.,2.8)); text.set_zorder(20)
                text.set_bbox({'facecolor':PAPER,'edgecolor':'none','pad':.4})
    if layout:
        fig.tight_layout(pad=.4, w_pad=1.8, h_pad=.7)
    path = HERE/f'{name}-slide.pdf'
    fig.savefig(path, facecolor=PAPER)
    pdf_record(path, name+'.pdf', 'original figure geometry; PT Serif and slide paper')
    if panels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for index, panel_name in panels:
            ax = fig.axes[index]
            if panel_name == 'example-lead-top':
                ax.tick_params(axis='x', labelbottom=True)
                ax.set_xlabel('Время упреждения, доли периода')
            if panel_name == 'example-lead-bottom':
                ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.02),
                          fontsize=9., frameon=False)
            # Do not include neighbouring axes when a newly exposed axis
            # label extends into the original shared subplot gutter.
            for j, other in enumerate(fig.axes):
                other.set_visible(j == index)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            box = fig.axes[index].get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            # A native PDF crop; no rasterization or axis stretching.
            box = Bbox.from_extents(box.x0-.035, box.y0-.035, box.x1+.035, box.y1+.035)
            path = HERE/f'{panel_name}-slide.pdf'
            fig.savefig(path, bbox_inches=box, facecolor=PAPER)
            pdf_record(path, name+'.pdf', 'single original panel; native PDF crop')
        for ax in fig.axes:
            ax.set_visible(True)
    plt.close(fig)
    return {'file_base': name, 'width_mm': 148, 'height_mm': old[1]/old[0]*148,
            'minimum_font_pt': 9.}


def concept_figures():
    geometry = load_module('geometry_probability_slides', 'geometry_probability.py')
    style()
    saved = json.loads((HERE/'concept-data.json').read_text())
    # The source plot functions keep their labels, points and curves; cached
    # answers replace their probability evaluator and scalar optimization.
    lookup = {}
    def key(b, sigma, radius):
        return tuple(np.round(np.r_[b, sigma, radius], 11))
    for case in saved['risk']['cases']:
        lookup[key(case['b_m'], saved['risk']['sigma_m'], saved['risk']['radius_m'])] = case['pc']
    dil = saved['dilution']
    for sigma, pc in zip(dil['sigma_m'], dil['pc']):
        lookup[key(dil['b_m'], [sigma, sigma], dil['radius_m'])] = pc
    for sigma in [20, 500]:
        lookup[key(dil['b_m'], [sigma, sigma], dil['radius_m'])] = dil[f'pc_sigma_{sigma}']
    geometry.pc_diagonal = lambda b, sigma, radius, tight=False: lookup[key(b, sigma, radius)]
    geometry.minimize_scalar = lambda *args, **kwargs: SimpleNamespace(
        x=dil['maximum_sigma_m'], fun=-dil['maximum_pc'])
    def save(fig, name):
        panels = [(0, 'concept-geometry-a'), (1, 'concept-geometry-b')] if name == 'concept-geometry' else None
        return finish(fig, name, panels)
    geometry.save = save
    geometry.geometry()
    geometry.risk()
    geometry.dilution()


def design_figures(result):
    # Importing via the source module retains its exact construction.
    # Register the name expected by design_diagrams' read-only import.
    load_module('avoidance_example', 'avoidance_example.py')
    diagrams = load_module('design_diagrams_slides', 'design_diagrams.py')
    style()
    diagrams.save = lambda fig, name: finish(fig, name, layout=False)
    diagrams.cylinder()
    diagrams.principal_axes()
    diagrams.mahalanobis()
    diagrams.impulse_map(result)
    # fixed_direction() would re-evaluate Pc; preserve its existing vector
    # drawing instead of repeating the probability calculation.
    restyle_pdf('design-fixed-direction')


def saved_example_curves(result):
    data = np.load(HERE/'example-data.npz')
    palette = [BLUE, GREEN, ORANGE, PURPLE, '#C44847']
    ff = data['lead_s']/result['primary_period_s']
    # Identical ordinates, line styles, limits, markers and legend entries.
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.80), sharex=True)
    for j, (name, color) in enumerate(zip(['Радиальный', 'Трансверсальный', 'Нормальный'], palette)):
        axs[0].plot(ff, data['axis_unit_miss_m'][:, j]/1000., color=color, label=name)
        vals = np.abs(data['axis_signed_dv_m_s'][:, j])
        axs[1].plot(ff, np.where(vals < 10., vals, np.nan), color=color, label=name)
    axs[0].set_ylabel(r'Промах $\|\mathbf{b}\|$, км')
    axs[1].set(ylabel=r'Достаточный $|\Delta V|$, м/с',
               xlabel='Время упреждения, доли периода', ylim=(.005, 10.))
    axs[1].set_yscale('log')
    axs[1].plot(ff, data['free_direction_norm_m_s'], color=PURPLE,
                lw=1.8, label='Свободное направление')
    selected = result['lead_time_comparison']
    axs[1].scatter([z['lead_periods'] for z in selected],
                    [data['free_direction_norm_m_s'][z['sweep_index']] for z in selected],
                    color=PURPLE, marker='D', s=45, zorder=5)
    axs[0].legend(ncol=3, loc='upper left', fontsize=9.)
    axs[1].legend(loc='upper right', fontsize=9.)
    for ax in axs:
        ax.axvline(.4, color='#999999', ls=':', lw=1.)
    finish(fig, 'example-lead-time', [(0, 'example-lead-top'), (1, 'example-lead-bottom')])

    risk_map_slide(result)


def risk_map_slide(result=None):
    """Square plot area for the left column of slide 40, from saved NPZ.

    This helper can be imported and called on its own. It writes only
    example-risk-map-slide.pdf and that asset's build preview.
    The native PDF is 80 x 63 mm; the data rectangle is 52 x 52 mm.
    """
    if result is None:
        result = json.loads((HERE/'example-results.json').read_text())
    data = np.load(HERE/'example-data.npz')
    ff = data['lead_s']/result['primary_period_s']
    style()
    with plt.rc_context({'font.size':8.5, 'axes.labelsize':8.5,
                         'xtick.labelsize':7.5, 'ytick.labelsize':8.,
                         'xtick.major.pad':2., 'ytick.major.pad':2.,
                         'axes.labelpad':2.}):
        fig = plt.figure(figsize=(80*MM,63*MM))
        ax = fig.add_axes([14/80,8.5/63,52/80,52/63])
        ax.grid(False)
        mesh = ax.pcolormesh(data['map_amplitudes_m_s'],ff,
                            np.maximum(data['map_pc'],1e-7),
                            norm=LogNorm(1e-7,1e-2),cmap='viridis',shading='auto',
                            edgecolors='none',linewidth=0,rasterized=True)
        ax.contour(data['map_amplitudes_m_s'],ff,data['map_pc'],
                   levels=[result['pc_limit']],colors='white',linewidths=1.4)
        ax.set_box_aspect(1)
        ax.set_xlabel(r'Трансверсальный импульс $\Delta V_t$, м/с')
        ax.set_ylabel('Упреждение,\nдоли периода')
        ax.set_xticks([-.15,-.10,-.05,0.,.05,.10,.15])
        ax.set_yticks([.25,.5,.75,1.,1.25,1.5,1.75,2.])
        ax.tick_params(length=2.5)
        cax = fig.add_axes([69/80,8.5/63,2.6/80,50/63])
        cb = fig.colorbar(mesh,cax=cax)
        cb.set_ticks([1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
        cb.ax.tick_params(labelsize=8.,length=2.5,pad=2.)
        fig.text(70.3/80,61./63,'$P_c$',fontsize=8.5,ha='center',va='center')
        fig.canvas.draw()
        size_mm = ax.get_window_extent().size/fig.dpi*25.4
        assert np.allclose(size_mm,[52.,52.],atol=1e-8),size_mm
        path = HERE/'example-risk-map-slide.pdf'
        fig.savefig(path,facecolor=PAPER,dpi=600)
        pdf_record(path,'example-data.npz','saved heatmap and contour; physical square plot 52 x 52 mm')
        plt.close(fig)
    return {'file':path.name,'width_mm':80.,'height_mm':63.,
            'plot_width_mm':float(size_mm[0]),'plot_height_mm':float(size_mm[1]),
            'source_shape':list(data['map_pc'].shape),'pc_limit':result['pc_limit']}


def restyle_pdf(name):
    """Preserve every source vector path; replace text with embedded PT Serif.

    The source PDF text matrices are retained, including rotated labels,
    sub/superscript locations and colour. Only BT...ET text objects are
    removed. All paths, clipping paths, alpha values and object occlusion
    stay in the PDF content stream, rather than being rasterized.
    """
    source = HERE/f'{name}.pdf'
    reader = PdfReader(source)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    runs, stack = [], []
    state = {'rgb': (0., 0., 0.)}
    def operand(op, args, cm, tm):
        if op == b'q':
            stack.append(state['rgb'])
        elif op == b'Q' and stack:
            state['rgb'] = stack.pop()
        elif op == b'rg':
            state['rgb'] = tuple(float(v) for v in args)
        elif op == b'g':
            state['rgb'] = (float(args[0]),)*3
    def visitor(text, cm, tm, font, size):
        text = text.strip()
        if text:
            base = str(font.get('/BaseFont', '')) if font else ''
            runs.append((text, cm.copy(), tm.copy(), base, size, state['rgb']))
    page.extract_text(visitor_operand_before=operand, visitor_text=visitor)
    content = ContentStream(page.get_contents(), writer)
    out, inside = [], False
    for args, op in content.operations:
        if op == b'BT':
            inside = True
            continue
        if inside:
            if op == b'ET':
                inside = False
            continue
        if (op == b'rg' and all(float(v) > .999 for v in args)) or (op == b'g' and float(args[0]) > .999):
            args, op = [FloatObject(v) for v in PAPER_RGB], b'rg'
        out.append((args, op))
    content.operations = out
    page.replace_contents(content)
    # All original BT...ET runs were replaced, so their font resources are
    # unused. The overlay introduces only the embedded PT Serif faces.
    page['/Resources'][NameObject('/Font')] = DictionaryObject()
    buf = io.BytesIO()
    drawing = canvas.Canvas(buf, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
    processed = []
    grouped = {}
    for run in runs:
        grouped.setdefault(tuple(run[1]), []).append(run)
    for key, group in grouped.items():
        joined = ''.join(run[0] for run in group)
        factor = 1.28 if name == 'design-fixed-direction' and not joined.startswith(('а)','б)')) else 1.
        footer = name == 'design-fixed-direction' and key[5] < 20.
        if footer:
            factor = 1.1
        for text,cm,tm,original_font,size,rgb in group:
            cm = cm.copy()
            if footer and '09913' in joined:
                cm[4] -= 15.
            tm = tm.copy(); tm[4] *= factor; tm[5] *= factor
            processed.append((text,cm,tm,original_font,size*factor,rgb))
    for text, cm, tm, original_font, size, rgb in processed:
        italic = any(k in original_font for k in ['Italic', 'Oblique'])
        bold = 'Bold' in original_font
        font = 'PTSerif' + ('Bold' if bold else '') + ('Italic' if italic else '')
        drawing.saveState()
        drawing.transform(*cm)
        drawing.transform(*tm)
        drawing.setFont(font, size)
        drawing.setFillColorRGB(*(INK_RGB if max(rgb) < .05 else rgb))
        drawing.drawString(0, 0, text)
        drawing.restoreState()
    drawing.showPage()
    drawing.save()
    overlay = PdfReader(buf).pages[0]
    page.merge_page(overlay)
    path = HERE/f'{name}-slide.pdf'
    with path.open('wb') as stream:
        writer.write(stream)
    pdf_record(path, source.name, 'unchanged original native vector paths; PT Serif text and slide paper')
    if name == 'example-local-encounter':
        # Source clipping rectangles: left 60.028..375.758 pt,
        # right 471.883..665.483 pt. Keep each panel's labels and margins.
        crop_pdf(path, 'example-local-distance', (0., 0., 387., float(page.mediabox.height)))
        crop_pdf(path, 'example-local-bplane', (413., 0., float(page.mediabox.width), float(page.mediabox.height)))


def crop_pdf(source, output, box):
    page = PdfReader(source).pages[0]
    x0, y0, x1, y1 = box
    target = PageObject.create_blank_page(width=x1-x0, height=y1-y0)
    target.merge_transformed_page(page, Transformation().translate(-x0, -y0))
    writer = PdfWriter()
    writer.add_page(target)
    path = HERE/f'{output}-slide.pdf'
    with path.open('wb') as stream:
        writer.write(stream)
    pdf_record(path, source.name, 'single original panel; native PDF crop')


def vector_source(name):
    """Read the existing PDF paths and decoded text; no dynamic evaluation."""
    reader = PdfReader(HERE/f'{name}.pdf')
    owner = PdfWriter()
    owner.add_page(reader.pages[0])
    page = owner.pages[0]
    runs = []
    def visitor(text, cm, tm, font, size):
        if text.strip():
            runs.append((text.strip(), cm.copy(), tm.copy(), size))
    page.extract_text(visitor_text=visitor)
    stream = ContentStream(page.get_contents(), owner)
    # Remove only the original legend's frame/handles, not curves hidden
    # underneath it. Matplotlib emits the whole legend as one q...Q group
    # for the square figures, and as a short final block of the first axes
    # for the local-distance figure.
    legend_corner = {'example-bplane': (70.828125,433.13),
                     'example-orbits': (14.9,432.018125),
                     'example-local-encounter': (256.60133,271.05625)}.get(name)
    excluded = set()
    if legend_corner:
        ops = stream.operations
        start_path = next(i for i,(a,op) in enumerate(ops)
                          if op == b'm' and len(a)==2 and np.allclose(np.array(a,float),legend_corner,atol=1e-4))
        if name == 'example-local-encounter':
            begin = start_path-6
            end = next(i for i in range(start_path+1,len(ops)) if ops[i][1]==b'gs' and str(ops[i][0][0])=='/A1')
        else:
            begin = max(i for i in range(start_path) if ops[i][1]==b'q')
            depth = 0
            for i in range(begin,len(ops)):
                depth += int(ops[i][1]==b'q')-int(ops[i][1]==b'Q')
                if depth == 0:
                    end = i+1
                    break
        excluded = set(range(begin,end))
    operations, inside = [], False
    for i,(args, op) in enumerate(stream.operations):
        if i in excluded:
            continue
        if op == b'BT':
            inside = True
            continue
        if inside:
            if op == b'ET':
                inside = False
            continue
        if (op == b'g' and float(args[0]) > .999) or (op == b'rg' and all(float(v) > .999 for v in args)):
            args, op = [FloatObject(v) for v in PAPER_RGB], b'rg'
        operations.append((args, op))
    stream.operations = operations
    page.replace_contents(stream)
    page['/Resources'][NameObject('/Font')] = DictionaryObject()
    return page, runs


def compact_native(name, output, width_mm, height_mm, source_box, target_box, kind):
    """Fit untouched vector geometry into a physical slide-sized panel.

    Axis labels and legends are re-typeset in physical points, instead of
    shrinking the manual's 180-mm chart and its text together.
    """
    page, runs = vector_source(name)
    x0, y0, x1, y1 = source_box
    tx, ty, target_width = [v*72/25.4 for v in target_box]
    scale = target_width/(x1-x0)
    dx, dy = tx-scale*x0, ty-scale*y0
    w, h = width_mm*72/25.4, height_mm*72/25.4
    writer = PdfWriter()
    target = writer.add_blank_page(w, h)
    bg = io.BytesIO()
    c = canvas.Canvas(bg, pagesize=(w, h))
    c.setFillColorRGB(*PAPER_RGB); c.rect(0, 0, w, h, fill=1, stroke=0)
    c.showPage(); c.save()
    target.merge_page(PdfReader(bg).pages[0])
    page.cropbox = RectangleObject(source_box)
    target.merge_transformed_page(page, Transformation((scale, 0, 0, scale, dx, dy)))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFillColorRGB(*INK_RGB)
    def text(x, y, string, size=7.5, align='left', angle=0, color=INK_RGB):
        c.saveState(); c.translate(x, y); c.rotate(angle)
        c.setFillColorRGB(*color); c.setFont('PTSerif', size)
        getattr(c, {'left':'drawString', 'right':'drawRightString', 'center':'drawCentredString'}[align])(0, 0, string)
        c.restoreState()
    def mask(box):
        a, b, e, f = box
        c.setFillColorRGB(*PAPER_RGB)
        c.rect(dx+scale*a, dy+scale*b, scale*(e-a), scale*(f-b), fill=1, stroke=0)
    def legend(x, y, entries, size=7.4, step=7.0):
        for label, color, marker in entries:
            rgb = tuple(int(color[j:j+2], 16)/255 for j in [1,3,5])
            c.setStrokeColorRGB(*rgb); c.setFillColorRGB(*rgb); c.setLineWidth(.8)
            if marker == 'line':
                c.line(x, y+2, x+6, y+2)
            elif marker == 'triangle':
                path = c.beginPath(); path.moveTo(x+3, y+4.7); path.lineTo(x, y-.6); path.lineTo(x+6, y-.6); path.close()
                c.drawPath(path, fill=1, stroke=0)
            else:
                c.circle(x+3, y+2, 2., fill=1, stroke=0)
            for k, line in enumerate(label.split('\n')):
                text(x+9, y-k*(size+1), line, size)
            y -= step*72/25.4
    if kind == 'bplane':
        # The original tick positions and data limits remain unchanged;
        # show three labelled ticks to keep a 35-mm square readable.
        data_lo, data_hi = -650., 650.
        for tick in [-600, 0, 600]:
            f = (tick-data_lo)/(data_hi-data_lo)
            text(tx+f*target_width, ty-10., str(tick).replace('-', '−'), 7.5, 'center')
            text(tx-4.5, ty+f*target_width-2.5, str(tick).replace('-', '−'), 7.5, 'right')
        text(tx+target_width/2, 3.2, 'b₁, м', 8., 'center')
        text(9., ty+target_width/2, 'b₂, м', 8., 'center', 90)
        grouped = {}
        for string, cm, tm, size in runs:
            if size <= 10.01:
                grouped.setdefault(tuple(cm), []).append((string, tm, size))
        # Keep all original contour-level values at the original locations.
        used = set()
        for cm, group in grouped.items():
            joined = ''.join(v[0] for v in group)
            if joined not in ['10−7','10−5','10−3'] or joined in used:
                continue
            used.add(joined)
            angle = np.rad2deg(np.arctan2(cm[1], cm[0]))
            c.saveState(); c.translate(dx+scale*cm[4], dy+scale*cm[5]); c.rotate(angle)
            c.setFillColorRGB(*PAPER_RGB); c.rect(-1, -1, 16, 9, fill=1, stroke=0)
            c.setFillColorRGB(*tuple(int(PURPLE[j:j+2],16)/255 for j in [1,3,5]))
            c.setFont('PTSerif', 6.8); c.drawString(0,0,'10')
            c.setFont('PTSerif', 5.); c.drawString(6.4,3.2,joined[2:])
            c.restoreState()
        legend(1.5*72/25.4, 46.3*72/25.4,
               [('Без манёвра', '#302B28', 'circle')], size=7.2)
        legend(23.5*72/25.4, 46.3*72/25.4,
               [('Максимальный промах', BLUE, 'circle')], size=7.2)
        legend(16.*72/25.4, 42.6*72/25.4,
               [('Минимальный импульс', PURPLE, 'triangle')], size=7.2)
    elif kind == 'distance':
        source_y_height = scale*(y1-y0)
        for tick in [-.10, -.05, 0., .05, .10]:
            # The source x limits are the exact Matplotlib ±0.132 s margin.
            xx = tx+(tick+.132)/.264*target_width
            text(xx, ty-10., f'{tick:.2f}'.replace('-', '−'), 7.5, 'center')
        # Recover the source's y affine transform from its labelled grid.
        ygrid = [(0, 43.829199001), (200, 86.5571350973), (400, 129.2850711937),
                 (600, 172.01300729), (800, 214.7409433863),
                 (1000, 257.4688794827), (1200, 300.196815579)]
        for value, source_y in ygrid:
            text(tx-4., dy+scale*source_y-2.5, str(value), 7.5, 'right')
        text(tx+target_width/2, 3.2, 'Время от исходного TCA, с', 7.5, 'center')
        text(7., ty+source_y_height/2, 'Расстояние, м', 8., 'center', 90)
        legend(tx+target_width-68., ty+source_y_height-10.,
               [('Без манёвра', '#687582', 'line'), ('После манёвра', GREEN, 'line')], step=4.5)
    elif kind == 'orbits':
        grouped = {}
        excluded = ['Управляемый КА', 'Опасный объект', 'Область сближения']
        for string, cm, tm, size in runs:
            if string not in excluded:
                grouped.setdefault(tuple(cm), []).append((string, tm, size))
        for cm, group in grouped.items():
            joined = ''.join(v[0] for v in group)
            old_size = max(v[2] for v in group)
            old_width = pdfmetrics.stringWidth(joined, 'PTSerif', old_size)*scale
            angle = np.rad2deg(np.arctan2(cm[1], cm[0]))
            if 'км' in joined:
                positions = {'X, км':(17.,1.,0.), 'Y, км':(43.,2.4,45.), 'Z, км':(46.,23.5,90.)}
                px,py,pa = positions[joined]
                text(px*72/25.4,py*72/25.4,joined,7.5,'center',pa)
                continue
            px,py = dx+scale*cm[4],dy+scale*cm[5]
            if 325 < cm[4] < 455 and cm[5] < 130:
                if joined == '−6000':
                    px += 3.*72/25.4; py += .5*72/25.4
                else:
                    py += 1.5*72/25.4
            c.saveState(); c.translate(px,py); c.rotate(angle)
            c.translate(old_width/2, 0)
            c.setFont('PTSerif', 7.)
            c.setFillColorRGB(*INK_RGB); c.drawCentredString(0,0,joined); c.restoreState()
        legend(44.25*72/25.4, 46*72/25.4,
               [('Управляемый\nКА', BLUE, 'line'),
                ('Опасный\nобъект', '#C44847', 'line'),
                ('Область\nсближения', '#302B28', 'circle')], size=7.2,step=9.)
    c.showPage(); c.save()
    target.merge_page(PdfReader(buf).pages[0])
    path = HERE/f'{output}-slide.pdf'
    with path.open('wb') as stream:
        writer.write(stream)
    pdf_record(path, name+'.pdf', 'original clipped vector paths; physical 7.5–8 pt labels and external legend')


def compact_native_figures():
    compact_native('example-bplane', 'example-bplane', 65., 49.,
                   (62.778125,42.325,513.658125,493.205), (15.,8.,33.), 'bplane')
    compact_native('example-local-encounter', 'example-local-distance', 65., 47.,
                   (60.028125,42.325,375.75758,308.525), (11.,8.,43.), 'distance')
    compact_native('example-orbits', 'example-orbits', 67., 49.,
                   (0.,0.,496.44,496.44), (.5,1.,45.), 'orbits')


def saved_lead_panels_native(result, data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper, ink = '#FBF9F3', '#302B28'
    colors = ['#187AA5', '#087F78', '#B57714', '#79539C']
    for font in sorted(Path('/usr/share/fonts/truetype/paratype').glob('PTF*.ttf')):
        font_manager.fontManager.addfont(str(font))
    style = {
        'font.family': 'PT Serif', 'font.size': 9.,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'PT Serif',
        'mathtext.it': 'PT Serif:italic', 'mathtext.bf': 'PT Serif:bold',
        'mathtext.fallback': 'cm', 'axes.labelsize': 9.,
        'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': ink, 'text.color': ink, 'axes.labelcolor': ink,
        'xtick.color': ink, 'ytick.color': ink,
        'figure.facecolor': paper, 'axes.facecolor': paper,
        'savefig.facecolor': paper, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.grid': True, 'grid.alpha': .16, 'grid.linewidth': .5,
        'xtick.major.pad': 2., 'ytick.major.pad': 2.,
        'axes.labelpad': 2.,
    }
    data = np.load(data_dir/'example-data.npz')
    ff = data['lead_s']/result['primary_period_s']
    names = ['Радиальный', 'Трансверсальный', 'Нормальный']
    selected = result['lead_time_comparison']
    paths = []
    with plt.rc_context(style):
        for bottom, height_mm in [(False,46.), (True,45.)]:
            fig = plt.figure(figsize=(148/25.4, height_mm/25.4))
            # Fixed physical margins reserve separate space for the legend,
            # tick labels and x label. Nothing is removed by PDF cropping.
            left_mm, right_mm, bottom_mm = 14., 3., 9.5
            top_mm = 10.7 if bottom else 6.9
            ax = fig.add_axes([left_mm/148., bottom_mm/height_mm,
                              (148.-left_mm-right_mm)/148.,
                              (height_mm-bottom_mm-top_mm)/height_mm])
            handles = []
            for j, (name, color) in enumerate(zip(names,colors)):
                if bottom:
                    values = np.abs(data['axis_signed_dv_m_s'][:,j])
                    values = np.where(values<10.,values,np.nan)
                else:
                    values = data['axis_unit_miss_m'][:,j]/1000.
                line, = ax.plot(ff, values, color=color, label=name)
                handles.append(line)
            ax.axvline(.4,color='#999999',ls=':',lw=1.)
            ax.set_xlabel('Время упреждения, доли периода')
            # Same default 5% horizontal data margins as the saved plot.
            ax.set_xlim(ff.min()-.05*np.ptp(ff),ff.max()+.05*np.ptp(ff))
            if bottom:
                ax.set_ylabel(r'Достаточный $|\Delta V|$, м/с')
                ax.set_yscale('log')
                ax.set_ylim(.005,10.)
                ax.set_yticks([.01,.1,1.,10.])
                line, = ax.plot(ff,data['free_direction_norm_m_s'],
                               color=colors[3],lw=1.8,label='Свободное направление')
                handles.append(line)
                ax.scatter([z['lead_periods'] for z in selected],
                           [data['free_direction_norm_m_s'][z['sweep_index']]
                            for z in selected],
                           color=colors[3],marker='D',s=45,zorder=5,clip_on=False)
                # Matplotlib fills legend columns; reorder handles so that
                # the rows read radial/transverse, then normal/free direction.
                order = [0,2,1,3]
                fig.legend(handles=[handles[j] for j in order],ncol=2,
                           loc='upper center',bbox_to_anchor=(.54,.999),
                           borderaxespad=0.,frameon=False,fontsize=8.5,
                           columnspacing=2.,handlelength=2.2,handletextpad=.6,
                           labelspacing=.25)
                name='example-lead-bottom-slide.pdf'
            else:
                ax.set_ylabel(r'Промах $\|\mathbf{b}\|$, км')
                fig.legend(handles=handles,ncol=3,loc='upper center',
                           bbox_to_anchor=(.54,.999),borderaxespad=0.,
                           frameon=False,fontsize=8.5,columnspacing=1.6,
                           handlelength=2.2,handletextpad=.6)
                name='example-lead-top-slide.pdf'
            path=output_dir/name
            fig.savefig(path,facecolor=paper)
            plt.close(fig)
            paths.append(path)
    return paths


def main():
    manifest_path = BUILD/'presentation-source-manifest.json'
    protected = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [HERE.parent/'collision-avoidance.tex', HERE.parent/'collision-avoidance.pdf', *HERE.iterdir()]
        if path.is_file() and path.name != 'slide_figures.py' and not path.name.endswith('-slide.pdf')}
    result = json.loads((HERE/'example-results.json').read_text())
    concept_figures()
    design_figures(result)
    style()
    saved_example_curves(result)
    for name in ['example-orbits', 'example-bplane', 'example-local-encounter']:
        restyle_pdf(name)
    compact_native_figures()
    # Each lead-time panel has its final physical slide size and own legend.
    lead_paths = saved_lead_panels_native(result, HERE, HERE)
    lead_names = {path.name for path in lead_paths}
    SAVED[:] = [row for row in SAVED if row['file'] not in lead_names]
    for path in lead_paths:
        pdf_record(path, 'example-data.npz', 'saved numerical arrays; native one-panel slide size')
    changed = [str(p) for p, digest in protected.items()
               if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError('Protected manual source changed: '+', '.join(changed))
    (PREVIEWS/'slide-figures-manifest.json').write_text(json.dumps(
        {'reproduce': 'python3 9/images/slide_figures.py', 'paper': PAPER,
         'font': 'PT Serif', 'manual_hashes_unchanged': True, 'figures': SAVED},
        ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'created_pdf': len(SAVED), 'manual_hashes_unchanged': True,
                      'names': [r['file'] for r in SAVED]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
