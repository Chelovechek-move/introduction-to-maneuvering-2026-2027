#!/usr/bin/env python3
"""Draw the approximate Galileo recovery geometry from published ESA heights.

Run from the course root: python3 11/images/galileo_case.py
The ellipses have a common, arbitrary orientation. They are not ephemerides.
Input heights and source URLs are preserved in galileo-data.json.
"""
from pathlib import Path
import json
import os

BASE = Path(__file__).resolve().parent
BUILD = BASE.parent / 'build'
PREVIEW = BUILD / 'preview'
PREVIEW.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BUILD / 'mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BLUE, GREEN, ORANGE = '#187AA5', '#087F78', '#B57714'
GRAY, INK = '#687582', '#25323F'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 13,
    'axes.labelsize': 13, 'legend.fontsize': 11.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42})

RE = 6378.1363
DATA = {'earth_radius_km': RE,
    'initial_heights_km': [13713., 25900.],
    'recovered_heights_km': [17200., 25900.],
    'nominal_circular_height_km': 23222.,
    'orientation': 'common arbitrary apsidal direction; no inclination information',
    'note': 'Heights from two ESA summaries, rounded and not a single fitted epoch.',
    'sources': [
        {'title': 'Galileo satellite recovered and transmitting navigation signals',
         'date': '2014-12-03',
         'url': 'https://www.esa.int/Newsroom/Press_Releases/Galileo_satellite_recovered_and_transmitting_navigation_signals'},
        {'title': 'Galileo: Facts and figures',
         'accessed': '2026-11-17',
         'url': 'https://www.esa.int/Applications/Satellite_navigation/Galileo/Facts_and_figures'}]}

def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.1),
                             gridspec_kw={'width_ratios': [1.12, 1.]})
    ax = axes[0]
    nu = np.linspace(0, 2*np.pi, 1000)
    for key, label, color in [('initial_heights_km', 'После выведения', ORANGE),
                               ('recovered_heights_km', 'После восстановления', GREEN)]:
        rp, ra = RE + np.array(DATA[key])
        a = (rp+ra)/2
        e = (ra-rp)/(ra+rp)
        r = a*(1-e*e)/(1+e*np.cos(nu))
        ax.plot(r*np.cos(nu)/1000, r*np.sin(nu)/1000, color=color,
                lw=2.2, label=label)
    r = RE + DATA['nominal_circular_height_km']
    ax.plot(r*np.cos(nu)/1000, r*np.sin(nu)/1000, color=BLUE,
            ls='--', lw=1.7, label='Штатная круговая')
    ax.add_patch(Circle((0, 0), RE/1000, facecolor=BLUE, alpha=.16,
                       edgecolor=BLUE))
    ax.text(0, 0, 'Земля', ha='center', va='center', fontsize=11, color=INK)
    ax.set_aspect('equal')
    ax.set_xlim(-35, 34)
    ax.set_ylim(-33, 33)
    ax.set_xlabel('Координата в плоскости орбиты, тыс. км')
    ax.set_ylabel('Координата, тыс. км')
    ax.grid(alpha=.15)
    ax.legend(loc='upper center', bbox_to_anchor=(.49, 1.21), frameon=False)
    ax = axes[1]
    specs = [('После\nвыведения', DATA['initial_heights_km'], ORANGE),
             ('После\nвосстановления', DATA['recovered_heights_km'], GREEN),
             ('Штатная\nкруговая', [23222., 23222.], BLUE)]
    for j, (label, heights, color) in enumerate(specs):
        ax.plot([j, j], np.array(heights)/1000, lw=5, color=color, alpha=.7)
        ax.plot([j, j], np.array(heights)/1000, 'o', color=color, ms=7)
        low, high = heights
        if low == high:
            ax.text(j, low/1000+.6, f'{low:,.0f}'.replace(',', ' '),
                    ha='center', color=color)
        else:
            ax.text(j, low/1000-.55, f'{low:,.0f}'.replace(',', ' '),
                    ha='center', va='top', color=color)
            ax.text(j, high/1000+.55, f'{high:,.0f}'.replace(',', ' '),
                    ha='center', color=color)
    ax.set_xticks(range(3), [s[0] for s in specs], fontsize=10.5)
    ax.set_xlim(-.55, 2.55)
    ax.set_ylim(11.5, 28.3)
    ax.set_ylabel('Высота над поверхностью Земли, тыс. км')
    ax.grid(axis='y', alpha=.15)
    ax.text(.5, 1.08, 'Перигей и апогей: подписи в км', transform=ax.transAxes,
            ha='center', fontsize=11)
    fig.subplots_adjust(left=.075, right=.99, bottom=.16, top=.85, wspace=.34)
    fig.savefig(BASE/'galileo-recovery.pdf', bbox_inches='tight', pad_inches=.04)
    fig.savefig(PREVIEW/'galileo-recovery.png', dpi=170, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    (BASE/'galileo-data.json').write_text(json.dumps(DATA, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
