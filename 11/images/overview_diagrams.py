#!/usr/bin/env python3
"""Overview diagrams for lecture 11.

Run from the course root: python3 11/images/overview_diagrams.py
All diagrams are conceptual: positions and lengths have no numerical scale.
The Akatsuki timeline records historical milestones, not an ephemeris.
The JSON files preserve labels, source references, and the mapping from the
18 stages of last year's lecture to five larger stages.
"""
from pathlib import Path
import json
import os

BASE = Path(__file__).resolve().parent
BUILD = BASE.parent / 'build'
PREVIEW = BUILD / 'overview-preview'
PREVIEW.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(BUILD / 'mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

BLUE, GREEN, ORANGE = '#187AA5', '#087F78', '#B57714'
PURPLE, RED, INK, GRAY = '#79539C', '#C44847', '#25323F', '#687582'
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 15,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

STAGES = [
    {'title': 'Оценка состояния и результата',
     'details': ['Телеметрия; орбита и ориентация.',
                 'Обновление оценок на борту и Земле.',
                 'Реконструкция и калибровка манёвра.'],
     'previous_stage_numbers': [1, 2, 3, 4, 5, 8], 'color': BLUE},
    {'title': 'Прогноз событий и ограничений',
     'details': ['Сеансы связи, тень, ограничения ориентации.',
                 'Прогноз опасных сближений.'],
     'previous_stage_numbers': [6, 7], 'color': GREEN},
    {'title': 'Расчёт продолжения',
     'details': ['Манёвры, модель ДУ, анализ ошибок и резервов.',
                 'Резервные действия при нештатном исполнении.'],
     'previous_stage_numbers': [9, 10, 11, 14], 'color': ORANGE},
    {'title': 'Подготовка и проверка команд',
     'details': ['Параметры манёвра и ориентации.',
                 'Послеманёвровый прогноз событий.',
                 'Проверка исполнения на имитационной модели.'],
     'previous_stage_numbers': [12, 13, 15], 'color': PURPLE},
    {'title': 'Исполнение и контроль',
     'details': ['Согласование и загрузка команд.',
                 'Контроль хода манёвра и состояния КА.'],
     'previous_stage_numbers': [16, 17, 18], 'color': RED},
]
DATA = {
    'kind': 'conceptual diagrams; no numerical time, fuel or geometric scale',
    'planning_phases': [
        {'phase': 'До запуска', 'input': 'Разбросы и резервные сценарии'},
        {'phase': 'После выведения', 'input': 'Фактическая начальная орбита'},
        {'phase': 'После манёвра', 'input': 'Достигнутое состояние'},
    ],
    'possible_responses': [
        'Исправить отклонение',
        'Изменить срок',
        'Выбрать другую цель',
        'Нет допустимого продолжения',
    ],
    'cycle_stages': STAGES,
    'historical_cycle_source': '/home/vladimir/introduction-to-maneuvering-2025-2026/10_2.tex',
    'book': 'DiPrinzio, Methods of Orbital Maneuvering, 2019, chapter 15, especially 15.4',
    'akatsuki': [
        {'period': 'Декабрь 2010', 'event': 'После нештатной работы ДУ требуемая орбита Венеры не получена.'},
        {'period': 'Между попытками', 'event': 'Построена новая программа полёта и подготовлено использование двигателей ориентации.'},
        {'period': 'Декабрь 2015', 'event': 'Двигатели ориентации использованы для манёвра; выход на орбиту подтверждён измерениями.'},
    ],
}
SOURCES = {
    'accessed': '2026-11-17',
    'media_download': {
        'status': 'unavailable in this authoring environment',
        'reason': 'urllib DNS resolution failed; no escalation requested for optional photos',
        'local_media_files': [],
        'fallback': 'overview-akatsuki.pdf is an original conceptual historical diagram, not a photo',
    },
    'akatsuki': {
        'facts': [
            '2010 orbit insertion did not achieve the planned orbit',
            '2015 orbit insertion used attitude-control thrusters',
            'The achieved orbit was confirmed by measurement on 9 December 2015',
        ],
        'sources': [
            {'title': 'Venus Climate Orbiter AKATSUKI Venus Observation Orbit Injection result',
             'date': '2010-12-08',
             'url': 'https://www.jaxa.jp/press/2010/12/20101208_akatsuki_e.html'},
            {'title': 'Attitude-control thruster firing result',
             'date': '2015-12-07',
             'url': 'https://www.jaxa.jp/press/2015/12/20151207_akatsuki_j.html'},
            {'title': "AKATSUKI inserted into Venus' orbit",
             'date': '2015-12-09',
             'url': 'https://global.jaxa.jp/press/2015/12/20151209_akatsuki.html'},
            {'title': 'JAXA/ISAS status report, 2015 results',
             'url': 'https://jaxa.repo.nii.ac.jp/record/39625/files/SA6100002000.pdf'},
        ],
        'available_official_image': {
            'description': 'Official 2015 orbit-pattern diagram, not a spacecraft photograph',
            'image_url': 'https://global.jaxa.jp/press/2015/12/images/20151209_akatsuki_1.jpg',
            'source_page': 'https://global.jaxa.jp/press/2015/12/20151209_akatsuki.html',
            'credit': 'JAXA',
            'downloaded': False,
        },
        'available_spacecraft_images_page': {
            'url': 'https://akatsuki.isas.jaxa.jp/en/mission/spacecraft/',
            'credit': 'JAXA',
            'description': 'Official spacecraft/instrument views, no local download',
        },
    },
    'galileo': {
        'source_page': 'https://www.esa.int/Newsroom/Press_Releases/Galileo_satellite_recovered_and_transmitting_navigation_signals',
        'available_official_image': {
            'source_page': 'https://www.esa.int/ESA_Multimedia/Images/2014/11/Galileo_satellite_s_revised_orbit',
            'description': "Official revised-orbit illustration; not a spacecraft photograph",
            'credit': 'ESA',
            'licence': 'ESA Standard Licence',
            'downloaded': False,
            'date_caution': 'Image-page metadata says 03/11/2014, but the text describes later November events; use 03 December press release to date the historical facts.',
        },
        'facts': [
            'The fifth satellite underwent 11 maneuvers over 17 days in November 2014',
            'The revised orbit allowed continuous Earth-sensor operation and reduced radiation exposure',
        ],
    },
}


def save(fig, name):
    fig.savefig(BASE / (name + '.pdf'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(PREVIEW / (name + '.png'), dpi=180,
                bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def arrow(ax, begin, end, color=GRAY, width=1.7, style='-|>', **kw):
    patch = FancyArrowPatch(begin, end, arrowstyle=style, color=color,
                            linewidth=width, mutation_scale=13,
                            shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(patch)
    return patch


def planning():
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(-.1, 5.2)
    ax.axis('off')
    xs = [1.2, 5., 8.8]
    titles = ['До запуска', 'После выведения', 'После манёвра']
    details = ['Разбросы и\nрезервные сценарии',
               'Фактическая\nначальная орбита',
               'Достигнутое\nсостояние']
    colors = [BLUE, GREEN, PURPLE]
    arrow(ax, (.5, 3.75), (9.5, 3.75), color=GRAY, width=1.2)
    for x, title, detail, c in zip(xs, titles, details, colors):
        ax.plot(x, 3.75, 'o', ms=10, color=c, zorder=5)
        ax.text(x, 4.8, title, ha='center', va='center',
                color=c, fontsize=15.5, fontweight='bold')
        ax.text(x, 4.45, detail, ha='center', va='top',
                color=INK, fontsize=13.5, linespacing=1.25)
        ax.plot([x, x], [3.65, 3.3], color=c, lw=1.4)
    ax.plot([xs[0], xs[-1]], [3.3, 3.3], color=GRAY, lw=1.4)
    arrow(ax, (5., 3.3), (5., 2.95), color=INK)
    ax.text(5., 2.68, 'Пересчёт и проверка ограничений',
            ha='center', va='center', fontsize=16., color=INK)
    ax.plot([5., 5.], [2.4, 2.1], color=GRAY, lw=1.5)
    ax.plot([1.1, 8.9], [2.1, 2.1], color=GRAY, lw=1.5)
    outcome_x = [1.1, 3.7, 6.3, 8.9]
    labels = ['Исправить\nотклонение',
              'Изменить\nсрок',
              'Выбрать\nдругую цель',
              'Нет допустимого\nпродолжения']
    notes = ['Рабочее тело,\nнаправление манёвра',
             'Время перехода,\nдоступные окна',
             'Пригодная орбита,\nполезная работа',
             'Пересмотр\nпрограммы миссии']
    colors = [GREEN, ORANGE, BLUE, RED]
    for x, title, note, c in zip(outcome_x, labels, notes, colors):
        arrow(ax, (x, 2.1), (x, 1.7), color=c)
        ax.text(x, 1.5, title, ha='center', va='top',
                color=c, fontsize=14.8, fontweight='bold', linespacing=1.25)
        ax.text(x, .72, note, ha='center', va='top',
                fontsize=12.5, color=INK, linespacing=1.25)
    fig.subplots_adjust(left=.025, right=.975, bottom=.035, top=.98)
    save(fig, 'overview-planning')


def cycle():
    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    ax.set_xlim(0, 9.8)
    ax.set_ylim(-.1, 10.3)
    ax.axis('off')
    ys = [9.35, 7.35, 5.35, 3.35, 1.35]
    for j, (s, y) in enumerate(zip(STAGES, ys), 1):
        ax.add_patch(Circle((.5, y), .23, facecolor=s['color'],
                            edgecolor='white', lw=1.2, zorder=4))
        ax.text(.5, y, str(j), ha='center', va='center',
                color='white', fontsize=15., fontweight='bold', zorder=5)
        ax.text(1.1, y+.13, s['title'], va='center', ha='left',
                fontsize=16.5, color=s['color'], fontweight='bold')
        ax.text(1.1, y-.28, '\n'.join(s['details']), va='top', ha='left',
                fontsize=14.2, color=INK, linespacing=1.35)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (.5, y1-.3), (.5, y2+.34), color=GRAY, width=1.7)
    # Feedback path is outside the text column, with one short horizontal label.
    path_x = 9.15
    ax.plot([.5, .5, path_x, path_x], [1.05, .05, .05, 9.35],
            color=GREEN, lw=1.8)
    arrow(ax, (path_x, 9.35), (8.35, 9.35), color=GREEN, width=1.8)
    ax.text(9.45, 5., 'Новые данные и следующий цикл',
            rotation=90, va='center', ha='center', color=GREEN, fontsize=13.5)
    fig.subplots_adjust(left=.03, right=.98, bottom=.025, top=.985)
    save(fig, 'overview-cycle')


def akatsuki():
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(-.1, 10.2)
    ax.set_ylim(-.05, 4.9)
    ax.axis('off')
    xs = [.8, 5., 9.2]
    colors = [RED, ORANGE, GREEN]
    dates = ['Декабрь 2010', 'Между попытками', 'Декабрь 2015']
    events = [
        'Нештатная работа ДУ.\nТребуемая орбита Венеры\nне получена.',
        'Новая программа полёта.\nПодготовка альтернативного\nуправляющего воздействия.',
        'Манёвр двигателями\nориентации. Выход\nна орбиту подтверждён.',
    ]
    arrow(ax, (.4, 3.7), (9.65, 3.7), color=GRAY, width=1.4)
    for x, date, event, c in zip(xs, dates, events, colors):
        ax.plot(x, 3.7, 'o', color=c, ms=10, zorder=4)
        ax.text(x, 4.3, date, ha='center', va='center',
                color=c, fontweight='bold', fontsize=15.)
        ax.text(x, 3.38, event, ha='center', va='top', fontsize=12.8,
                linespacing=1.35, color=INK)
    ax.text(1.7, 1.56, 'Маршевый\nдвигатель', ha='center', va='center',
            color=RED, fontsize=15.)
    ax.plot([.64, 2.77], [1.01, 2.04], color=RED, lw=2., alpha=.7)
    arrow(ax, (3.15, 1.54), (4.5, 1.54), color=ORANGE, width=2.2)
    ax.text(6.8, 1.56, 'Двигатели ориентации\nи новая программа перехода',
            ha='center', va='center', color=GREEN, fontsize=15., linespacing=1.35)
    ax.text(5., .48, 'Сохранение миссии при изменившихся возможностях КА',
            ha='center', va='center', fontsize=14.5, color=INK)
    ax.text(5., .05, 'Историческая схема: интервалы времени изображены условно',
            ha='center', va='center', fontsize=11.7, color=GRAY)
    fig.subplots_adjust(left=.035, right=.965, bottom=.035, top=.97)
    save(fig, 'overview-akatsuki')


def main():
    mapped = sorted(n for s in STAGES for n in s['previous_stage_numbers'])
    assert mapped == list(range(1, 19)), mapped
    planning()
    cycle()
    akatsuki()
    (BASE / 'overview-diagrams-data.json').write_text(
        json.dumps(DATA, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (BASE / 'overview-case-sources.json').write_text(
        json.dumps(SOURCES, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
