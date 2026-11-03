"""Animate the recorded MP_full policy actions and four-leg controller states.

This is a data visualization, not a second simulation. The input NPZ must come
from render_mp_full_showcase.py so that every quantity shares one rollout clock.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / 'docs/terrain/mp_full_group_showcase_max_2026-10-28'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
W, H = 1600, 900
BG = '#0a1120'
PANEL = '#121e30'
PANEL_INNER = '#1a2a40'
TEXT = '#f2f6fb'
MUTED = '#a9b9cc'
GRID = '#37506a'
ACTION = '#f4c76c'
STATE = '#7fd2e9'
COLORS = ('#54b7ff', '#ff9c6b', '#8bd078', '#d99aff')
LEGS = ('FL', 'FR', 'RL', 'RR')
GAITS = ('walk', 'trot', 'canter', 'bound')


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


FONTS = {n: font(n) for n in (15, 17, 18, 20, 22, 24, 28, 34)}


def text(draw, xy, value, size=20, fill=TEXT, anchor=None):
    draw.text(xy, str(value), font=FONTS[size], fill=fill, anchor=anchor)


def panel(draw, box, radius=16, fill=PANEL):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def arrow(draw, x0, x1, y, color):
    draw.line((x0, y, x1 - 9, y), fill=color, width=3)
    draw.polygon(((x1, y), (x1 - 11, y - 6), (x1 - 11, y + 6)), fill=color)


def bipolar_bar(draw, box, value, color):
    x0, y0, x1, y1 = box
    center = (x0 + x1) / 2
    panel(draw, box, radius=5, fill=PANEL_INNER)
    draw.line((center, y0 - 2, center, y1 + 2), fill=GRID, width=2)
    xvalue = center + float(np.clip(value, -1, 1)) * (x1 - x0) / 2
    if abs(xvalue - center) > 1:
        draw.rounded_rectangle((min(center, xvalue), y0, max(center, xvalue), y1),
                               radius=4, fill=color)


def state_bar(draw, box, value, color):
    x0, y0, x1, y1 = box
    panel(draw, box, radius=5, fill=PANEL_INNER)
    end = x0 + float(np.clip(value, 0, 1)) * (x1 - x0)
    if end > x0 + 1:
        draw.rounded_rectangle((x0, y0, end, y1), radius=4, fill=color)


def validate_trace(trace):
    expected = {
        'time_s': (1,), 'base_pos_m': (2,), 'action': (2,),
        'gait_id': (1,), 'amplitude_x_state': (2,),
        'amplitude_z_state': (2,), 'phase_rad': (2,),
        'frequency_hz': (2,), 'frequency_scale': (2,),
        'duty_factor': (2,), 'offset_m': (3,),
        'offset_target_m': (3,), 'foot_contact': (2,),
        'foot_target_m': (3,),
    }
    n = len(trace['time_s'])
    for key, (ndim,) in expected.items():
        array = trace[key]
        if array.ndim != ndim or array.shape[0] != n:
            raise ValueError(f'{key} has unexpected shape {array.shape}')
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f'{key} contains nonfinite values')
    if trace['action'].shape[1] != 19 or trace['phase_rad'].shape[1] != 4:
        raise ValueError('Expected 19 actions and four leg phases')
    if not np.all(np.diff(trace['time_s']) > 0):
        raise ValueError('time_s must increase strictly')


def stage_name(x):
    if x < 2.4:
        return '平地'
    if x < 3.4:
        return '上坡'
    if x < 4.4:
        return '坡顶'
    if x < 5.4:
        return '下坡'
    if x < 7.2:
        return '接近 hurdle'
    if x < 7.4:
        return '跨越 hurdle'
    if x < 10.5:
        return '接近 gap'
    if x < 10.9:
        return '跨越 gap'
    return 'gap 后着陆'


def draw_course(draw, x, t):
    x0, x1 = 110, 1490
    y0, y1 = 222, 237
    panel(draw, (x0, y0, x1, y1), radius=7, fill=PANEL_INNER)
    for a, b, color in ((2.4, 3.4, '#456a9c'), (4.4, 5.4, '#456a9c'),
                        (7.2, 7.4, '#e3a455'), (10.5, 10.9, '#e16b86')):
        xa = x0 + int((a / 12.5) * (x1 - x0))
        xb = x0 + int((b / 12.5) * (x1 - x0))
        panel(draw, (xa, y0, xb, y1), radius=2, fill=color)
    px = x0 + int(np.clip(x / 12.5, 0, 1) * (x1 - x0))
    draw.ellipse((px - 10, y0 - 6, px + 10, y1 + 6), fill=TEXT)
    text(draw, (54, 220), '0 m', 17, MUTED, 'lm')
    text(draw, (1545, 220), '12.5 m', 17, MUTED, 'rm')
    for pos, label in ((2.9, 'ramp'), (7.3, 'hurdle'), (10.7, 'gap')):
        lx = x0 + int(pos / 12.5 * (x1 - x0))
        text(draw, (lx, 251), label, 17, MUTED, 'mt')
    text(draw, (1495, 186), f'{stage_name(x)}  ·  x={x:.2f} m  ·  t={t:.2f} s',
         20, TEXT, 'ra')


def draw_shared(draw, trace, i):
    action = trace['action'][i]
    gait = int(trace['gait_id'][i])
    boxes = ((55, 110, 382, 178), (402, 110, 732, 178),
             (752, 110, 1082, 178), (1102, 110, 1545, 178))
    for box in boxes:
        panel(draw, box, radius=12, fill=PANEL_INNER)
    labels = ('步态选择 · a0', '共享频率 · a17', '支撑占空比 · a18', '四肢瞬时频率')
    for box, label in zip(boxes, labels):
        text(draw, (box[0] + 16, 118), label, 18, MUTED)
    text(draw, (71, 143), f'{gait}: {GAITS[gait]}', 24)
    text(draw, (418, 143), f'{action[17]:+.2f}  →  {trace["frequency_scale"][i,0]:.2f}×', 24)
    text(draw, (768, 143), f'{action[18]:+.2f}  →  {trace["duty_factor"][i,0]:.2f}', 24)
    frequencies = trace['frequency_hz'][i]
    text(draw, (1118, 143), f'{frequencies.min():.1f}–{frequencies.max():.1f} Hz', 24)


def foot_plot(draw, trace, i, leg, row_y, color):
    x0, y0, x1, y1 = 1172, row_y + 18, 1516, row_y + 111
    panel(draw, (x0, y0, x1, y1), radius=8, fill=PANEL_INNER)
    mid = x0 + (x1 - x0) / 2
    draw.line((mid, y0 + 5, mid, y1 - 5), fill=GRID, width=1)
    zline = y0 + ((-.20 - -.05) / (-.35 - -.05)) * (y1 - y0)
    draw.line((x0 + 5, zline, x1 - 5, zline), fill=GRID, width=1)

    def project(point):
        px = x0 + (np.clip(float(point[0]), -.20, .20) + .20) / .40 * (x1 - x0)
        py = y0 + (np.clip(float(point[2]), -.35, -.05) + .05) / -.30 * (y1 - y0)
        return round(px), round(py)

    times = trace['time_s']
    start = int(np.searchsorted(times, times[i] - .28))
    points = [project(p) for p in trace['foot_target_m'][start:i + 1, leg]]
    if len(points) >= 2:
        draw.line(points, fill=color, width=3, joint='curve')
    if i > 0:
        px, py = points[-1]
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=TEXT,
                     outline=color, width=2)
    text(draw, (x0 + 7, y1 - 2), '后 ← X → 前', 15, MUTED, 'lb')
    text(draw, (x1 - 8, y0 + 3), 'Z ↑', 15, MUTED, 'ra')


def draw_leg(draw, trace, i, leg, row_y):
    color = COLORS[leg]
    panel(draw, (50, row_y, 1550, row_y + 127))
    panel(draw, (65, row_y + 15, 126, row_y + 51), radius=9, fill=color)
    text(draw, (96, row_y + 32), LEGS[leg], 20, BG, 'mm')
    contact = bool(trace['foot_contact'][i, leg])
    text(draw, (68, row_y + 67), '接触' if contact else '摆动', 18,
         '#9be09d' if contact else '#e9b48c')
    actions = trace['action'][i]
    for j, (name, index) in enumerate((('水平幅值', 1 + leg),
                                      ('抬腿幅值', 5 + leg),
                                      ('X 平衡', 9 + leg),
                                      ('Z 平衡', 13 + leg))):
        y = row_y + 11 + 28 * j
        text(draw, (151, y - 1), name, 17, MUTED)
        bipolar_bar(draw, (259, y + 3, 500, y + 18), actions[index], ACTION)
        text(draw, (510, y - 2), f'{actions[index]:+.2f}', 17, TEXT)
    for j, (name, value) in enumerate((('rx', trace['amplitude_x_state'][i, leg]),
                                      ('rz', trace['amplitude_z_state'][i, leg]))):
        y = row_y + 28 + 43 * j
        text(draw, (625, y - 7), name, 20, MUTED)
        state_bar(draw, (674, y, 831, y + 16), value, STATE)
        text(draw, (840, y - 7), f'{value:.2f}', 20)
    phase = float(trace['phase_rad'][i, leg])
    cx, cy, radius = 954, row_y + 64, 37
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 outline=GRID, width=3)
    draw.line((cx, cy, cx + radius * math.cos(phase - math.pi / 2),
               cy + radius * math.sin(phase - math.pi / 2)), fill=color, width=4)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=TEXT)
    text(draw, (954, row_y + 115), f'φ={phase / math.pi:.2f}π', 17, MUTED, 'mb')
    ox = 1008
    target = trace['offset_target_m'][i, leg]
    offset = trace['offset_m'][i, leg]
    text(draw, (ox, row_y + 18), '偏移 [cm]: 现←目标', 17, MUTED)
    text(draw, (ox, row_y + 51), f'X {offset[0]*100:+.1f}  ← {target[0]*100:+.1f}', 17)
    text(draw, (ox, row_y + 80), f'Z {offset[2]*100:+.1f}  ← {target[2]*100:+.1f}', 17)
    foot_plot(draw, trace, i, leg, row_y, color)


def make_frame(trace, i):
    image = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (51, 22), 'Action 如何调制四肢内部动力学', 34)
    text(draw, (1548, 36), 'A1 MP_full · 最高参数地形 · 同一次真实回放',
         20, MUTED, 'ra')
    panel(draw, (40, 97, 1560, 273))
    draw_shared(draw, trace, i)
    draw_course(draw, float(trace['base_pos_m'][i, 0]), float(trace['time_s'][i]))
    text(draw, (175, 288), 'RL action · 归一化目标  [−1, 1]', 22, ACTION)
    arrow(draw, 564, 610, 303, ACTION)
    text(draw, (703, 288), '内部状态 · 滤波、耦合、接触反馈', 22, STATE)
    arrow(draw, 1106, 1152, 303, STATE)
    text(draw, (1207, 288), '局部足端目标 · 送入 IK 前', 22, TEXT)
    for leg, row_y in enumerate((324, 458, 592, 726)):
        draw_leg(draw, trace, i, leg, row_y)
    text(draw, (51, 882),
         'a0 选相位模板；a17/a18 共享频率与占空比。四肢相位 φ 由振子耦合和接触反馈演化，并非四个独立相位 action。',
         18, MUTED, 'lb')
    return image


def run_ffmpeg(command, *, input_bytes=None):
    process = subprocess.run(command, input=input_bytes, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors='replace')[-3000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_DIR / 'rollout_trace.npz')
    parser.add_argument('--output', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--poster-only', action='store_true')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as archive:
        trace = {key: archive[key] for key in archive.files}
    validate_trace(trace)
    poster_i = int(np.argmin(np.abs(trace['time_s'] - 7.6)))
    poster = args.output / 'action_modulation_poster.png'
    make_frame(trace, poster_i).save(poster)
    if args.poster_only:
        print(poster)
        return

    # Resample onto the requested presentation frame clock without editing the
    # rollout or mixing steps from different episodes.
    frame_times = np.arange(0, float(trace['time_s'][-1]) + 1e-6, 1 / args.fps)
    indices = np.searchsorted(trace['time_s'], frame_times, side='left')
    indices = np.clip(indices, 0, len(trace['time_s']) - 1).tolist()
    video = args.output / 'action_modulation_four_legs.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W}x{H}', '-r', str(args.fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for i in indices:
            process.stdin.write(make_frame(trace, i).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        if process.wait() != 0:
            raise RuntimeError(error[-3000:])
    except Exception:
        process.kill()
        process.wait()
        raise

    gif = args.output / 'action_modulation_four_legs.gif'
    filter_graph = ('[0:v]fps=15,scale=1600:-1:flags=lanczos,split[a][b];'
                    '[a]palettegen=stats_mode=full[p];'
                    '[b][p]paletteuse=dither=bayer:bayer_scale=5')
    run_ffmpeg(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
                '-filter_complex', filter_graph, '-loop', '0', str(gif)])
    print(f'poster: {poster}\nvideo: {video}\ngif: {gif}\nframes: {len(indices)}')


if __name__ == '__main__':
    main()
