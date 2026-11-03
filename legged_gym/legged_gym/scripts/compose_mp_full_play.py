"""Combine one MP_full camera recording with its synchronized dynamics trace."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / 'docs/terrain/mp_full_play_showcase_100fps_2026-11-02'
W, H = 2560, 1440
CAM_W, CAM_H = 1600, 900
BG = '#08121f'
PANEL = '#111f31'
INNER = '#1a2a40'
TEXT = '#f2f7fc'
MUTED = '#aac0d3'
GRID = '#365069'
BLUE = '#62c4ee'
TERRAIN = '#7ce0bd'
LEG_COLORS = ('#63bdff', '#ffac77', '#98da92', '#daa1f5')
LEG_NAMES = ('FL', 'FR', 'RL', 'RR')
GAIT_NAMES = ('walk', 'trot', 'canter', 'bound')
STAGE_NAMES = {
    'flat': '平地', 'rough_slope': '粗糙坡', 'ramp': '坡道',
    'gap_continuous': '连续沟', 'gap_single': '单沟', 'hurdle': '栏架',
    'stairs': '台阶', 'pit': '坑',
}
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT = {size: ImageFont.truetype(FONT_PATH, size) for size in
        (14, 16, 17, 18, 20, 22, 24, 28, 32, 36, 42)}


def label(draw, xy, value, size=20, fill=TEXT, anchor=None):
    draw.text(xy, str(value), font=FONT[size], fill=fill, anchor=anchor)


def panel(draw, box, *, fill=PANEL, radius=14, outline=None):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def sparkline(draw, series, index, box, bounds, color, window=150):
    x0, y0, x1, y1 = box
    panel(draw, box, fill=INNER, radius=5)
    lo, hi = bounds
    if lo < 0 < hi:
        zero = y1 - (0 - lo) / (hi - lo) * (y1 - y0)
        draw.line((x0 + 2, zero, x1 - 2, zero), fill=GRID, width=1)
    first = max(0, index - window)
    picks = np.unique(np.linspace(first, index, min(45, index - first + 1)).astype(int))
    if len(picks) < 2:
        return
    values = np.clip(series[picks], lo, hi)
    points = [(round(x0 + 3 + (j / (len(picks) - 1)) * (x1 - x0 - 6)),
               round(y1 - 3 - (float(value) - lo) / (hi - lo) * (y1 - y0 - 6)))
              for j, value in enumerate(values)]
    draw.line(points, fill=color, width=2, joint='curve')
    x, y = points[-1]
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_static(trace, metadata, polar=False):
    image = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(image)
    label(draw, (40, 18), 'A1 · 连续地形越障回放', 36)
    fps_note = f"  ·  {metadata['video_fps']:.0f} fps" if polar else ''
    label(draw, (2520, 26), Path(metadata['checkpoint']).name + '  ·  实际推理轨迹' + fps_note,
          20, MUTED, 'ra')
    panel(draw, (29, 89, 1651, 1011), outline=GRID)
    panel(draw, (1672, 89, 2531, 1011), outline=GRID)
    panel(draw, (29, 1030, 2531, 1415), outline=GRID)

    if polar:
        label(draw, (1698, 212), '四足振子相位 · 状态空间示意', 22)
        label(draw, (1698, 240), '中心=局部平衡偏移  ·  椭圆=X/Z振幅  ·  光点=相位  ·  尾迹=最近0.12秒', 16, MUTED)
        for leg in range(4):
            col, row = leg % 2, leg // 2
            left, top = 1692 + col * 413, 270 + row * 217
            panel(draw, (left, top, left + 405, top + 207), fill='#14243a', radius=9)
            cx, cy = left + 103, top + 103
            draw.line((cx - 89, cy, cx + 89, cy), fill=GRID, width=1)
            draw.line((cx, cy - 88, cx, cy + 88), fill=GRID, width=1)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=MUTED)
            label(draw, (cx + 82, cy + 3), 'X', 14, MUTED, 'mm')
            label(draw, (cx + 3, cy - 82), 'Z', 14, MUTED, 'mm')

    label(draw, (1698, 719), '机体系六维速度 · 最近 3 秒', 22)
    label(draw, (1698, 745), 'v: m/s   ·   ω: rad/s', 16, MUTED)
    for row in range(2):
        for col in range(3):
            x = 1694 + col * 273
            y = 774 + row * 110
            panel(draw, (x, y, x + 259, y + 101), fill='#14243a', radius=9)

    label(draw, (54, 1045), '地形与机体高度 · 同一次回放', 24)
    label(draw, (2510, 1051), '绿色：地形剖面    蓝色：机体轨迹    白线：当前位置',
          17, MUTED, 'ra')
    rows = metadata['terrain_names']
    length = float(metadata['terrain_length_m'])
    total = len(rows) * length
    bar_x0, bar_x1 = 89, 2495
    bar_y0, bar_y1 = 1092, 1144
    for row, name in enumerate(rows):
        x0 = round(bar_x0 + row / len(rows) * (bar_x1 - bar_x0))
        x1 = round(bar_x0 + (row + 1) / len(rows) * (bar_x1 - bar_x0))
        fill = '#263f58' if row % 2 == 0 else '#314961'
        draw.rectangle((x0, bar_y0, x1 - 2, bar_y1), fill=fill)
        label(draw, ((x0 + x1) / 2, 1101), f'{row + 1}. {STAGE_NAMES.get(name, name)}',
              20, TEXT, 'mt')
        label(draw, ((x0 + x1) / 2, 1126), f'{row * length:.0f}–{(row+1)*length:.0f}m',
              16, MUTED, 'mt')

    plot = (89, 1170, 2495, 1384)
    x0, y0, x1, y1 = plot
    panel(draw, plot, fill='#0d1b2c', radius=9)
    heights = trace['terrain_height_m']
    base_height = trace['base_pos_m'][:, 2]
    low = float(heights.min()) - .1
    high = float(max(heights.max(), base_height.max())) + .1

    def xy(x, h):
        return (round(x0 + (float(x) / total) * (x1 - x0)),
                round(y1 - (float(h) - low) / (high - low) * (y1 - y0)))

    for row in range(len(rows) + 1):
        xx = xy(row * length, low)[0]
        draw.line((xx, y0, xx, y1), fill=GRID, width=1)
    for value in np.linspace(low, high, 5):
        yy = xy(0, value)[1]
        draw.line((x0, yy, x1, yy), fill='#263a50', width=1)
        label(draw, (52, yy), f'{value:+.1f}', 14, MUTED, 'lm')
    profile = [xy(x, h) for x, h in zip(trace['terrain_x_m'], heights)]
    draw.line(profile, fill=TERRAIN, width=3, joint='curve')
    return image, xy, (bar_x0, bar_x1, bar_y0, bar_y1), total


def draw_polar_dynamics(draw, trace, metadata, index):
    """Show oscillator state, not a literal foot trajectory."""
    eq = trace['equilibrium_m'] * 100.0
    ax = trace['amplitude_x']
    az = trace['amplitude_z']
    phase = trace['phase_rad']
    frequency = trace['frequency_hz']
    lookback = max(1, round(.12 * metadata['trace_rate_hz']))

    def state_point(sample, leg, cx, cy):
        ox = float(eq[sample, leg, 0]) * 4.0
        oz = float(eq[sample, leg, 1]) * 4.0
        rx = 15.0 + 42.0 * float(ax[sample, leg])
        rz = 15.0 + 42.0 * float(az[sample, leg])
        theta = float(phase[sample, leg])
        return (cx + ox + rx * np.cos(theta), cy - oz - rz * np.sin(theta))

    for leg in range(4):
        col, row = leg % 2, leg // 2
        left, top = 1692 + col * 413, 270 + row * 217
        cx, cy = left + 103, top + 103
        ox = float(eq[index, leg, 0]) * 4.0
        oz = float(eq[index, leg, 1]) * 4.0
        ex, ey = cx + ox, cy - oz
        rx = 15.0 + 42.0 * float(ax[index, leg])
        rz = 15.0 + 42.0 * float(az[index, leg])
        color = LEG_COLORS[leg]
        draw.line((cx, cy, ex, ey), fill='#e9d292', width=2)
        draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill='#e9d292')
        ellipse = [(ex + rx * np.cos(a), ey - rz * np.sin(a))
                   for a in np.linspace(0., 2 * np.pi, 65)]
        draw.line(ellipse, fill=color, width=2, joint='curve')
        first = max(0, index - lookback)
        picks = list(range(first, index + 1))
        if len(picks) >= 2:
            base_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            for j in range(1, len(picks)):
                frac = .25 + .6 * j / (len(picks) - 1)
                faded = tuple(int(20 * (1-frac) + c * frac) for c in base_rgb)
                draw.line((*state_point(picks[j-1], leg, cx, cy),
                           *state_point(picks[j], leg, cx, cy)),
                          fill=faded, width=3)
        px, py = state_point(index, leg, cx, cy)
        draw.line((ex, ey, px, py), fill=color, width=2)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=TEXT, outline=color, width=3)
        tx = left + 209
        label(draw, (tx, top + 12), LEG_NAMES[leg], 28, color)
        phase_deg = np.degrees(float(phase[index, leg])) % 360.0
        label(draw, (left + 389, top + 21), f'θ {phase_deg:.0f}°', 16, MUTED, 'ra')
        label(draw, (tx, top + 54), f'f = {float(frequency[index,leg]):.1f} Hz', 20)
        label(draw, (tx, top + 87), f'平衡 X {float(eq[index,leg,0]):+.1f} cm', 17, MUTED)
        label(draw, (tx, top + 111), f'平衡 Z {float(eq[index,leg,1]):+.1f} cm', 17, MUTED)
        label(draw, (tx, top + 145), f'振幅 X {float(ax[index,leg]):.2f}', 17)
        label(draw, (tx, top + 169), f'振幅 Z {float(az[index,leg]):.2f}', 17)


def draw_dynamic(base, camera, trace, metadata, index, xy, bar, total, polar=False):
    image = base.copy()
    image.paste(camera, (40, 100))
    draw = ImageDraw.Draw(image)
    t = float(trace['time_s'][index])
    pos = trace['base_pos_m'][index]
    x = float(pos[0])
    row = min(max(int(x / metadata['terrain_length_m']), 0), len(metadata['terrain_names']) - 1)
    stage = STAGE_NAMES.get(metadata['terrain_names'][row], metadata['terrain_names'][row])
    gait = int(trace['gait_id'][index])

    panel(draw, (59, 115, 595, 203), fill='#15263b', radius=10)
    label(draw, (75, 125), f'{stage}  ·  第 {row + 1}/{len(metadata["terrain_names"])} 段', 28)
    label(draw, (75, 164), f't = {t:05.2f} s    x = {x:05.2f} m', 22, MUTED)
    panel(draw, (1689, 104, 2511, 203), fill='#14243a', radius=10)
    label(draw, (1708, 113), '当前步态选择', 18, MUTED)
    for j, name in enumerate(GAIT_NAMES):
        bx0 = 1845 + j * 160
        color = LEG_COLORS[j] if j == gait else '#263b54'
        panel(draw, (bx0, 115, bx0 + 145, 157), fill=color, radius=8)
        label(draw, (bx0 + 72, 136), name, 18, BG if j == gait else MUTED, 'mm')
    label(draw, (1708, 164), f'目标前进 {float(trace["command_mps"][index,0]):.1f} m/s', 20)
    label(draw, (2495, 166), '四足振子 · FL / FR / RL / RR', 17, MUTED, 'ra')

    if polar:
        draw_polar_dynamics(draw, trace, metadata, index)
    else:
        eq_cm = trace['equilibrium_m'] * 100
        dynamics = (eq_cm[:, :, 0], eq_cm[:, :, 1], trace['frequency_hz'],
                    trace['amplitude_x'], trace['amplitude_z'])
        ranges = ((-8., 8.), (-6., 6.), (0., 16.), (0., 1.1), (0., 1.1))
        formats = ('{:+.1f}', '{:+.1f}', '{:.1f}', '{:.2f}', '{:.2f}')
        for leg in range(4):
            y = 253 + leg * 112
            for j, values in enumerate(dynamics):
                bx = 1758 + j * 149
                sparkline(draw, values[:, leg], index, (bx, y + 29, bx + 139, y + 72),
                          ranges[j], LEG_COLORS[leg])
                label(draw, (bx + 69, y + 78), formats[j].format(float(values[index, leg])),
                      17, TEXT, 'mt')

    velocities = np.concatenate((trace['body_lin_vel_mps'],
                                 trace['body_ang_vel_radps']), axis=1)
    names = ('vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz')
    bounds = ((-.6, 4.), (-1., 1.), (-1.5, 1.5),
              (-7., 7.), (-7., 7.), (-3., 3.))
    for j in range(6):
        col, row_v = j % 3, j // 3
        bx = 1694 + col * 273
        by = 774 + row_v * 110
        label(draw, (bx + 13, by + 5), names[j], 18, MUTED)
        label(draw, (bx + 246, by + 5), f'{velocities[index,j]:+.2f}', 20, TEXT, 'ra')
        sparkline(draw, velocities[:, j], index, (bx + 12, by + 39, bx + 246, by + 87),
                  bounds[j], BLUE if j < 3 else '#f1b587',
                  window=max(1, round(3 * metadata.get('trace_rate_hz', 50))))

    bar_x0, bar_x1, bar_y0, bar_y1 = bar
    cursor = round(bar_x0 + np.clip(x / total, 0, 1) * (bar_x1 - bar_x0))
    draw.line((cursor, bar_y0 - 5, cursor, bar_y1 + 5), fill=TEXT, width=4)
    draw.ellipse((cursor - 6, bar_y0 - 8, cursor + 6, bar_y0 + 4), fill=TEXT)
    _, plot_y = xy(x, pos[2])
    draw.line((cursor, 1170, cursor, 1384), fill='#dceefb', width=2)
    stride = max(1, index // 280)
    history = [xy(px, pz) for px, pz in trace['base_pos_m'][:index+1:stride][:, [0, 2]]]
    if len(history) > 1:
        draw.line(history, fill=BLUE, width=3, joint='curve')
    draw.ellipse((cursor - 7, plot_y - 7, cursor + 7, plot_y + 7),
                 fill=TEXT, outline=BLUE, width=3)
    return image


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError(f'Camera video ended with {remaining} bytes still needed')
        chunks.append(part)
        remaining -= len(part)
    return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--limit-frames', type=int, default=None)
    parser.add_argument('--polar', action='store_true', help='Use four oscillator phase diagrams')
    args = parser.parse_args()
    source = args.input.resolve()
    destination = (args.output or source).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / 'metadata.json').read_text())
    with np.load(source / 'rollout_trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    steps = trace['camera_steps'].astype(int)
    if len(steps) != metadata['video_frames'] or not np.all(np.diff(steps) > 0):
        raise ValueError('Camera frame indexes do not match recording metadata')
    for key in ('base_pos_m', 'body_lin_vel_mps', 'body_ang_vel_radps',
                'equilibrium_m', 'frequency_hz', 'amplitude_x', 'amplitude_z'):
        if not np.isfinite(trace[key]).all():
            raise ValueError(f'{key} contains nonfinite values')
    if trace['equilibrium_m'].shape[1:] != (4, 2):
        raise ValueError('Expected four X/Z equilibrium trajectories')
    if args.polar and ('phase_rad' not in trace or trace['phase_rad'].shape[1:] != (4,)):
        raise ValueError('Polar view needs four recorded oscillator phases')
    if metadata['camera_resolution'] != [CAM_W, CAM_H]:
        raise ValueError('This dashboard expects 1600×900 camera frames')

    static, xy, bar, total = draw_static(trace, metadata, polar=args.polar)
    n_frames = len(steps) if args.limit_frames is None else min(len(steps), args.limit_frames)
    decoder = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source / 'camera_raw.mp4'),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_video = destination / ('mp_full_play_polar_1440p_100fps.mp4' if args.polar
                                  else 'mp_full_play_showcase_1440p.mp4')
    encoder = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(round(metadata['video_fps'])),
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_video)
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    poster_target = (metadata['terrain_names'].index('hurdle') + .5) * metadata['terrain_length_m'] \
        if 'hurdle' in metadata['terrain_names'] else total / 2
    poster_frame = int(np.argmin(np.abs(trace['base_pos_m'][steps, 0] - poster_target)))
    if poster_frame >= n_frames:
        poster_frame = n_frames - 1
    try:
        for frame_number, step in enumerate(steps[:n_frames]):
            camera = Image.frombytes('RGB', (CAM_W, CAM_H),
                                     read_exact(decoder.stdout, CAM_W * CAM_H * 3))
            frame = draw_dynamic(static, camera, trace, metadata, step, xy, bar, total,
                                 polar=args.polar)
            if frame_number == poster_frame:
                frame.save(destination / 'showcase_poster.png')
            encoder.stdin.write(frame.tobytes())
            if frame_number % 100 == 0:
                print(f'Composed {frame_number + 1}/{n_frames} frames', flush=True)
        encoder.stdin.close()
        encode_error = encoder.stderr.read().decode(errors='replace')
        if encoder.wait() != 0:
            raise RuntimeError(encode_error[-3000:])
        decoder.stdout.close()
        decoder.wait()
    except Exception:
        encoder.kill()
        decoder.kill()
        encoder.wait()
        decoder.wait()
        raise
    print(output_video)


if __name__ == '__main__':
    main()
