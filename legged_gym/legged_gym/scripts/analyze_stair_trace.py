"""Summarize saved stair traces without loading Isaac Gym or a policy."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory):
    report = json.loads((directory / 'report.json').read_text())
    trace = np.load(directory / 'trace.npz')
    cfg = report['effective_config']['terrain']
    distance = cfg['terrain_length'] / 2 - cfg['boundary_margin']
    flat_edge = next((row['first_riser_x_m'] for row in report['rows']
                      if row['first_riser_x_m'] is not None), 1.5)
    rows = []
    for row in report['rows']:
        episodes = []
        for episode in row['episodes']:
            i = episode['env']
            terminal = np.flatnonzero(trace['failed'][:, i] | trace['timeout'][:, i])
            end = int(terminal[0] + 1) if len(terminal) else len(trace['root'])
            root = trace['root'][:end, i]
            origin = trace['origins'][i]
            forward, lateral = root[:, 0] - origin[0], root[:, 1] - origin[1]
            crossings = np.flatnonzero(forward >= distance)
            first_cross = int(crossings[0]) if len(crossings) else None
            passed = (first_cross is not None
                      and not trace['failed'][:first_cross + 1, i].any()
                      and np.abs(lateral[:first_cross + 1]).max() <= 1.5)
            edge = row['first_riser_x_m']
            # Flat-ground comparison uses the same spatial window.
            edge = flat_edge if edge is None else edge
            near = (forward >= edge - .5) & (forward <= edge + .8)
            phase = trace['vertical_phase'][:end, i]
            contact = trace['force'][:end, i, :, 2] > 5.
            stance = phase >= np.pi
            late = (phase >= 1.5 * np.pi) & (phase <= 1.95 * np.pi)
            def missing_fraction(mask):
                mask = mask & near[:, None]
                return float((mask & ~contact).sum() / mask.sum()) if mask.any() else None
            qx, qy, qz, qw = root[:, 3:7].T
            pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
            roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx*qx + qy*qy))
            e = dict(episode, strict_success=bool(passed),
                     max_lateral_m=float(np.abs(lateral).max()),
                     max_pitch_deg=float(np.rad2deg(np.abs(pitch)).max()),
                     max_roll_deg=float(np.rad2deg(np.abs(roll)).max()),
                     near_riser_samples=int(near.sum()),
                     stance_missing_contact_fraction=missing_fraction(stance),
                     late_stance_missing_contact_fraction=missing_fraction(late))
            for key in ('frequency_hz', 'duty_factor', 'q_late', 'stance_extension',
                        'mean_joint_error', 'reference_speed_excess', 'saturation'):
                if key in trace and near.any():
                    e['near_riser_' + key] = float(trace[key][:end, i][near].mean())
            episodes.append(e)
        rows.append(dict(height_m=row['height_m'], episodes=episodes,
                         passed=sum(e['strict_success'] for e in episodes),
                         falls=sum(e['failed'] for e in episodes)))
    return dict(checkpoint=report['checkpoint'], rows=rows,
                notes='First episodes only. Success requires forward crossing before fall and lateral deviation <=1.5m. '
                      'Contact/phase statistics use policy-rate samples, not a complete substep contact history. '
                      'Near-riser window is body x in [first edge - .5, first edge + .8] metres; '
                      'missing nominal stance contact is descriptive and does not alone establish a cause.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result['rows']:
        print(f"{row['height_m']:.3f}m: {row['passed']}/{len(row['episodes'])} passed; {row['falls']} falls")


if __name__ == '__main__':
    main()
