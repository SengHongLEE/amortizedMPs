"""Configurable task terrain. Height arrays are indexed [world x, world y].

Built-in shapes use metres (slope is dz/dx), with lengths snapped to grid cells.
SDK overrides retain native quantization. No policy observations are added.
"""
from contextlib import contextmanager
from copy import deepcopy
import inspect
from numbers import Integral, Real

import numpy as np
from isaacgym import terrain_utils


# *_range is a scalar or [value_at_difficulty_0, value_at_difficulty_1].
# Rubble's stone_size_range is instead the random size interval of each block.
DEFAULT_PARAMS = {
    'flat': {'height': 0.},
    'slope': {'slope_range': [.0, .3], 'slope_signs': [-1, 1]},
    'ramp': {'slope_range': [.0, .3], 'slope_signs': [-1, 1], 'center_platform': False},
    'rough': {'height_range': [.005, .04], 'height_step': .005, 'downsampled_scale': .2},
    'rough_slope': {'slope_range': [.0, .2], 'slope_signs': [-1, 1],
                    'height_range': [.005, .025], 'height_step': .005, 'downsampled_scale': .2},
    'rubble': {'height_range': [.04, .12], 'stone_size_range': [1., 2.], 'num_rects': 20},
    'stairs': {'step_height_range': [.03, .12], 'step_width': .3, 'step_signs': [-1, 1],
               'mesh_vertical_faces': False},
    'stairs_linear': {'step_height_range': [.03, .12], 'step_width': .3, 'step_signs': [-1, 1],
                      'mesh_vertical_faces': False},
    'stepping_stones': {'stone_size_range': [.8, .3], 'stone_distance_range': [.1, .3],
                        'height_range': [.0, .05], 'depth_range': [.2, .6]},
    'wave': {'amplitude_range': [.02, .15], 'num_waves': 3},
    'gap': {'width_range': [.1, .5], 'depth_range': [.2, .6], 'spacing': 1.5,
            'pattern': 'mirrored', 'num_gaps': None, 'single_position': 'after_platform'},
    'pit': {'size_range': [2.5, 3.], 'depth_range': [.1, .5]},
    'hurdle': {'width_range': [.2, .3], 'height_range': [.05, .2], 'spacing': 1.5,
               'pattern': 'mirrored'},
    'beam': {'width_range': [1., .3], 'depth_range': [.2, .6]},
}
COMMON_PARAMS = {'axis': 'x', 'platform_size': None, 'height_offset': 0., 'roughness': None}
OFFICIAL_GENERATORS = {name: getattr(terrain_utils, name) for name in (
    'sloped_terrain', 'pyramid_sloped_terrain', 'random_uniform_terrain',
    'discrete_obstacles_terrain', 'stairs_terrain', 'pyramid_stairs_terrain',
    'stepping_stones_terrain', 'wave_terrain')}


@contextmanager
def _terrain_seed(seed):
    """SDK generators use global NumPy RNG; preserve the training RNG."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{label} must be within [{minimum}, {maximum}], got {value}')
    return float(value)


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return int(value)


def _pair(value, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f'{label} must contain two finite numbers')
    return [_number(v, label) for v in value]


def _resolve(value, difficulty, label):
    if isinstance(value, Real):
        return _number(value, label, 0)
    a, b = _pair(value, label)
    if min(a, b) < 0:
        raise ValueError(f'{label} must be nonnegative')
    return (1 - difficulty) * a + difficulty * b


def _rough(terrain, p):
    terrain_utils.random_uniform_terrain(
        terrain, min_height=-p['height_range'], max_height=p['height_range'],
        step=p['height_step'], downsampled_scale=p['downsampled_scale'])


def _center_slice(shape, size, scale):
    pixels = max(1, int(round(size / scale)))
    return tuple(slice(max(0, (n - pixels) // 2), min(n, (n - pixels) // 2 + pixels)) for n in shape)


def _sdk_length(value, scale):
    """Snap wrapper dimensions to cells, avoiding int(0.3 / 0.1) == 2 in SDK."""
    cells = round(value / scale)
    return np.nextafter(cells * scale, np.inf) if cells else 0.


def obstacle_bands(length, horizontal_scale, params, kind):
    """Return obstacle [start, end) intervals in local travel-axis cells."""
    width = int(round(params['width_range'] / horizontal_scale))
    count = (params.get('num_gaps') if kind == 'gap'
             and params['pattern'] in ('uniform', 'centered') else None)
    if count is not None:
        if params['pattern'] == 'centered':
            platform = int(round(params['platform_size'] / horizontal_scale))
            span = count * width + (count - 1) * platform
            if span > length:
                return []
            first = (length - span) // 2
            return [(first + index * (width + platform),
                     first + index * (width + platform) + width)
                    for index in range(count)]
        stride = length / count
        # Keep the tile centre on a landing segment for either parity.
        phase = stride / 2 if count % 2 == 0 else 1
        starts = [round(phase + index * stride) for index in range(count)]
        return [(start, start + width) for start in starts if start + width <= length]
    stride = int(round(params['spacing'] / horizontal_scale))
    if kind == 'hurdle' and params['pattern'] == 'single':
        start = (length - width) // 2
        return [(start, start + width)]
    if kind == 'gap' and params['pattern'] == 'single':
        if params['single_position'] == 'center':
            start = (length - width) // 2
        else:
            start = length // 2 + max(int(np.ceil(params['platform_size'] / horizontal_scale / 2)), stride)
        return [(start, min(length, start + width))] if start < length else []
    if kind in ('gap', 'hurdle') and params['pattern'] == 'uniform':
        return [(start, start + width) for start in range(stride // 2, length, stride)
                if start + width <= length]
    center = length // 2
    half = int(np.ceil(params['platform_size'] / horizontal_scale / 2))
    bands = []
    for offset in range(half, max(center, length - center), stride):
        bands.extend(((max(0, center - offset - width), max(0, center - offset)),
                      (center + offset, min(length, center + offset + width))))
    return sorted((a, b) for a, b in bands if a < b)


def _generate(terrain, name, p):
    """Generate in local x; the assembler transposes for a world-y axis."""
    scale = terrain.horizontal_scale
    platform = _sdk_length(p['platform_size'], scale)
    if name == 'flat':
        terrain.height_field_raw.fill(int(p['height'] / terrain.vertical_scale))
    elif name in ('slope', 'rough_slope', 'ramp'):
        grade = p['slope_range'] * p['slope_signs'][0]
        if name == 'ramp':
            if p['center_platform']:
                # Shift the ramps outwards: continuous transitions to a level landing,
                # unlike flattening an existing slope, which leaves artificial cliffs.
                x = (np.arange(terrain.width) - terrain.width // 2) * scale
                run = np.sign(x) * np.maximum(np.abs(x) - p['platform_size'] / 2, 0.)
                terrain.height_field_raw[:] = np.rint(
                    grade * run / terrain.vertical_scale).astype(np.int16)[:, None]
            else:
                terrain_utils.sloped_terrain(terrain, slope=grade)
        else:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=grade, platform_size=platform)
        if name == 'rough_slope':
            _rough(terrain, p)
    elif name == 'rough':
        _rough(terrain, p)
    elif name == 'rubble':
        terrain_utils.discrete_obstacles_terrain(
            terrain, max_height=p['height_range'], min_size=_sdk_length(p['stone_size_range'][0], scale),
            max_size=_sdk_length(p['stone_size_range'][1], scale), num_rects=p['num_rects'], platform_size=platform)
    elif name in ('stairs', 'stairs_linear'):
        options = dict(step_width=_sdk_length(p['step_width'], scale),
                       step_height=p['step_height_range'] * p['step_signs'][0])
        if name == 'stairs':
            terrain_utils.pyramid_stairs_terrain(terrain, platform_size=platform, **options)
        else:
            terrain_utils.stairs_terrain(terrain, **options)
            end = terrain.width // round(p['step_width'] / scale) * round(p['step_width'] / scale)
            # SDK leaves the final partial tread at zero, creating a false cliff.
            terrain.height_field_raw[end:, :] = terrain.height_field_raw[end-1, :]
    elif name == 'stepping_stones':
        terrain_utils.stepping_stones_terrain(
            terrain, stone_size=_sdk_length(p['stone_size_range'], scale),
            stone_distance=_sdk_length(p['stone_distance_range'], scale),
            max_height=p['height_range'], depth=-p['depth_range'], platform_size=platform)
    elif name == 'wave':
        terrain_utils.wave_terrain(terrain, num_waves=p['num_waves'], amplitude=p['amplitude_range'])
    elif name in ('gap', 'hurdle'):
        h, v = terrain.horizontal_scale, terrain.vertical_scale
        height = -p['depth_range'] if name == 'gap' else p['height_range']
        for start, end in obstacle_bands(terrain.width, h, p, name):
            terrain.height_field_raw[start:end, :] = int(height / v)
    elif name == 'pit':
        region = _center_slice(terrain.height_field_raw.shape, p['size_range'], terrain.horizontal_scale)
        terrain.height_field_raw[region] = -int(p['depth_range'] / terrain.vertical_scale)
    elif name == 'beam':
        terrain.height_field_raw.fill(-int(p['depth_range'] / terrain.vertical_scale))
        width = int(round(p['width_range'] / terrain.horizontal_scale))
        start = (terrain.length - width) // 2
        terrain.height_field_raw[:, start:start+width] = 0
        terrain.height_field_raw[_center_slice(terrain.height_field_raw.shape, platform, terrain.horizontal_scale)] = 0


def _registered_generator(name):
    def generate(terrain, difficulty, params, platform_size):
        options = dict(DEFAULT_PARAMS[name], **params)
        options.setdefault('platform_size', platform_size)
        for key, value in options.items():
            if key.endswith('_range') and not (name == 'rubble' and key == 'stone_size_range'):
                options[key] = _resolve(value, difficulty, f'{name}.{key}')
        _generate(terrain, name, options)
        return terrain
    return generate


# Preserve the generator(SubTerrain, difficulty, params, platform_size) interface.
TERRAIN_GENERATORS = {name: _registered_generator(name) for name in DEFAULT_PARAMS}


class BlindTerrain:
    def __init__(self, cfg, num_robots):
        self.cfg, self.num_robots, self.type = cfg, num_robots, cfg.mesh_type
        self._validate_config()
        self.horizontal_scale, self.vertical_scale = cfg.horizontal_scale, cfg.vertical_scale
        self.border = round(cfg.border_size / self.horizontal_scale)
        self.border_size = self.border * self.horizontal_scale
        self.length_per_env_pixels = round(cfg.terrain_length / self.horizontal_scale)
        self.width_per_env_pixels = round(cfg.terrain_width / self.horizontal_scale)
        self.env_length = self.length_per_env_pixels * self.horizontal_scale
        self.env_width = self.width_per_env_pixels * self.horizontal_scale
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border
        rows = self._rows()
        columns = None if rows is not None else self._columns()
        self.difficulties = self._difficulties()
        self.tile_parameters = np.empty((cfg.num_rows, cfg.num_cols), dtype=object)
        self.terrain_names = (np.array(rows, dtype=object)[:, None] if rows is not None
                              else np.tile(np.array(columns, dtype=object), (cfg.num_rows, 1)))
        # Resolve and validate all parameters before allocating the map or invoking SDK.
        occurrences = {}
        for col in range(cfg.num_cols):
            for row in range(cfg.num_rows):
                name = self.terrain_names[row, col]
                occurrence = occurrences.get(name, 0)
                if rows is not None or row == 0:
                    occurrences[name] = occurrence + 1
                self.tile_parameters[row, col] = self._parameters(name, self.difficulties[row, col], occurrence)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        stair_faces = []
        for col in range(cfg.num_cols):
            previous_ramp_edge = None
            previous_row_edge = None
            for row in range(cfg.num_rows):
                name = self.terrain_names[row, col]
                p = self.tile_parameters[row, col]
                transpose = p.get('axis', 'x') == 'y'
                shape = (self.width_per_env_pixels, self.length_per_env_pixels) if transpose else (
                    self.length_per_env_pixels, self.width_per_env_pixels)
                terrain = terrain_utils.SubTerrain(width=shape[0], length=shape[1],
                                                   horizontal_scale=self.horizontal_scale, vertical_scale=self.vertical_scale)
                clean_stairs = None
                with _terrain_seed((int(cfg.seed) + row * 1000003 + col * 9176) % 2**32):
                    if self.direct_options is not None:
                        OFFICIAL_GENERATORS[self.direct_options['type']](terrain, **p)
                    else:
                        _generate(terrain, self._kind(name), p)
                        if self.type == 'trimesh' and p.get('mesh_vertical_faces', False):
                            # Record the underlying risers before optional surface noise.
                            clean_stairs = terrain.height_field_raw.copy()
                        if p['roughness'] is not None:
                            _rough(terrain, p['roughness'])
                raw = terrain.height_field_raw.T if transpose else terrain.height_field_raw
                raw = raw.astype(np.int64) + int(p.get('height_offset', 0.) / self.vertical_scale)
                if getattr(cfg, 'spawn_platform', False):
                    region = _center_slice(raw.shape, p.get('platform_size', cfg.platform_size), self.horizontal_scale)
                    raw[region] = raw[raw.shape[0] // 2, raw.shape[1] // 2]
                if rows is None and self._kind(name) == 'ramp' and p['axis'] == 'x':
                    if previous_ramp_edge is not None:
                        raw += previous_ramp_edge - raw[0, raw.shape[1] // 2]
                    previous_ramp_edge = int(raw[-1, raw.shape[1] // 2])
                if rows is not None:
                    center = raw.shape[1] // 2
                    if previous_row_edge is not None:
                        raw += previous_row_edge - raw[0, center]
                    previous_row_edge = int(raw[-1, center])
                if raw.min() < -32768 or raw.max() > 32767:
                    raise ValueError(f'{name}: generated heights exceed int16; increase vertical_scale')
                if clean_stairs is not None:
                    clean_stairs = clean_stairs.T if transpose else clean_stairs
                    if getattr(cfg, 'spawn_platform', False):
                        clean_stairs[region] = clean_stairs[raw.shape[0] // 2, raw.shape[1] // 2]
                    stair_faces.append((row, col, self._riser_displacements(clean_stairs)))
                self._add_tile(raw, row, col)
        self.heightsamples = self.height_field_raw
        if self.type == 'trimesh':
            # SDK computes neighbour differences: widen to prevent int16 subtraction overflow.
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.heightsamples.astype(np.int32), self.horizontal_scale, self.vertical_scale,
                slope_threshold=cfg.slope_threshold)
            self._apply_stair_faces(stair_faces)

    def _riser_displacements(self, raw):
        """SDK grid-cell shifts for true risers, independent of global slope cutoff."""
        vertices, _ = terrain_utils.convert_heightfield_to_trimesh(
            raw.astype(np.int32), self.horizontal_scale, self.vertical_scale, slope_threshold=0.)
        xy = vertices.reshape(*raw.shape, 3)[..., :2] / self.horizontal_scale
        grid = np.stack(np.meshgrid(np.arange(raw.shape[0]), np.arange(raw.shape[1]),
                                    indexing='ij'), axis=-1)
        return np.rint(xy - grid).astype(np.int8)

    def _apply_stair_faces(self, patches):
        """Replace only riser XY coordinates; keep tread noise and other tiles intact."""
        vertices = self.vertices.reshape(self.tot_rows, self.tot_cols, 3)
        x = np.linspace(0, (self.tot_rows - 1) * self.horizontal_scale, self.tot_rows)
        y = np.linspace(0, (self.tot_cols - 1) * self.horizontal_scale, self.tot_cols)
        for row, col, shifts in patches:
            sx = self.border + row * self.length_per_env_pixels
            sy = self.border + col * self.width_per_env_pixels
            nx, ny = shifts.shape[:2]
            patch = vertices[sx:sx+nx, sy:sy+ny]
            for axis, coordinates in enumerate((x[sx:sx+nx, None], y[None, sy:sy+ny])):
                mask = shifts[..., axis] != 0
                target = coordinates + shifts[..., axis] * self.horizontal_scale
                patch[..., axis][mask] = target[mask]

    def _kind(self, name):
        return self.variants[name]['type'] if name in self.variants else name

    def _rows(self):
        explicit = getattr(self.cfg, 'terrain_rows', None)
        if explicit is None:
            return None
        known = set(DEFAULT_PARAMS) | set(self.variants)
        if (self.cfg.num_cols != 1 or self.cfg.selected is not None
                or getattr(self.cfg, 'terrain_columns', None) is not None
                or not isinstance(explicit, (list, tuple))
                or len(explicit) != self.cfg.num_rows
                or any(name not in known for name in explicit)):
            raise ValueError('terrain_rows requires one column and num_rows known terrain names')
        return list(explicit)

    def _columns(self):
        cfg = self.cfg
        known = set(DEFAULT_PARAMS) | set(self.variants)
        weights = cfg.terrain_proportions
        if not isinstance(weights, dict) or any(k not in known for k in weights):
            raise ValueError('terrain_proportions must name registered terrain types or variants')
        for name, weight in weights.items():
            _number(weight, f'terrain_proportions.{name}', 0)
        if cfg.selected is not None:
            if cfg.selected not in known:
                raise ValueError(f'Unknown selected terrain: {cfg.selected}')
            return [cfg.selected] * cfg.num_cols
        explicit = getattr(cfg, 'terrain_columns', None)
        if explicit is not None:
            if not isinstance(explicit, (list, tuple)) or len(explicit) != cfg.num_cols or any(n not in known for n in explicit):
                raise ValueError('terrain_columns must contain num_cols known terrain names')
            return list(explicit)
        active = [n for n, w in weights.items() if w > 0]
        if not active:
            raise ValueError('terrain_proportions must have a positive sum in mixture mode')
        if len(active) > cfg.num_cols:
            raise ValueError('num_cols must be >= the number of positive terrain weights; or specify terrain_columns')
        # Closest integer quotas with at least one column for each positive type.
        w = np.array([weights[n] for n in active], dtype=float)
        w /= w.max()
        quotas = cfg.num_cols * w / w.sum()
        counts = np.maximum(1, np.floor(quotas).astype(int))
        while counts.sum() < cfg.num_cols:
            counts[np.argmax(quotas - counts)] += 1
        while counts.sum() > cfg.num_cols:
            excess = np.where(counts > 1, counts - quotas, -np.inf)
            counts[np.argmax(excess)] -= 1
        return [n for n, count in zip(active, counts) for _ in range(count)]

    def _difficulties(self):
        cfg = self.cfg
        shape = (cfg.num_rows, cfg.num_cols)
        mode = getattr(cfg, 'difficulty_mode', 'linear')
        if mode not in ('linear', 'random'):
            raise ValueError('difficulty_mode must be linear or random')
        explicit = getattr(cfg, 'row_difficulties', None)
        if explicit is not None:
            if not isinstance(explicit, (list, tuple, np.ndarray)) or len(explicit) != cfg.num_rows:
                raise ValueError('row_difficulties must have num_rows values')
            values = [_number(v, 'row_difficulties', 0, 1) for v in explicit]
            return np.repeat(np.array(values)[:, None], cfg.num_cols, axis=1)
        if mode == 'random':
            return np.random.RandomState(cfg.seed).uniform(*cfg.difficulty_range, size=shape)
        return np.repeat(np.linspace(*cfg.difficulty_range, cfg.num_rows)[:, None], cfg.num_cols, axis=1)

    def _parameters(self, name, difficulty, occurrence, overrides=None):
        if self.direct_options is not None and overrides is None:
            return self._sdk_parameters()
        kind = self._kind(name)
        p = deepcopy(DEFAULT_PARAMS[kind])
        p.update(COMMON_PARAMS)
        p['platform_size'] = self.cfg.platform_size
        if overrides is None:
            p.update(deepcopy(self.cfg.terrain_params.get(kind, {})))
            if name in self.variants:
                p.update(deepcopy(self.variants[name].get('params', {})))
        else:
            p.update(overrides)
        for key in p:
            if key.endswith('_range') and not (kind == 'rubble' and key == 'stone_size_range'):
                p[key] = _resolve(p[key], difficulty, f'{name}.{key}')
        for key in ('slope_signs', 'step_signs'):
            if key in p:
                signs = p[key]
                if not isinstance(signs, (list, tuple)) or not signs or any(s not in (-1, 1) for s in signs):
                    raise ValueError(f'{name}.{key} must contain -1 and/or 1')
                p[key] = [signs[occurrence % len(signs)]]
        if p['axis'] not in ('x', 'y'):
            raise ValueError(f'{name}.axis must be x or y')
        self._platform(p['platform_size'], name)
        _number(p['height_offset'], f'{name}.height_offset')
        if p['roughness'] is not None:
            if not isinstance(p['roughness'], dict) or set(p['roughness']) - set(DEFAULT_PARAMS['rough']):
                raise ValueError(f'{name}.roughness accepts only {list(DEFAULT_PARAMS["rough"])}')
            noise = dict(DEFAULT_PARAMS['rough'], **p['roughness'])
            noise['height_range'] = _resolve(noise['height_range'], difficulty, f'{name}.roughness.height_range')
            self._validate_rough(noise, name + '.roughness')
            p['roughness'] = noise
        if kind in ('rough', 'rough_slope'):
            self._validate_rough(p, name)
        if kind == 'flat':
            _number(p['height'], f'{name}.height')
        if kind == 'ramp' and not isinstance(p['center_platform'], bool):
            raise ValueError(f'{name}.center_platform must be a bool')
        if kind == 'rubble':
            low, high = _pair(p['stone_size_range'], f'{name}.stone_size_range')
            quantize = int if overrides is not None else round
            if not self.horizontal_scale <= low < high < min(self.env_length, self.env_width) or quantize(low / self.horizontal_scale) >= quantize(high / self.horizontal_scale):
                raise ValueError(f'{name}.stone_size_range must contain distinct resolved sizes smaller than the tile')
            _integer(p['num_rects'], f'{name}.num_rects')
        if kind in ('stairs', 'stairs_linear'):
            if not isinstance(p['mesh_vertical_faces'], bool):
                raise ValueError(f'{name}.mesh_vertical_faces must be a bool')
            _number(p['step_width'], f'{name}.step_width', self.horizontal_scale,
                    min(self.env_length, self.env_width) / 2)
        if kind == 'wave':
            _integer(p['num_waves'], f'{name}.num_waves', 1)
            if p['num_waves'] * 2 > min(self.length_per_env_pixels, self.width_per_env_pixels):
                raise ValueError(f'{name}.num_waves cannot be resolved by horizontal_scale')
        if kind == 'stepping_stones':
            _number(p['stone_size_range'], f'{name}.stone_size_range', self.horizontal_scale,
                    min(self.env_length, self.env_width) - self.horizontal_scale)
            _number(p['stone_distance_range'], f'{name}.stone_distance_range', 0,
                    min(self.env_length, self.env_width) - self.horizontal_scale)
        if kind in ('gap', 'hurdle'):
            length = self.env_width if p['axis'] == 'y' else self.env_length
            _number(p['width_range'], f'{name}.width_range', self.horizontal_scale,
                    (length - p['platform_size']) / 2)
            if kind == 'gap' and p['num_gaps'] is not None:
                count = _integer(p['num_gaps'], f'{name}.num_gaps', 1)
                if p['pattern'] not in ('uniform', 'centered'):
                    raise ValueError(f'{name}.num_gaps requires a uniform or centered pattern')
                axis_cells = self.width_per_env_pixels if p['axis'] == 'y' else self.length_per_env_pixels
                if p['pattern'] == 'uniform':
                    p['spacing'] = length / count
                else:
                    p['spacing'] = (round(p['width_range'] / self.horizontal_scale)
                                    + round(p['platform_size'] / self.horizontal_scale)) * self.horizontal_scale
            if p['pattern'] != 'centered':
                _number(p['spacing'], f'{name}.spacing', p['width_range'] + self.horizontal_scale, length)
            if kind == 'hurdle' and p['pattern'] not in ('mirrored', 'uniform', 'single'):
                raise ValueError(f'{name}.pattern must be mirrored, uniform or single')
            if kind == 'gap':
                if p['pattern'] not in ('mirrored', 'uniform', 'centered', 'single'):
                    raise ValueError(f'{name}.pattern must be mirrored, uniform, centered or single')
                if p['pattern'] == 'centered' and p['num_gaps'] is None:
                    raise ValueError(f'{name}.centered pattern requires num_gaps')
                if p['single_position'] not in ('after_platform', 'center'):
                    raise ValueError(f'{name}.single_position must be after_platform or center')
            if kind == 'gap' and p['num_gaps'] is not None:
                bands = obstacle_bands(axis_cells, self.horizontal_scale, p, kind)
                if len(bands) != count:
                    raise ValueError(f'{name}.num_gaps cannot fit inside the tile')
                sample = _center_slice((axis_cells, 1), getattr(self.cfg, 'origin_sample_size', 2.),
                                       self.horizontal_scale)[0]
                if any(start < sample.stop and end > sample.start for start, end in bands):
                    raise ValueError(f'{name}.num_gaps leaves no safe centre landing segment')
            elif p['pattern'] == 'uniform':
                axis_cells = self.width_per_env_pixels if p['axis'] == 'y' else self.length_per_env_pixels
                spacing_cells = round(p['spacing'] / self.horizontal_scale)
                if axis_cells % spacing_cells:
                    raise ValueError(f'{name}.spacing must divide the travel-axis tile length in cells')
        if kind == 'pit':
            _number(p['size_range'], f'{name}.size_range', self.horizontal_scale,
                    min(self.env_length, self.env_width) - 2 * self.horizontal_scale)
        if kind == 'beam':
            width = self.env_length if p['axis'] == 'y' else self.env_width
            _number(p['width_range'], f'{name}.width_range', self.horizontal_scale, width - self.horizontal_scale)
        self._check_height_bound(kind, p)
        return p

    def _check_height_bound(self, kind, p):
        # Conservative bound BEFORE SDK casts or accumulates into its int16 array.
        length = max(self.env_length, self.env_width)
        bound = abs(p.get('height', 0.)) + p.get('height_range', 0.) + p.get('depth_range', 0.)
        bound += p.get('slope_range', 0.) * length + p.get('amplitude_range', 0.)
        if 'step_height_range' in p:
            bound += p['step_height_range'] * (length / (int(p['step_width'] / self.horizontal_scale) * self.horizontal_scale) + 1)
        noise = p.get('roughness')
        bound += (noise['height_range'] + noise['height_step']) if noise else 0.
        bound += p.get('height_step', 0.) + abs(p['height_offset']) + self.vertical_scale
        if bound / self.vertical_scale > 32767:
            raise ValueError(f'{kind}: heights may overflow int16; reduce heights or increase vertical_scale')

    def _validate_rough(self, p, label):
        _number(p['height_step'], label + '.height_step', self.vertical_scale)
        _number(p['downsampled_scale'], label + '.downsampled_scale', self.horizontal_scale,
                min(self.env_length, self.env_width) / 2)

    def _sdk_parameters(self):
        options = dict(self.direct_options)
        name = options.pop('type')
        try:
            bound = inspect.signature(OFFICIAL_GENERATORS[name]).bind(None, **options)
            bound.apply_defaults()
        except TypeError as exc:
            raise ValueError(f'selected_kwargs for {name}: {exc}') from exc
        p = dict(bound.arguments)
        p.pop('terrain')
        for key, value in p.items():
            if key == 'downsampled_scale' and value is None:
                continue
            _number(value, f'selected_kwargs.{key}')
        # Validate direct SDK calls through equivalent wrapper contracts.
        mapping = {
            'sloped_terrain': ('ramp', {'slope_range': abs(p.get('slope', 0))}),
            'pyramid_sloped_terrain': ('slope', {'slope_range': abs(p.get('slope', 0))}),
            'stairs_terrain': ('stairs_linear', {'step_height_range': abs(p.get('step_height', 0)), 'step_width': p.get('step_width')}),
            'pyramid_stairs_terrain': ('stairs', {'step_height_range': abs(p.get('step_height', 0)), 'step_width': p.get('step_width')}),
            'wave_terrain': ('wave', {'amplitude_range': abs(p.get('amplitude', 0)), 'num_waves': p.get('num_waves')}),
            'discrete_obstacles_terrain': ('rubble', {'height_range': p.get('max_height'), 'stone_size_range': [p.get('min_size'), p.get('max_size')], 'num_rects': p.get('num_rects')}),
            'stepping_stones_terrain': ('stepping_stones', {'height_range': p.get('max_height'), 'stone_size_range': p.get('stone_size'), 'stone_distance_range': p.get('stone_distance'), 'depth_range': abs(p.get('depth', 0))}),
        }
        if name == 'random_uniform_terrain':
            low = _number(p['min_height'], 'selected_kwargs.min_height')
            high = _number(p['max_height'], 'selected_kwargs.max_height')
            if low > high:
                raise ValueError('selected_kwargs.min_height must be <= max_height')
            kind, equivalent = 'rough', dict(height_range=max(abs(low), abs(high)), height_step=p['step'],
                                            downsampled_scale=self.horizontal_scale if p['downsampled_scale'] is None else p['downsampled_scale'])
        else:
            kind, equivalent = mapping[name]
        if 'platform_size' in p:
            equivalent['platform_size'] = p['platform_size']
        self._parameters(kind, 0., 0, overrides=equivalent)
        return p

    def _add_tile(self, raw, row, col):
        # Origin sampling is independent of whether the geometry has a flat platform.
        sample_size = getattr(self.cfg, 'origin_sample_size', 2.)
        if getattr(self.cfg, 'spawn_platform', False):
            sample_size = min(sample_size, self.tile_parameters[row, col].get('platform_size', self.cfg.platform_size))
        region = _center_slice(raw.shape, sample_size, self.horizontal_scale)
        height = int(np.max(raw[region]))
        ix = self.border + row * self.length_per_env_pixels
        iy = self.border + col * self.width_per_env_pixels
        self.height_field_raw[ix:ix+raw.shape[0], iy:iy+raw.shape[1]] = raw
        self.env_origins[row, col] = ((row + .5) * self.env_length,
                                      (col + .5) * self.env_width, height * self.vertical_scale)

    def _platform(self, size, label):
        _number(size, label + '.platform_size', self.cfg.horizontal_scale)
        if size + 2 * self.cfg.horizontal_scale >= min(self.cfg.terrain_length, self.cfg.terrain_width):
            raise ValueError(f'{label}.platform_size must fit inside the tile')
        if self.cfg.spawn_xy_jitter >= size / 2:
            raise ValueError(f'{label}.platform_size must contain spawn_xy_jitter')

    def _validate_config(self):
        cfg = self.cfg
        if cfg.mesh_type not in ('trimesh', 'heightfield'):
            raise ValueError('BlindTerrain requires trimesh or heightfield')
        for key in ('horizontal_scale', 'vertical_scale', 'terrain_length', 'terrain_width'):
            if _number(getattr(cfg, key), key, 0) == 0:
                raise ValueError(f'{key} must be positive')
        for key in ('num_rows', 'num_cols'):
            _integer(getattr(cfg, key), key, 1)
        _integer(cfg.seed, 'seed')
        if cfg.seed >= 2**32:
            raise ValueError('seed must be < 2**32')
        _number(cfg.border_size, 'border_size', 0)
        for value in (cfg.terrain_length, cfg.terrain_width, cfg.border_size):
            if not np.isclose(value / cfg.horizontal_scale, round(value / cfg.horizontal_scale)):
                raise ValueError('Tile dimensions and border must be multiples of horizontal_scale')
        _number(cfg.spawn_xy_jitter, 'spawn_xy_jitter', 0)
        self._platform(cfg.platform_size, 'terrain')
        _number(cfg.boundary_margin, 'boundary_margin', 0)
        if cfg.boundary_margin >= min(cfg.terrain_length, cfg.terrain_width) / 2:
            raise ValueError('boundary_margin must fit inside a tile')
        if cfg.slope_threshold is not None and _number(cfg.slope_threshold, 'slope_threshold', 0) == 0:
            raise ValueError('slope_threshold must be positive or None')
        a, b = _pair(cfg.difficulty_range, 'difficulty_range')
        if not 0 <= a <= b <= 1:
            raise ValueError('difficulty_range must be ordered within [0, 1]')
        sample = _number(getattr(cfg, 'origin_sample_size', 2.), 'origin_sample_size', cfg.horizontal_scale,
                         min(cfg.terrain_length, cfg.terrain_width))
        if sample / 2 <= cfg.spawn_xy_jitter:
            raise ValueError('origin_sample_size must cover spawn_xy_jitter')
        if not isinstance(getattr(cfg, 'spawn_platform', False), bool):
            raise ValueError('spawn_platform must be a bool')
        self.variants = deepcopy(getattr(cfg, 'terrain_variants', {}))
        if not isinstance(self.variants, dict):
            raise ValueError('terrain_variants must be a dict')
        if not isinstance(cfg.terrain_params, dict):
            raise ValueError('terrain_params must be a dict')
        for name, params in cfg.terrain_params.items():
            self._check_keys(name, params)
        for name, variant in self.variants.items():
            if not isinstance(name, str) or name in DEFAULT_PARAMS or not isinstance(variant, dict) or set(variant) - {'type', 'params'}:
                raise ValueError(f'Invalid terrain variant: {name}')
            self._check_keys(variant.get('type'), variant.get('params', {}))
        self.direct_options = getattr(cfg, 'selected_kwargs', None) if cfg.selected else None
        if self.direct_options is not None and (not isinstance(self.direct_options, dict) or self.direct_options.get('type') not in OFFICIAL_GENERATORS):
            raise ValueError('selected_kwargs.type must name an official terrain generator')

    @staticmethod
    def _check_keys(kind, params):
        if kind not in DEFAULT_PARAMS:
            raise ValueError(f'Unknown terrain type: {kind}')
        if not isinstance(params, dict):
            raise ValueError(f'terrain_params.{kind} must be a dict')
        unknown = set(params) - set(DEFAULT_PARAMS[kind]) - set(COMMON_PARAMS)
        if unknown:
            raise ValueError(f'{kind}: unknown parameters {sorted(unknown)}')
