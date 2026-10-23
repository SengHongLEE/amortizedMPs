# MP multi primitive 地形配置

配置入口：`legged_gym/legged_gym/envs/a1/mp_multi_primitive_config.py` 中的
`MotorPrimitivesCfg.terrain`。生成器：`legged_gym/legged_gym/utils/terrains/blind_terrain.py`。
所有尺寸和高度单位为米，坡度为 `dz/dx`（不是角度）。

地形库支持 14 类地形；实际启用类型由 config 中的选择和非零权重决定。
默认使用 `trimesh`、4 行 × 20 列，启用全部 14 类。类型覆盖不代表已经验证了该分布上的训练收敛，
可以先缩小类型和难度范围。
`plane` 或 `None` 沿用任务原有逻辑，不构建这些地形。

## 常用配置

以下片段放入已有的 `class terrain`，替换同名字段即可。

只训练固定高度的上行直楼梯：

```python
selected = 'stairs_linear'
terrain_params = {
    'stairs_linear': {
        'step_height_range': 0.08,
        'step_width': 0.3,
        'step_signs': [1],
        'axis': 'x',
    },
}
```

混合训练：

```python
selected = None
terrain_columns = None
num_cols = 10
terrain_proportions = {'flat': 4, 'rough': 3, 'stairs': 2, 'gap': 1}
num_rows = 4
row_difficulties = None
difficulty_mode = 'linear'
difficulty_range = [0.1, 0.8]
```

类型按列分配，权重无需归一化。分配会尽量接近权重比例，同时保证每个正权重类型
至少一列。上例恰好分配 4/3/2/1 列；极小权重仍占一列，所以列数较少时实际比例
与权重可能有差异。`num_cols` 少于正权重类型数量时明确报错；权重为 0 的类型不生成。
任务中的机器人按列索引均匀分配；机器人数量少于列数时，部分列不会分到机器人。

同一种地形使用不同参数，并指定精确布局：

```python
selected = None
num_cols = 3
terrain_variants = {
    'gentle_up': {'type': 'ramp', 'params': {
        'slope_range': [0.05, 0.15], 'slope_signs': [1]}},
    'steep_down': {'type': 'ramp', 'params': {
        'slope_range': [0.15, 0.35], 'slope_signs': [-1]}},
    'side_rough': {'type': 'ramp', 'params': {
        'axis': 'y', 'slope_range': 0.1, 'slope_signs': [1],
        'roughness': {'height_range': [0.005, 0.03]}}},
}
terrain_columns = ['gentle_up', 'steep_down', 'side_rough']
```

变体名也可用于 `selected` 或 `terrain_proportions`；变体不能覆盖内建名称。
参数优先级为：内建默认参数 → `terrain_params[类型]` → `terrain_variants[名称]['params']`。
允许只覆盖部分参数。未知类型和拼错的参数名会报错。

布局优先级：`selected` → `terrain_columns` → `terrain_proportions`。
`terrain_columns` 必须恰好包含 `num_cols` 个名称，可重复。

## 地形类型与参数

`*_range` 通常接受固定标量或 `[难度0的值, 难度1的值]`，线性插值。
允许倒序，例如踏石尺寸 `[0.8, 0.3]` 表示越难越小。
**例外：`rubble.stone_size_range` 是每个随机矩形的尺寸区间 `[min, max)`，必须递增。**

| 类型 | 几何含义 | 专用参数 |
| --- | --- | --- |
| `flat` | 水平地面 | `height`：地面高度 |
| `slope` | SDK 金字塔形坡面，中心平台 | `slope_range`、`slope_signs` |
| `ramp` | 单方向连续直坡 | `slope_range`、`slope_signs` |
| `rough` | SDK 插值随机粗糙面 | `height_range`：正负扰动幅值、`height_step`、`downsampled_scale` |
| `rough_slope` | 金字塔坡面叠加粗糙面 | slope 与 rough 的参数 |
| `rubble` | 随机矩形凸块及凹陷，可重叠 | `height_range`、`stone_size_range`、`num_rects` |
| `stairs` | 向中心递进的金字塔楼梯 | `step_height_range`、`step_width`、`step_signs` |
| `stairs_linear` | 单方向直楼梯 | 同 stairs |
| `stepping_stones` | 随机高度踏石，石间为凹陷 | `stone_size_range`、`stone_distance_range`、`height_range`、`depth_range` |
| `wave` | SDK 正弦/余弦二维波浪 | `amplitude_range`、`num_waves` |
| `gap` | 中心安全区域两侧的重复横向沟 | `width_range`：沟宽、`depth_range`、`spacing`：沟起点间距 |
| `pit` | 中心方形下陷区域，可训练从坑内向外攀爬 | `size_range`：坑边长、`depth_range` |
| `hurdle` | 中心安全区域两侧的重复横栏 | `width_range`：横栏厚度、`height_range`、`spacing`：横栏起点间距 |
| `beam` | 沿行进方向的窄平衡木，中心有平台 | `width_range`：木宽、`depth_range`：两侧落差 |

`depth_range` 使用非负深度，生成器转换成负地面高度。
`spacing` 必须大于横栏/沟宽，且至少保留一个网格的间隔。
`wave` 的 SDK 参数 `amplitude` 为两个正弦分量的总幅值尺度；方形地块上峰谷差约为其两倍。
SDK `rubble` 是矩形块，不是不规则多面岩石；高度场无法表达悬空、洞穴或多层可行走表面。

每种类型和变体都可配置：

| 参数 | 含义 |
| --- | --- |
| `axis` | `'x'` 或 `'y'`；`y` 在交换矩形地块尺寸后生成，再转置回世界坐标 |
| `platform_size` | 覆盖全局平台边长；传给支持平台的生成器；直坡、直楼梯、rough、wave 默认不会自动变平 |
| `height_offset` | 整块地形额外高度偏移，可正可负 |
| `roughness` | `None` 或 rough 参数字典，在基础几何上额外叠加粗糙度 |

`slope_signs`、`step_signs` 可设 `[1]`、`[-1]` 或 `[-1, 1]`。
列表按同一类型/变体的列出现次序循环，同一列所有行方向固定。
只有一列时只使用列表第一个方向；需要同时覆盖上下行时，用两个显式变体列。

## 难度、随机性与出生位置

- `difficulty_mode='linear'`：按行等距覆盖 `difficulty_range` 的两个端点。
  单行使用下端点；固定难度用 `[d, d]`。
- `difficulty_mode='random'`：每块地形在范围内独立均匀采样难度。
- `row_difficulties=[0., .2, .5, 1.]`：指定每行难度，长度必须等于 `num_rows`，
  优先于上述两种模式。各值在 `[0,1]` 内，可以乱序。
- `max_init_terrain_level=None`：初始化随机使用所有行；整数表示可用的最大行索引（含）。
  这限制机器人初始行，**不是**动态难度课程；当前任务没有自动升降行逻辑。
- `seed`：固定后布局、随机难度和几何可复现。每块地形使用由 seed、行、列确定的随机种子，
  构建后恢复训练用的 NumPy 全局随机状态。
- `spawn_platform=False`：默认不修改生成器输出。
  `True` 会在所有类型上将中心 `platform_size` 方形区域设为中心样本的高度，
  在粗糙度叠加后执行。它会改变直坡等几何，只在需要平坦出生区域时启用。
- `origin_sample_size=2.0`：取中心方形区域最高点作为环境原点 z。
  启用 `spawn_platform` 时采样范围不超过该地块平台；因此原点与平台高度一致。
- `spawn_xy_jitter`：reset 时 xy 均匀扰动范围。必须落在平台尺寸和原点采样范围内。
  非平坦出生区的最高点可能高于机器人所在点，机器人初始化会有一段下落。
  自定义 `init_state.pos[:2]` 偏移时，需同时确保它仍落在配置的出生区域。
  对于 pit，若希望原点在坑底，`origin_sample_size` 应不大于坑边长；
  否则最高点采样会包含坑外地面。
- `boundary_margin`：接近地块边缘时按超时结束 rollout。相邻地块直接拼接，
  不自动平滑边界，存在高度差时会产生断面。

`horizontal_scale` 决定地形细节分辨率；`vertical_scale` 决定高度量化。
地块长宽和 border 必须是水平分辨率的整数倍。普通配置中的台阶宽度、踏石尺寸、
块尺寸、平台、沟宽和梁宽使用最近网格，避免浮点取整把 0.3 米误变为 0.2 米。
直楼梯末端不足一个完整踏面的部分会延续最后一级，不会突然归零。
高度仍按 SDK/高度场单位量化，不能保证任意连续米制参数在离散网格中精确表示。
直接 SDK 模式保留官方原生量化和形状。
参数验证会拒绝无法解析的宽度、无效步长、越界平台及可能导致 int16 高度溢出的配置。

`trimesh` 使用 SDK 网格转换和 `slope_threshold` 形成陡直侧面；`heightfield`
保留高度场表示。二者在台阶边缘的实际碰撞形状有区别。

## 直接使用 SDK 参数（兼容旧配置）

```python
selected = 'slope'
selected_kwargs = {
    'type': 'pyramid_sloped_terrain',
    'slope': 0.2,
    'platform_size': 3.0,
}
```

`selected_kwargs` 非空且 `selected` 非空时，直接使用官方函数参数；
不套用难度插值、类型参数、axis 或 roughness。每块地形使用相同参数，
随机几何仍使用各自种子。`selected_kwargs=None` 恢复普通参数包装。
混合模式忽略此字段。支持 `sloped_terrain`、`pyramid_sloped_terrain`、
`random_uniform_terrain`、`discrete_obstacles_terrain`、`stairs_terrain`、
`pyramid_stairs_terrain`、`stepping_stones_terrain` 和 `wave_terrain`。

## 预览和验证

在装有 Isaac Gym 的 Python 环境中，从仓库根目录执行；本机环境为 `cpgil`。

```bash
# 当前配置；显示最后一行的全部列
python legged_gym/legged_gym/scripts/preview_mp_terrain.py
# 全部内建类型的最高难度对照
python legged_gym/legged_gym/scripts/preview_mp_terrain.py --gallery
# 单类型/变体、固定难度、自定义输出
python legged_gym/legged_gym/scripts/preview_mp_terrain.py --terrain uphill --difficulty 0.5 --output /tmp/uphill.png
# 不启动仿真的数值高度图（仍需 Isaac Gym 的 Python 包）
python legged_gym/legged_gym/scripts/preview_mp_terrain.py --gallery --heightmap
# 检查指定行，或限制显示列数
python legged_gym/legged_gym/scripts/preview_mp_terrain.py --row 0 --max-tiles 4

python -m unittest discover -s legged_gym/tests
python legged_gym/tests/smoke_blind_terrain.py --mesh trimesh
python legged_gym/tests/smoke_blind_terrain.py --mesh heightfield
python legged_gym/tests/smoke_mp_terrain.py --device cuda:0 --all-types
```

预览会打印每列实际类型、难度及解析后的参数。程序中可以检查
`terrain.terrain_names`、`terrain.difficulties`、`terrain.tile_parameters`
（形状均为 `[num_rows, num_cols]`）。`heightsamples` 和 `env_origins` 保持训练接口。

[14 类地形高度图](terrain_heightmaps.png) 由上述 `--gallery --heightmap` 命令生成。
[14 类真实网格画廊](configurable_terrain_gallery.png) 由 `--gallery` 命令生成。
原来的 `official_terrain_preview.png` 和 `training_terrain_preview.png` 是旧配置快照，
不代表当前默认混合分布。

## 与旧配置的差异

- 由单一 `selected='slope'` 改为按配置权重混合；可设回 `selected='slope'`。
- 难度插值现在包含最大端点；旧方式是 `row / num_rows`，达不到配置上限。
  如需旧四行难度，可设 `row_difficulties=[0., .25, .5, .75]`。
- 混合类型按有最小覆盖保证的整数配额分配，不再按累计概率的列中点取样。
- 每块地形单独设种子，旧 seed 生成的随机地图与新版本不逐像素相同。
- actor/critic 观测、奖励和环境 reset 接口没有增加地形感知逻辑。
