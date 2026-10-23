# mp_proprio_terrain：本体感知地形与课程学习

配置：`legged_gym/legged_gym/envs/a1/mp_proprio_terrain_config.py`。
注册任务名：`MP_pro_terrain_a1`。

默认启用课程学习：10 行 × 16 列，从第 0–1 行起步，根据每个机器人完成回合的表现逐行升级或降级。
最高难度包含约 16.7° 坡面、8 cm 台阶和最大 8 cm 的相邻凸凹块高差。
这些上限是待训练验证的挑战范围，不是已经测得的稳定通过能力。

## 开关与难度控制

在 `class terrain` 中修改，随后重新创建环境：

```python
# 逐步学习：初始随机分配第 0–1 行，最多进入第 9 行。
curriculum = True
curriculum_initial_level = 1
curriculum_min_level = 0
curriculum_max_level = None       # None 表示 num_rows - 1
curriculum_promote_distance = 2.5 # 沿命令方向的净前进距离，m
curriculum_promote_ratio = 0.8
curriculum_demote_ratio = 0.5
```

关闭课程、混合全部难度：

```python
curriculum = False
fixed_terrain_level = None
max_init_terrain_level = None
```

关闭课程、指定固定难度，例如直接检查最高难度：

```python
curriculum = False
fixed_terrain_level = 9
```

关闭时，每个环境的随机或固定等级在创建时确定，以后 reset 保持该等级。
`max_init_terrain_level` 只限制关闭课程后的随机初始行；开启时由
`curriculum_initial_level` 控制初始上限。`fixed_terrain_level` 只在课程关闭时生效。
`cfg.env.play=True` 自动停止课程更新，使用同样的固定/随机评估设置。
平面地形不运行地形课程。
`play.py` 会先读取训练 run 保存的配置，但当前脚本随后会对 trimesh 强制覆盖为
10 行、2 列 stairs。评估其他地形时，应同步修改这一覆盖段。
若要改变回放难度，在 `load_run_config` 之后覆盖相应 terrain 字段。
单机器人默认分配第一列，因此单独评估台阶时还需设置 `terrain.selected='stairs'`。

如果已有基础行走能力，可以把 `curriculum_initial_level` 提高到 3 或 4；
这表示在最低行与该行之间随机初始化，并不保证所有环境都从第 4 行开始。
若需要更难的统一起点，可同时调整 `curriculum_min_level`。
`curriculum_max_level=5` 可暂时限制挑战上限，之后再放开。

### 回合判定

- 升一级：没有摔倒；沿累计命令方向的净位移至少 2.5 m；同时达到实际累计命令路程的 80%。
- 降一级：摔倒，或者沿命令方向的净位移不足累计命令路程的 50%。
- 其他情况保持等级。每次最多变化一行，达到上下限后停留在该行。
- 升降级发生在自动结束回合之后、下一次出生之前；先切换地块原点，再 reset 机器人。
- 手动 `reset()` / `reset_idx()` 不改变等级，零长度回合不计成绩。
- 命令路程逐步累计，因此中途重新采样速度也有效；不是用最后一个命令乘整个回合时长。
- 净位移取命令方向投影，横向漂移或倒退不会因欧氏距离大而升级；距离从实际出生位置算起。
- 2.5 m 大于默认中心平台的半长 1.5 m，机器人必须走出起步区才可能升级。
- 摔倒和越界/超时同时发生时，按失败处理，不当作正常超时引导价值估计。

当前命令是沿 x 前进，航向为零。累计方向支持命令和航向变化；如果改成往返或闭环路径，
净位移不适合作为完整路径任务的成绩，应相应改用航点完成度。

训练的 episode 日志增加 `terrain_level`、`terrain_difficulty`、`terrain_curriculum_enabled`。
前两者是所有环境的当前平均等级/难度，包括作为基础训练的平地列。
改变等级的逻辑位于 `utils/terrains/terrain_curriculum.py`，不向 actor 添加地形信息。

课程要求各列的行难度非递减：默认 `difficulty_mode='linear'`；也可以设置长度等于
`num_rows` 的非递减 `row_difficulties`。无明确行难度的 `difficulty_mode='random'`
会在开启课程时直接报错。单行地形需设置 `curriculum_initial_level=0`。
自定义参数的 easy/hard 两端仍需由使用者保证实际难度顺序；例如减小石块间距可能反而更容易。

## 观测与默认地形

actor 每帧 **124 维**，环境保存 5 帧历史；当前低层 SNN 使用最新一帧，
高层 reservoir 使用历史及其内部状态。critic 为 **131 维**。
原有 54 维包括机身角速度、重力投影、关节位置/速度、四足接触、
前次动作和速度命令；新增 70 维全部是控制器内部状态：
腿相位 sin/cos（8）、共享时钟 sin/cos（2）、幅值及其速度（16）、
相位修正（4）、三类接触反射状态（12）、支撑延伸（4）、
实际平衡位置偏移及其速度（24）。幅值、速度、偏移等按控制器尺度归一化。
critic 额外获得机身线速度和姿态四元数。
二者均无高度图、深度图或前方落脚点信息；高度场仅用于出生和终止判断。
开启 `noise.add_noise` 时只对角速度、重力、关节位置/速度叠加噪声，
不扰动动作、命令或控制器内部状态。

### 动力系统动作接口（cpg_equilibrium_v1）

动作共 17 维：1 个离散步态选择 + 16 个连续调制量。
每组四条腿均按 **FL、FR、RL、RR** 排列：

| action 切片 | 参数 | 默认物理范围 |
| --- | --- | --- |
| `0` | 步态 ID | 0 walk / 1 pace / 2 trot / 3 canter / 4 bound / 5 gallop |
| `1:5` | 水平振荡器幅值平方目标 | 归一化动作 −1…1 映射到 0…1，水平半行程最多 0.10 m |
| `5:9` | 前后平衡位置目标 bx | ±0.04 m |
| `9:13` | 左右平衡位置目标 by | ±0.025 m |
| `13:17` | 垂直平衡位置目标 bz | ±0.04 m |

三个方向均为髋关节坐标系：x 正向前、y 正向左、z 正向上；
by 不含腿的固有 ±0.0838 m 髋外展偏置。四腿的正 y 均指向机身左侧，
因此对称增宽支撑需要左腿 by 为正、右腿 by 为负。

RL 输出目标 `b*`，实际偏移 `b` 由临界阻尼系统积分得到：
`b_ddot = ω² (b* - b) - 2ω b_dot`，`ω = 2π frequency_hz`。
足端参考为 `p_CPG + b`，之后经过工作空间处理、有界 IK 和原有 PD。
**IK 之后不再叠加任何 RL 关节角残差。**
垂直平衡位置作用于摆动及支撑全周期，不等同于抬脚高度，也不是地面高度估计。
动作 `control.action_scale` 只缩放连续部分，不改变步态 ID。

在 `class control` 内配置：

```python
class equilibrium:
    offset_limits = [0.04, 0.025, 0.04]  # XYZ 目标范围，m
    frequency_hz = 3.0                 # 偏移响应频率，不是步态频率
    swing_velocity_limits = [0.4, 0.25, 0.4]   # m/s
    stance_velocity_limits = [0.06, 0.04, 0.08]
    acceleration_limits = [4.0, 3.0, 4.0]      # m/s²
    joint_limit_margin = 0.02                 # URDF 硬限位内的 rad 裕量
    min_foot_depth = 0.06                     # IK 前要求足端位于髋下方，m
    min_sagittal_reach = 0.06                 # 避免髋轴附近的折叠奇异区，m
```

名义支撑期或实际接触时采用支撑速度上限；限速仅作用于偏移，不冻结 CPG 自身轨迹。
进入支撑时速度会立即受更低上限约束，`acceleration_limits` 约束两次门控切换之间的积分。
偏移有位置限幅及防积分积累。超出工作空间的足端会被投影，关节目标最终限制在
URDF 硬限位减去裕量的范围内。可用 `env.foot_targets_requested` 与
`env.foot_targets_reachable` 检查投影影响；后者是最终关节目标的 FK。
这不是碰撞检测、执行器速度保证或闭环稳定性证明。

动作版本由 `env.action_space='cpg_equilibrium_v1'` 标识。
**旧关节残差 checkpoint 和旧观测维度不兼容，必须新建训练 run。**
默认 `runner.resume=False`，新 run 名含 `equilibrium-v1`。
不要只修改旧配置的动作版本字段来继续训练；旧策略输出不具备新动作含义。

单位为米；参数按行从 easy 插值至 hard。实际高度还会按 0.005 m 网格量化。

| 类型 | 参数范围 | 说明 |
| --- | --- | --- |
| flat | 高度 0 | 基础步态、速度跟踪、恢复能力 |
| slope | 坡度 0.02–0.30，两个方向 | 金字塔坡面，名义坡角最高约 16.7° |
| ramp | 坡度 0.02–0.30，两个方向 | 单向坡，中心平坦起步段与两侧坡面连续连接 |
| rough_slope | 坡度 0.02–0.20；扰动幅值 0.005–0.025 | 坡面与粗糙度组合；粗采样间距 0.3 |
| rough | 扰动幅值 0.005–0.04 | 连续粗糙面；粗采样间距 0.3 |
| rubble | 凸凹幅值 0.01–0.04；块尺寸 0.6–1.2；12 块 | 正负块相邻时最大高差 8 cm |
| stairs | 每级 0.01–0.08；踏面 0.6；两个方向 | 从低台阶逐渐过渡到有挑战的台阶 |
| wave | SDK amplitude 0.01–0.12；num_waves=2 | 连续波浪；二维表面的峰谷总差约为 amplitude 的两倍 |

每类两列，上下坡、上下台阶均得到覆盖。初始第 0–1 行的采样相邻高差不超过 2 cm；
保留容易的起点，不再把容易的参数作为整个训练过程的上限。
`stairs_linear` 保留相同台阶范围但默认权重为零；`side_slope` 变体坡度为 0.02–0.20。

`ramp.center_platform=True` 为本任务启用的起步段选项，长度使用 `platform_size=3.0`。
它把两侧坡面向外平移后接到平台，避免直接压平已有直坡产生断崖；也避免高难度直坡
按最高点出生时悬空约 18 cm。其他任务的 ramp 默认保持旧行为。
全局 `spawn_platform=False` 保持不变。

### 默认排除

`stepping_stones`、`gap`、`beam`、`pit`、`hurdle` 的权重仍为零。
当前连续前进命令与本体观测缺少前方落脚点、沟边位置和窄梁路径信息；这些地形应作为
单独评估任务逐项验证，不宜仅为增加难度就混入训练分布。生成器仍保留全部配置接口。

本体感知并非只能走平地；已有研究展示了本体历史和接触适应对复杂地形的价值，
但不同机器人与控制器的结果不能直接作为本项目的通过保证。
参考：[Lee 等的 blind locomotion 项目](https://leggedrobotics.github.io/rl-blindloco/)，
以及使用视觉信息处理复杂地形的 [Agarwal 等，CoRL 2022](https://arxiv.org/abs/2211.07638)。

## 机器人参数与验证边界

- IK 大腿/小腿长度为 0.200 m，与 `a1_updated_v2.urdf` 一致；水平半行程 0.10 m。
- 名义抬脚 0.15 m，接触反射上限 0.20 m；与平衡位置组合后可能触及工作空间边界，
  此时由有界 IK 约束最终关节目标。
- 速度命令 0.3–0.8 m/s、摩擦系数 0.6–1.0、外部推扰速度 0.3 m/s。
- 新动作空间尚未完成策略训练，不能把旧 checkpoint 的楼梯表现当作新控制器的通过率。

早期配置的几何筛查记录：[proprio_feasibility.json](proprio_feasibility.json)。
预览：[最高难度高度图](proprio_terrain_heightmaps.png)。

种子 1、7、19 共 480 块地形：全难度最大相邻高差 8 cm，出生高度差最大约 6 cm；
初始第 0–1 行的 96 块地形最大相邻高差为 2 cm。wave 最大采样坡度约 0.141，
低于 0.75 的陡面修正阈值。

该早期轨迹记录使用 4 种 primitive 和 15 cm 抬脚上限，不覆盖当前平衡位置动作空间。
当前共享 primitive 频率为 walk/pace/trot/canter/bound/gallop = 6/6/8/8/10/10 Hz。
本次没有修改共享 primitive 频率或奖励。位置可达不代表关节速度、力矩和接触稳定性可达；
实际通过能力需要训练 checkpoint 按类型、等级统计穿越率、摔倒率、速度误差与力矩饱和比例。

当前动作空间的报告：[equilibrium_feasibility.json](equilibrium_feasibility.json)。
覆盖六步态、四腿、三个水平幅值、两个抬脚幅值、128 个相位及 XYZ 的 27 个边界/中心组合，
共 497,664 个单腿样本。最终关节目标全部在 URDF 限位内，最小余量约 0.02 rad。
原始目标有 45,500 个位置不可达；包含奇异区与限位裕量的投影影响约 10.71% 样本，
最大足端修正约 4.06 cm。这是刻意覆盖极端组合的静态筛查比例，不是训练中的发生率。
边界组合的参考关节速度估计最高仍为 URDF 速度限值的约 5.19 倍；
偏移滤波并不限制原有 CPG 的全部运动速度，`dynamic_tracking_certified` 仍为 false。

## 复现

在 `cpgil` 环境中从仓库根目录执行：

```bash
python -B -m unittest discover -s legged_gym/tests -p 'test*terrain*.py'
python -B -m unittest discover -s legged_gym/tests -p 'test_equilibrium_dynamics.py'
python -B -m unittest discover -s rsl_rl/tests -p 'test_multi_interoceptive_empty_updates.py'
python -B legged_gym/legged_gym/scripts/audit_proprio_terrain.py --output docs/terrain/equilibrium_feasibility.json
python -B legged_gym/tests/smoke_mp_terrain.py --task proprio --mesh plane --device cpu --steps 20 --equilibrium-actions --noise --policy-update
python -B legged_gym/tests/smoke_mp_terrain.py --task proprio --configured --device cuda:0 --steps 20
python -B legged_gym/tests/smoke_mp_terrain.py --task proprio --configured --mesh heightfield --device cpu --steps 20 --curriculum off --fixed-level 9
python -B legged_gym/legged_gym/scripts/preview_mp_terrain.py --task proprio --heightmap
```

smoke 检查真实仿真步进、有限观测/奖励、手动 reset 不改等级，以及注入已完成回合成绩后
的等级和原点切换；它不是策略已经通过地形的测试。课程启用/禁用均有覆盖。
`--equilibrium-actions` 覆盖三个偏移轴的边界；`--policy-update` 使用配置的真实
SNN/reservoir actor 完成一次短 PPO 更新、保存/加载及推理，产物在临时目录中自动清理。

重新训练入口：

```bash
python -B legged_gym/legged_gym/scripts/train.py --task MP_pro_terrain_a1 --headless
```
