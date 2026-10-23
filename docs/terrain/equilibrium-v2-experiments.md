# 动力学动作 v2 实验

当前是实验版本，接口及 PPO 更新已验证，跨越能力须以保存的评估结果为准。
默认 `MP_pro_terrain_a1` 仍为 v1；独立任务 `MP_pro_terrain_v2_a1` 为 v2。

## 动作与状态

| action 下标 | 参数 | 零动作 | 范围 |
|---|---|---|---|
| 0 | 步态 ID | walk | 0–5，离散 |
| 1:5 | 四腿水平幅值 | 与 v1 相同 | 与 v1 相同 |
| 5:17 | bx4/by4/bz4 平衡目标 | 0 m | XYZ ±[.04,.025,.04] m |
| 17:21 | 四腿摆动高度目标 | .15 m | [.06,.20] m |
| 21 | 步频倍率目标 | 1 | [.35,1.2] |
| 22 | 支撑占空比偏移目标 | 0 | [−.15,.25] |

腿序均为 FL/FR/RL/RR。连续动作裁剪至 [−1,1] 后，以零为中性值分段线性映射。
频率倍率和占空比偏移通过时间常数 .10 s 的一阶动力系统。
最终占空比限于 [.30,.80]。高度目标通过已有二阶振子幅值动力学；
接触反射仍可把高度提高到 .20 m。平衡位置仍采用二阶滤波、速度和加速度限制。
没有新增电机关节残差。

actor 132 维、critic 139 维。新增六个历史动作值和两个实际时序滤波状态。
actor 仍只读取本体感知与控制器内部状态，没有地形高度图。

所有参数位于 `MotorPrimitivesCfg.control.timing`。
`align_stroke=True` 用垂直相位 psi 同时生成水平轨迹，使水平换向与着地对齐。
设为 False 可隔离对齐效应；它不改变新动作和观测维度。
课程开关仍为 `terrain.curriculum`，关闭时可设置 `fixed_terrain_level`。

## 显式热启动与训练

使用 cpgil 环境并确保其 bin 在 PATH：

```bash
python -B legged_gym/legged_gym/scripts/train_adaptive_equilibrium.py \
  --checkpoint <v1或v2的checkpoint路径> \
  --output <新的实验目录> --iterations 100 --num-envs 2048
```

v1 不可直接 load 为 v2。脚本明确映射新旧观测位置、扩展低层输出，
保留旧权重，新增输入权重和输出均值为零，六个新动作标准差默认 .2。
测试覆盖低层均值、高层 logits/内部状态和 critic 的一致性。
优化器重新初始化；v2 输入会严格加载相同维度网络。
`--iterations 0` 只保存可加载的 `warmstart.pt`，便于独立验证。
`--legacy-stroke` 用于关闭水平相位对齐的消融实验。
输出目录保存配置、源 checkpoint SHA256、源码副本和实验参数。

## 独立混合评估

```bash
python -B legged_gym/legged_gym/scripts/evaluate_proprio_terrain.py \
  --checkpoint <checkpoint路径> --output <报告目录> \
  --repeats 8 --speeds .5 .8 1.2 --difficulties .5 1 \
  --seed 123 --terrain-seed 1
```

从源 run 配置读取地形类型、范围及模型维度。每类两列覆盖两个方向或不同 tile。
不使用课程选择测试难度，不计复位后的 episode。正向越界、未跌倒且全过程侧偏
不超过 1.5 m 才记成功。默认关闭域随机化以比较控制能力，后续需单独评估扰动鲁棒性。
原始 report 保存逐 episode 结果，避免混合平均掩盖楼梯失败。
