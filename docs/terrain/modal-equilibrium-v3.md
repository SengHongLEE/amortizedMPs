# 协同平衡动作 v3（实验）

该版本改变动作坐标与探索结构。控制器可表达的足端平衡目标范围与 v2 相同，
尚未证明能可靠通过 8 cm 上阶。默认任务和已训练的 v1/v2 权重仍可加载。

## 参数含义

任务名 `MP_pro_terrain_modal_a1`，`env.action_space='cpg_modal_equilibrium_v3'`。
仍为 23 个动作、132 维 actor observation、139 维 critic observation。

| 下标 | 参数 |
|---|---|
| 0 | 步态 ID |
| 1:5 | FL/FR/RL/RR 的水平振子幅值 |
| 5:9 | x 方向：整体、前后差、左右差、对角差 |
| 9:13 | y 方向：整体、前后差、左右差、对角差 |
| 13:17 | z 方向：整体、前后差、左右差、对角差 |
| 17:21 | 每腿摆动高度目标 |
| 21 | 步频倍率目标 |
| 22 | 支撑占空比偏移目标 |

每轴四个模式按下式组合成 FL/FR/RL/RR 的归一化平衡目标：

```text
FL = common + front_back + left_right + diagonal
FR = common + front_back - left_right - diagonal
RL = common - front_back + left_right - diagonal
RR = common - front_back - left_right + diagonal
```

环境入口先在 Cartesian 坐标裁剪至原动作范围，再反变换为模式动作，
同时用于控制器和下一次 observation 的动作历史。控制器随后映射到米制 XYZ
范围，进入相同的二阶平衡动力学。不能对原始模式分量直接裁剪后存作历史，
否则即使网络权重变换正确，闭环输入也不再与旧策略一致。
例如 z 前后差为 +0.5、其他模式为零时，前腿目标为 +2 cm、后腿为 −2 cm。
这是足端平衡位置差，不等于直接指定机身俯仰角；实际姿态取决于接触与负载。
模式动作不单独裁剪，Cartesian 合成目标负责限幅，因此变换前后的目标集合一致。
不会增加电机关节残差或地形观测。

## 显式迁移

```bash
python -u -B legged_gym/legged_gym/scripts/train_adaptive_equilibrium.py \
  --checkpoint <v2_checkpoint> --output <新目录> --iterations 0 \
  --modal-actions --modal-std-factors .75 1.5 1 .25
```

迁移同时变换低层输出、actor/history 输入、高层 reservoir 输入和两个 critic 的输入。
测试核对转换前后物理动作均值、高层 logits/状态、critic 输出以及控制器轨迹一致。
旧 v2 权重不能作为 v3 直接加载，必须经过显式迁移并保存新的配置。

对角 Gaussian 无法保留原分布的全部跨模式协方差。
默认迁移保留每个模式的边缘方差；`--modal-std-factors` 可进一步指定四类模式的
探索倍数，按 XYZ 三个轴重复。示例减少对角扭曲的探索，并增加前后协调的探索。
这些倍数只在该次显式调用时应用，继续训练时不要无意重复施加。
探索结构改变是否有益，以独立混合地形评估为准。
动作变化惩罚仍按原足端坐标计算，避免坐标缩放无意改变奖励权重。

## 当前验证范围

- 含越界连续动作的坐标变换，保持原 Cartesian 限位与动力学轨迹。
- 环境 step 入口的越界动作处理及历史观测一致性；曾发现旧入口逐分量
  裁剪破坏该约定，已修正。此前 v3 warmstart 回放及首轮短训练只作为缺陷记录，
  不用于宣称无损迁移或训练提升。
- 高低层网络和 critic 的显式迁移测试。
- 真实 CUDA 仿真、极端动作、部分重置、传感器噪声。
- 高低层各一次 PPO 更新、checkpoint 保存、重新加载和推理。
- 尚待完成：迁移后全地形回归、强化训练和多种子 6/7/8 cm 上下阶验收。
