# MP_full_a1 实验结果（2026-10-26）

> 本报告记录的是旧版 22 维连续动作与旧 gap/ramp 地形上的历史结果。当前训练代码已改为 18 维连续动作，并修正了地形几何；旧 checkpoint 和下文的旧版评测命令不能在当前代码上直接运行，成功率也不能与新版训练直接比较。新版请从头训练并重新评测。

## 结论与推荐

按**跨越当前地形格后继续通过下一格第二道障碍**的成功数，当前推荐 `1026_10-51-28_...-modC3-yaw1/model_19_200.pt`。它在 seed 123 的 128 次 gap/hurdle 测试中成功 29 次，在 seed 456 中成功 25 次，两种子合计 54/256；同期复测的 yaw2 训练 10 次模型合计 51/256。它**没有解决高难 hurdle**（两个种子均为 0/32），高难 gap 也仅有 2/32、0/32。因此“最佳”仅指目前的固定条件跨格评测，不能解释成复杂地形任务已经达标。

若只关心当前地形格第二道障碍，`1026_10-07-23_...-modC3-yaw2/model_0_10.pt` 在 seed 123 的高难 hurdle 为 12/32，明显高于推荐模型的 4/32；但其跨格总数为 26/128，且高难 gap 跨格为 0/32。当前没有一个 checkpoint 同时在两种障碍的中高难、短程和跨格指标上占优。

## 评测口径

- 使用 `evaluate_mp_full_terrain.py`，确定性 actor、固定前进速度 0.8/1.2 m/s、无推搡和随机化；每个地形与难度组合包含两个地形列 × 两个速度 × 每列每速度 8 次，即 32 次，整套 gap/hurdle 为 128 次。
- 难度取 0.5 和 0.75。“短程”要求在 10 s 内越过当前格第二道障碍；“跨格”要求在 24 s 内越过下一格第二道障碍。达到目标之前跌倒、离开 ±1.5 m 通道或超时均算失败。
- 训练 episode 仍为 10 s，跨格评测的 24 s 是额外的连续行走压力测试。下表除注明外使用 seed 123；不同运行即使起始状态哈希一致，GPU PhysX 结果也可能略有差异。
- 这些数字**不是**训练日志中的 `terrain_*_failure_rate`。训练的升级距离为 4.25 m，且失败率以 episode 结束为口径，不能直接代表下一格通过率。

## 可比 checkpoint 结果

每格数字为成功次数 / 32；顺序为 gap 中难、gap 高难、hurdle 中难、hurdle 高难。

| 候选 checkpoint | 当前格第二道障碍（10 s） | 下一格第二道障碍（24 s） | 合计短程 / 跨格 |
| --- | --- | --- | --- |
| modC3 基线，60 次 | 29, 17, 25, 8 | 15, 0, 8, 1 | 79 / 24 |
| modC4，50 次 | 31, 20, 25, 7 | 11, 0, 13, 1 | 83 / 25 |
| yaw2 + modC3，10 次 | 28, 15, 27, 12 | 15, 0, 10, 1 | 82 / 26 |
| yaw2 + lane 0.05，40 次 | 29, 19, 27, 9 | 8, 0, 14, 0 | 84 / 22 |
| **yaw1 + modC3，200 次** | **29, 18, 26, 4** | **17, 2, 10, 0** | **77 / 29** |

原始数据：

- [modC3 短程](docs/terrain/mp_full_v2_modcycle3_train60_second_band_128/report.json)、[跨格](docs/terrain/mp_full_v2_modcycle3_train60_next_tile_128/report.json)
- [modC4 短程](docs/terrain/mp_full_v2_modcycle4_train50_second_band_128/report.json)、[跨格](docs/terrain/mp_full_v2_modcycle4_train50_next_tile_128/report.json)
- [yaw2 短程](docs/terrain/mp_full_v2_modcycle3_yaw2_train10_second_band_128/report.json)、[跨格](docs/terrain/mp_full_v2_modcycle3_yaw2_train10_next_tile_128/report.json)
- [yaw2 + lane 短程](docs/terrain/mp_full_v2_modcycle3_yaw2_lane005_train40_second_band_128/report.json)、[跨格](docs/terrain/mp_full_v2_modcycle3_yaw2_lane005_train40_next_tile_128/report.json)
- [200 次短程](docs/terrain/mp_full_v2_modcycle3_yaw1_train200_second_band_128/report.json)、[跨格 seed 123](docs/terrain/mp_full_v2_modcycle3_yaw1_train200_next_tile_128/report.json)、[跨格 seed 456](docs/terrain/mp_full_v2_modcycle3_yaw1_train200_next_tile_seed456_128/report.json)
- [yaw2 跨格 seed 456](docs/terrain/mp_full_v2_modcycle3_yaw2_train10_next_tile_seed456_128/report.json)

跨格第二种子分项：推荐模型为 gap 中/高难 16/32、0/32，hurdle 中/高难 9/32、0/32；yaw2 训练 10 次模型分别为 13/32、0/32、12/32、0/32。两者总数相同，推荐模型在 gap 上较好，在 hurdle 上较差；两个种子的总差仅 3/256，不足以支持强泛化结论。

## 目前能解释的失败模式

1. **跨格高难 gap 几乎未掌握。** 推荐模型跨格仅 2/32（seed 123）和 0/32（seed 456）。[16 个环境的逐帧轨迹](docs/terrain/mp_full_v2_modcycle3_next_tile_trace_128/report.json)显示，许多机器人在下一格首道 gap 附近失去支撑；高度扫描已读到坑，调制动作也有变化，因此不能简单归因于“看不到地形”。这份轨迹来自 modC3 基线，属于机制线索，不是对推荐模型的直接因果证明。
2. **hurdle 的持续航向控制偏弱。** 推荐模型 seed 123 跨格中难 hurdle 32 次有 20 次记录横向出界，高难有 23 次；seed 456 则分别为 14 和 27 次。先前轨迹中还出现了指令已朝反方向纠偏，但实际偏航未及时改变的情况。yaw2 增强短程高难 hurdle，却未稳定改善跨格。
3. **后腿 calf 接触仍需关注。** 推荐模型 seed 123 跨格的后腿 calf 接触步占比，gap 中/高难约 0.10%/0.15%，hurdle 中/高难约 0.35%/0.28%。后腿前摆上限维持 0.07 m；先前将它增至 0.09/0.12 m 的尝试使高难 gap 和 calf 接触变差。
4. **当前推荐模型没有明显的动作限幅坍缩。** 200 次训练终点日志的 `applied_action_near_limit_fraction` 约 1.99%；seed 123 跨格 gap 中/高难的评测均值约 0.68%/1.28%，hurdle 约 0%。旧版未约束高斯输出的 smoke 测试约 36.9% 动作接近限幅，不能与当前模型混为同一问题。
5. **22 维连续动作中只有 18 维实际影响足端轨迹。** 四维 `lift` 当前仅写入 `nominal_clearance`，后续 z 幅值计算未读取它；单独扰动每一维的控制器测试显示，步幅 x/z 各 4 维、平衡位置 x/z 各 4 维、频率和 duty 共 18 维会改变足端目标，而 `lift` 四维无变化。这四维仍参与动作历史与动作变化惩罚。修复后需要重新评测已有 checkpoint，不能直接套用本报告的成功率。

## 历史 checkpoint 的旧版评测命令（当前代码不支持）

以下命令仅供对照旧实验记录。当前评测脚本会明确拒绝旧版 checkpoint；运行历史模型需要对应的旧版源代码和环境。新版模型训练完成后应使用新版 `evaluate_mp_full_terrain.py` 分别评测 `gap_continuous` 与 `gap_single`。

```bash
cd /home/ubuntu/amortizedMPs
PATH=/home/ubuntu/anaconda3/envs/cpgil/bin:$PATH \
PYTHONPATH=legged_gym:rsl_rl \
TORCH_EXTENSIONS_DIR=/home/ubuntu/amortizedMPs/legged_gym/legged_gym/test/torch_extensions \
PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/cpgil/bin/python \
  legged_gym/legged_gym/scripts/evaluate_mp_full_terrain.py \
  --checkpoint legged_gym/logs/A1/1026_10-51-28_A1MPfull-v2-psF_3.333e+00-mpF_1e+01-MF_1e+03-noResume-modC3-yaw1/model_19_200.pt \
  --output docs/terrain/mp_full_best_recheck \
  --terrains gap hurdle --difficulties 0.5 0.75 \
  --speeds 0.8 1.2 --repeats 8 --seed 123 --seconds 24 \
  --hazard-target next_tile_second_band --modulation-update-cycle 3
```

完成后读取 `docs/terrain/mp_full_best_recheck/report.json` 的各组 `success_rate` 和 `episodes`。原始 checkpoint 的 SHA-256 为 `af32384dc8a4d09681a3a9b6f52b2c6af79b80998461df208c4b14a8d6dac4e4`。普通 `play.py` 目前把 trimesh 固定为 hurdle 0.3，并把 checkpoint 重置为自动选最新文件；它不适合用来复现上表的精确 checkpoint 与地形协议。

### 从头训练与复现训练配方

直接执行 `python legged_gym/legged_gym/scripts/train.py --task MP_full_a1` 会从随机权重开始训练，**不会加载上述 checkpoint**。当前默认值为 1000 次更新、课程初始等级 1、调制更新周期 5、学习率 `1e-4`、`gamma=0.99`、`lam=0.95`，并随机化初始 episode 长度。推荐 checkpoint 是从既有 modC3 模型热启动，在课程初始等级 4、调制周期 3、学习率 `3e-5`、`gamma=0.994`、`lam=0.97` 下额外训练 200 次得到的；奖励中的两个 yaw 权重均设为 `-1`。

若要重跑该训练配方，在上面相同的 `PATH`、`PYTHONPATH` 和 `TORCH_EXTENSIONS_DIR` 环境下执行：

```bash
/home/ubuntu/anaconda3/envs/cpgil/bin/python \
  legged_gym/legged_gym/scripts/train_mp_full_optimized.py \
  --task MP_full_a1 --headless \
  --warmstart legged_gym/logs/A1/1026_08-19-25_A1MPfull-v2-psF_3.333e+00-mpF_1e+01-MF_1e+03-noResume-modC3/model_6_60.pt \
  --initial-level 4 --learning-rate 3e-5 \
  --modulation-update-cycle 3 --gamma 0.994 --gae-lambda 0.97 \
  --heading-error-scale -1 --iterations 200
```

这会产生**新的训练运行**；随机训练和 GPU 物理仿真的差异可能导致新的 checkpoint 性能不同。若目标只是体验当前已有的最佳模型，请直接加载并评测上面的 `model_19_200.pt`。

复现口径需要区分：加载同一个 checkpoint 重跑评测，可以复现协议与总体表现，但单次成功数不保证逐项完全相同。例如 modC3 基线的原始跨格评测与逐帧轨迹评测使用相同 checkpoint、seed 123 和相同初始状态哈希，结果仍从 24/128 变为 30/128。重新训练更无法保证生成相同权重；推荐模型保存的 `run_manifest.json` 所列 8 个源码文件中，当前 `mp_full.py` 已与训练时的 SHA-256 不同，因此当前代码也不构成原训练的逐字节源码快照。

## 局限和下一步

- 该评测只覆盖 gap/hurdle 的固定速度，不涵盖训练混合中的 flat、rough slope、rough、ramp 的同期可比结果，因而不能声称推荐模型是全地形最优。
- 每个难度只有 32 次、两个随机种子；当前 0–2/32 的高难 gap 和 0/32 的高难 hurdle 说明目标仍未完成。下一轮实验应优先提升跨格支撑恢复与横向控制，再用相同协议、多个种子以及完整地形混合验证。
