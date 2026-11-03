# MP_full_a1 接触触发 jump reflex 小样本验证（2026-10-26）

## 问题与实验边界

验证一个短时动作序列能否由足端接触力与后腿相位触发，并改善 gap、hurdle 通过率。实验独立运行；`train.py`、默认任务配置、已有评测脚本均没有接入该 reflex。

使用已有五原语、18 维连续动作 checkpoint `model_31_320.pt`。它没有 jump 选择输出，因此实验在已知障碍前沿的固定距离**预先挂起** reflex（oracle arm）；真正开始预压仍由足端力和后腿支撑相位决定。这是触发与动作序列的上界探针，不代表已学会第六原语或无需感知障碍位置。

反射状态为 `ARMED → PRELOAD → PUSH → FLIGHT → RECOVER`，每个 episode 最多执行一次。触发条件为至少两足受力大于 5 N，且一条后腿受力并处于指定支撑相位；预压 55 ms，足端目标抬高 3 cm，蹬伸约 105 ms，目标下探 4.5 cm，随后按四足接触信号切换收腿和落地。这里的 `FLIGHT` 事件仅表示四足力低于阈值，不能单独证明形成了有用的竖直动量。

## 配对设置

| 设置 | 值 |
| --- | --- |
| checkpoint | `legged_gym/logs/A1/1026_14-00-13_A1MPfull-v3-18D-psF_3.333e+00-mpF_1.667e+01-MF_1e+03-noResume/model_31_320.pt` |
| 地形 | `gap_single`、`gap_continuous`、`hurdle` |
| 难度与指令速度 | 0.5；1.2 m/s |
| 样本数 | 每地形 8 个环境，共 24 个；每次仅测第一次障碍带 |
| 时长与随机种子 | 5 s；seed 123；确定性 actor |
| 挂起距离 | 障碍前沿前 0.20、0.35、0.55 m |
| 初始状态 SHA256 | 四组均为 `907642989b1a7664e771857037eaa599cc5ce0842322a080754e6e96363331d9` |

## 结果

| 地形 | 基线 | 0.20 m | 0.35 m | 0.55 m |
| --- | ---: | ---: | ---: | ---: |
| 单次 gap | 4/8 | 5/8 | 4/8 | 4/8 |
| 连续 gap | 4/8 | 3/8 | 2/8 | 3/8 |
| hurdle | 3/8 | 0/8 | 3/8 | 1/8 |
| 合计 | **11/24** | **8/24** | **9/24** | **8/24** |

0.20、0.35、0.55 m 三组分别有 17、20、21 个环境进入预压，16、17、20 个环境记录到四足无接触事件；24 个环境均被挂起。按同一环境逐个比较，0.20 m 的单次 gap 救回 1 例，但 hurdle 损失基线成功的 3 例；0.35 m 的 hurdle 救回 2 例、也损失 2 例。所有设置都没有带来整体收益。

起跳时机确实随挂起距离变化：连续 gap 的第一次四足无接触位置相对障碍前沿中位数约为 +0.06、−0.08、−0.21 m（分别对应 0.20、0.35、0.55 m；正值表示已经进入障碍）。0.20 m 挂起过晚；提前到 0.55 m 后，hurdle 成功率仍降至 1/8。抽样轨迹中，hurdle 环境 16 在 0.55 m 组的障碍附近世界坐标竖直速度峰值约为 1.13 m/s，基线约为 0.75 m/s，但这个样本依然未通过。因此单纯产生向上的速度或四足无接触并不足以完成越障。

## 结论与限制

接触/相位触发的一次性预压—蹬伸—收腿序列在仿真中**可以执行**，但当前参数与旧策略结合时**未提高**这批地形的通过率，连续 gap 和 hurdle 更容易退化。原因不能仅归结于挂起过早或过晚：三档扫描均未提供稳定收益。下一轮若要验证训练价值，应让策略输出“选择 jump/准备跳”，并在训练中加入该状态机；同时测量起跳前后 COM 竖直冲量、障碍前沿处姿态与落地点，并针对连续 gap 设计再触发或跨格衔接。当前 8 例/地形、单个 seed 是诊断性小样本，不能据此断言这种架构永远无效。

## 复现

在仓库根目录、带 Isaac Gym 的 `cpgil` 环境中运行：

```bash
export PATH=/home/ubuntu/anaconda3/envs/cpgil/bin:$PATH
PY=/home/ubuntu/anaconda3/envs/cpgil/bin/python
CKPT=legged_gym/logs/A1/1026_14-00-13_A1MPfull-v3-18D-psF_3.333e+00-mpF_1.667e+01-MF_1e+03-noResume/model_31_320.pt
for MODE_DIST in baseline:0.35 reflex:0.20 reflex:0.35 reflex:0.55; do
  MODE=${MODE_DIST%%:*}
  DIST=${MODE_DIST#*:}
  "$PY" legged_gym/legged_gym/scripts/probe_contact_jump_reflex.py \
    --jump-mode "$MODE" --arm-distance "$DIST" --checkpoint "$CKPT" \
    --terrains gap_single gap_continuous hurdle --difficulties 0.5 \
    --speeds 1.2 --repeats 4 --seconds 5 --seed 123 \
    --hazard-target first_band --trace-env-ids 0 16 \
    --output "docs/terrain/contact_jump_probe_${MODE}_${DIST}"
done
```

本次保存的原始报告与轨迹位于 `docs/terrain/contact_jump_probe_{baseline_24,reflex020_24,reflex035_24,reflex055_24}/`。上面的命令写入新目录，以避免覆盖这些原始文件。
