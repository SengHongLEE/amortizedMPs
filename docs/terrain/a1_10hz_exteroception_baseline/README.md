# A1 MP_full：10 Hz 外感知基线（2026-11-02）

## 范围

在同一冻结 checkpoint 上，比较每个 20 ms 环境步更新地形扫描与每 5 步（10 Hz）更新一次扫描。两种模式都保留 gait 每 15 步、modulation 每 3 步的原始 actor 时钟；本次没有实施 `forward / hold` 决策规则。10 Hz 模式仅在 reset 和到期步读取静态地形，其余步按采样时与当前的平面位姿重投影缓存的 33×7 个最低/最高高度，再用当前机身高度生成原有 160 维 `signed_dense` 观测。重投影采用相邻采样点的最低/最高值保留窄沟和凸起；超出旧帧范围时使用边界值，并单独记录为未知。

## 配对评估

Checkpoint：`legged_gym/logs/A1/1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt`。RTX 4090 / Isaac Gym；种子 321、322；地形种子 3；`flat`、`stairs`、`hurdle`、`gap_single`、`gap_continuous`、`pit`；难度 0.3、0.8；速度 0.8、1.5、2.2 m/s；每组 4 次；每次最多 12 s。共 288 个配对初始 episode。两个模式在每个种子上的 checkpoint 与源码哈希、初始状态哈希一致，初始观测逐元素相同。关闭噪声、随机推力、摩擦与质量随机化。成功定义为在 1.5 m 走廊内前进 3.4 m 且未触发失败。

| 地形 | 每步扫描 | 10 Hz 扫描 |
| --- | ---: | ---: |
| flat | 48/48 | 48/48 |
| stairs | 48/48 | 48/48 |
| hurdle | 48/48 | 47/48 |
| gap_single | 48/48 | 48/48 |
| gap_continuous | 48/48 | 47/48 |
| pit | 46/48 | 48/48 |
| **合计** | **286/288** | **286/288** |

10 Hz 模式有 2 次原先失败变为成功，同时有 2 次原先成功变为失败。新增失败分别为种子 321、难度 0.8、速度 2.2 m/s 的 hurdle，以及种子 322、难度 0.8、速度 0.8 m/s 的 gap_continuous。因此总通过数相同并不表示逐场景行为相同。两模式各有 2 次失败、0 次超时。逐 episode 的平均绝对速度误差等权平均为 0.556→0.576 m/s，增加约 0.020 m/s。相同配置下重跑每步扫描时，episode 结束时间有轻微差异，说明此 GPU 仿真不能视作逐时间步完全可复现；这组结果只支持所列种子和轨迹的比较。

扫描帧总数为 40,300→8,082，减少约 79.9%；10 Hz 模式的 episode 等权平均帧龄为 2.00 个环境步，最大 4 步，未知点比例均值为 15.4%。gait/modulation 的**调用周期没有变化**；总调用数 2802/13433→2770/13266，差别来自各轨迹结束时间不同。此次只是降低外感知采样率，没有跳过 actor forward；也没有测得推理耗时下降。

## 提前量限制

扫描最远前向点为 3.3 m。当前 8 m 地形块从块中心附近出生，出生 XY 抖动为 ±0.1 m。按地形几何，首个前方 hurdle、单沟和连续沟分别约在出生点前方 2.0、3.0、0.7 m。以最高目标速度 2.2 m/s 作简单除法，连续沟首个前缘的名义接近时间约为 0.27–0.36 s；reset 虽然立即取首帧，最多 0.30 s 的 gait 决策等待已经几乎用尽这个窗口，还未计入控制响应。对后续才进入视野的障碍，另需预算最多 0.10 s 等帧时间。这里是几何与目标速度的预算检查，并非逐轨迹测得的首次有效视觉观测、实际接触时刻或控制响应延迟；不能据此证明 10 Hz 足以支持所有提前切换。

## 复现

在仓库根目录运行；每个输出目录含 `policy.json` 和 `policy_initial.npz`。JSON 记录 checkpoint/源码哈希、完整有效配置和逐 episode 指标。
本报告使用 `seed*_fresh` 与 `seed*_conservative`；目录中的 `seed*_10hz` 是早期双线性重投影试验，不计入上表。

```bash
for seed in 321 322; do
  for period in 1 5; do
    if [ "$period" = 1 ]; then mode=fresh; else mode=conservative; fi
    PATH=/home/ubuntu/anaconda3/envs/cpgil/bin:$PATH \
      /home/ubuntu/anaconda3/envs/cpgil/bin/python \
      legged_gym/legged_gym/scripts/evaluate_mp_full_gaits.py \
      --checkpoint legged_gym/logs/A1/1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt \
      --output docs/terrain/a1_10hz_exteroception_baseline/seed${seed}_${mode} \
      --gait policy --terrains flat stairs hurdle gap_single gap_continuous pit \
      --difficulties 0.3 0.8 --speeds 0.8 1.5 2.2 \
      --repeats 4 --seconds 12 --seed "$seed" --terrain-seed 3 \
      --scan-period-steps "$period"
  done
done
```

此实验使用理想化的静态高度扫描；没有模拟深度相机的遮挡、视角、定位误差、采集延迟或噪声。10 Hz 基线只支持后续显式规则的归因评测，不代表真机视觉链路已经验证。

## 验证状态

`test_temporal_scan.py`、`test_mp_full_scan_clock.py`、`test_signed_dense_scan.py` 合计 12 个测试通过；修改的 Python 文件通过语法编译，`git diff --check` 无空白错误。两个种子的 `policy_initial.npz` 中 root、关节位置及初始观测均逐元素一致。现有 `test_mp_full_training_contract.py` 无法收集：其依赖的 `mp_proprio_terrain_config.py` 在当前工作区已被删除；本次未恢复该无关文件，因此不把这项测试计为通过。
