# A1 MP_full 推理阶段 forward/hold：首轮评估（2026-11-02）

## 实现范围

冻结 checkpoint `1101_11-11-00_A1MPfull-gait-balanced-v1`；外感知保持 10 Hz 缓存扫描。推理的 gait、modulation actor 仅在原有 15/3 步候选时刻决定 forward 或保持上次 action 与 reservoir 状态。两个 actor 的最大连续跳过次数分别可配，首轮为 2/3。episode 首步、命令变化、姿态或角速度异常、已见障碍/未知区域进入安全提前量、达到跳过上限时强制 forward。前向走廊半宽 0.2 m，地形高度偏差阈值 0.05 m；只读取已缓存的最低/最高高度和已知掩码。`training_rollout` 没有规则钩子，仍以固定频率更新。播放和评估均需显式开启；默认关闭。

## 配对地形结果

RTX 4090 / Isaac Gym。两组使用相同 checkpoint 和源码哈希、种子 321/322、地形种子 3、六种地形、难度 0.3/0.8、速度 0.8/1.5/2.2 m/s、每组 4 次、每次至多 12 s，共 288 个配对初始 episode。每个种子的初始 root、关节位置和观测逐元素一致。两组关闭噪声、推力和域随机化。

| 地形 | 10 Hz 固定更新 | 10 Hz forward/hold |
| --- | ---: | ---: |
| flat | 48/48 | 48/48 |
| stairs | 48/48 | 48/48 |
| hurdle | 47/48 | 48/48 |
| gap_single | 48/48 | 48/48 |
| gap_continuous | 47/48 | 47/48 |
| pit | 47/48 | 45/48 |
| **合计** | **285/288** | **284/288** |

规则组在 hurdle 增加 1 次成功，在 pit 有 3 次成功变失败、1 次失败变成功；合计净少 1 次成功。新增失败集中在 pit，不能把这个参数组合视为保持原越障能力。逐 episode 等权平均绝对速度误差为 0.575→0.583 m/s，增加 0.008 m/s。相同配置下 GPU 仿真存在轻微轨迹差异，表中差值不能解释为广泛泛化结论。触发原因计数允许重叠，例如同一步可同时由角速度与障碍触发。

| 指标（288 次 episode 合计） | 固定更新 | forward/hold |
| --- | ---: | ---: |
| gait 有效 forward / 候选 | 2765/2765 | 2290/2762 |
| modulation 有效 forward / 候选 | 13245/13245 | 10596/13243 |
| gait 实际输入环境数（含已结束 episode 的继续仿真） | 4942 | 4011 |
| modulation 实际输入环境数（同上） | 24086 | 18396 |
| gait 非空 actor 调用批次 | 170 | 179 |
| modulation 非空 actor 调用批次 | 321 | 371 |
| 两种子仿真墙钟总和 | 47.657 s | 48.393 s |
| 墙钟 / 模拟环境步 | 95.31 ms | 96.79 ms |

有效 forward 环境数分别减少约 17.2% 和 20.0%，但批量运行时不同环境的 episode 时钟错开，两个 actor 的非空网络调用批次都增加。两组各运行 500 个模拟环境步；规则组墙钟多 0.736 s，约慢 1.5%。这组批量仿真没有显示端到端加速。仿真耗时包含物理、扫描、规则和网络，不能单独代表真机推理延迟。
两组视觉帧总数分别为 8065 与 8061；episode 等权平均帧龄均约 2.00 步，最大 4 步，未知点比例约 15.4%。视觉频率没有因跳过 actor 而变化。

单机器人 flat、目标速度 1.5 m/s 的两组种子中，gait forward 从每次 8 次降到 3 次，modulation 从 40→22（种子 321）和 38→18（种子 322）。配对 rollout 墙钟分别为 10.038→9.859 s 与 9.515→9.364 s；episode 步数分别为 120→119 与 114→112。按模拟步归一后，两种子约为 83.65→82.85 ms/步与 83.47→83.61 ms/步，尚无稳定的单机端到端延迟收益。

独立的单机器人 GPU 组件基准在固定观测上预热 40 次，再进行 5 组各 300 次调用，每组前后同步 GPU，取中位数：gait actor 1.230 ms/次、modulation actor 0.890 ms/次、规则 0.554 ms/候选。把这些中位数乘以上述单机器人实际调用数，估计的总推理组件耗时分别为种子 321 的 45.46→49.89 ms、种子 322 的 43.68→45.22 ms。该估算不含 runner 索引/环境开销，说明当前逐候选扫描整张 33×7 缓存的规则成本足以抵消一部分网络节省。下一版若追求延迟，应在每次 10 Hz 视觉帧到来时提取已见障碍与未知边界，帧间只用里程估算剩余距离，并重新测量规则开销与 pit 稳定性。

## 本体触发阈值放宽试验

同一 checkpoint、10 Hz 扫描和地形矩阵下，种子 321/322 的 288 次 episode 结果如下。表中的平均 Hz 是实际有效 forward 总数除以有效环境仿真时间。只调整推理规则，训练时钟不变。

| 推理参数 | 成功 | gait forward / 平均 Hz | modulation forward / 平均 Hz | pit 成功 |
| --- | ---: | ---: | ---: | ---: |
| 原阈值：倾斜 0.25、水平角速度 1.0、yaw 1.5 | 284/288 | 2290 / 2.91 | 10596 / 13.44 | 45/48 |
| 只将水平角速度阈值改为 1.15 rad/s | 285/288 | 2204 / 2.78 | 10219 / 12.91 | 46/48 |
| 只将水平角速度阈值改为 1.25 rad/s | 284/288 | 2168 / 2.72 | 10019 / 12.55 | 44/48 |
| 倾斜 0.30、水平角速度 1.25、yaw 1.75 | 282/288 | 2157 / 2.71 | 9969 / 12.51 | 42/48 |
| 倾斜 0.35、水平角速度 1.5、yaw 2.0 | 282/288 | 2080 / 2.60 | 9484 / 11.88 | 44/48 |

水平角速度是本体规则的主要触发项；同时放宽多个阈值虽然进一步降频，但 pit 成功率下降。只把最大连续跳过次数改为 gait 4、modulation 6，在种子 321 下 forward 从 1165/5317 略增至 1181/5325：当时的实现会让更长的最大保持时间扩大地形安全视距，使未知区域更早强制 forward。这一耦合现已修正，见下文。

未参与调参的种子 323 上，原阈值为 144/144、gait 1167 次、modulation 5330 次；水平角速度 1.15 rad/s 为 143/144、gait 1130 次、modulation 5185 次，少一次 pit 成功。因此本节实验结束时默认阈值保持 1.0 rad/s；1.15 仅作为当时的可选降频参数，不能据此宣称越障能力不变或端到端推理更快。当前默认值见下文。

播放时可显式传入 `--adaptive_forward --horizontal_rate_limit 1.15`；评估脚本对应 `--adaptive-forward --scan-period-steps 5 --horizontal-rate-limit 1.15`。本节实验时未传参使用 1.0 rad/s；当前默认值见下文。

## 地形视距与跳过上限诊断

进一步诊断种子 321 的 144 个 episode：原规则的地形条件单独强制 gait 121 次、modulation 425 次，其中已知障碍分别占 121/423 次，未知边界单独触发仅 0/1 次。因此放宽未知边界几乎不能降低频率；地形条件的可调空间主要在已见障碍的提前量，需要警惕动作开始过晚。

在相同种子和配置上，试验性修改的有效 forward 次数如下；原规则为 gait 1165、modulation 5317、成功 141/144。GPU PhysX 重跑有少量轨迹波动，成功数的单次变化不能解释为可靠收益。

| 试验 | gait forward | modulation forward | 成功 |
| --- | ---: | ---: | ---: |
| 响应余量从 0.10 s 改为 0.05 s | 1157 | 5280 | 141/144 |
| 仅去掉视距中的视觉周期和帧龄附加项 | 1161 | 5221 | 142/144 |
| 去掉上述附加项，并将最大跳过次数改为 gait 4、modulation 6 | 1159 | 5220 | 142/144 |

后两项通过运行时临时替换规则入参进行诊断，报告中的源码哈希不包括该替换；它们不是正式候选配置。缓存每步重投影旧静态扫描帧，因此在完美位姿和静态障碍假设下，视觉帧龄不直接造成几何距离滞后；真实相机的时延、遮挡和里程误差仍需要余量。即使去掉该附加项，单种子 modulation forward 只减少约 1.8%，gait 只减少约 0.3%。更大的频率空间可能需要重新定义正常步态下的本体触发，例如对角速度加入短时持续性或分级阈值，同时让异常姿态和近障保持立即 forward；这需要独立验证 pit、hurdle 和 gap。

## 普通本体晃动滤波试验

推理开关 `--filter_normal_sway`（评估脚本为 `--filter-normal-sway`）让两个 actor 共用每个控制步更新的水平角速度向量指数平滑，系数 0.5。本节实验当时的阈值为：平滑后的模长超过 1.0 rad/s 时 forward；瞬时模长超过 2.0 rad/s 时立即 forward；瞬时模长超过 1.0 rad/s 且投影重力的水平模长超过 0.1 时也立即 forward。当时的 0.25 倾斜、1.5 yaw、命令、近障、未知和跳过上限规则仍生效；episode 重置时清空滤波历史。训练路径未接入滤波，默认推理也不启用。当前已移除 yaw 触发，水平角速度阈值见下文。

未加 0.1 倾斜保护的首版滤波在种子 321/322 上仅成功 278/288，说明直接平滑会错过越障所需更新。加入姿态保护后的三组配对种子 321/322/323 共 432 个初始 episode，checkpoint、初始 root/关节状态和 10 Hz 扫描配置逐对一致：

| 指标 | 原推理规则 | 晃动滤波加姿态保护 |
| --- | ---: | ---: |
| 成功 | 428/432 | 428/432 |
| gait 实际 forward / 平均 Hz | 3457 / 2.92 | 3218 / 2.69 |
| modulation 实际 forward / 平均 Hz | 15926 / 13.44 | 14467 / 12.09 |
| hurdle 成功 | 72/72 | 71/72 |
| gap_continuous 成功 | 71/72 | 71/72 |
| pit 成功 | 69/72 | 70/72 |
| 批量仿真墙钟 / 模拟步 | 96.00 ms | 96.23 ms |

滤波使 gait/modulation 的平均有效更新频率约降 7.8%/10.0%，但地形成败有互换；总成功数持平不等于安全性已证明。单机器人 flat 种子 321 的 modulation forward 为 22→15，每步墙钟 87.96→85.40 ms；种子 322 为 18→17，每步墙钟 85.71→86.20 ms，没有稳定的端到端加速。单机器人 GPU 微基准中，滤波额外增加约 0.04 ms/候选判断和 0.04 ms/控制步；批量仿真仍受异步 actor 批次、物理和扫描开销影响。此模式保持可选；真机视觉误差和更多地形种子尚未验证。

## 连续策略独立本体阈值试验

新增可选参数 `--modulation_horizontal_rate_limit`（播放）和 `--modulation-horizontal-rate-limit`（评估），仅覆盖 modulation 的水平角速度阈值；本节实验当时不传参数会沿用 `--horizontal_rate_limit` 的值。当前两者各有默认阈值，见下文。此参数仅在启用 adaptive forward/hold 的推理中使用。

三组配对种子 321/322/323、共 432 个 episode，10 Hz 扫描、相同 checkpoint 和每种子的初始状态哈希；使用 modulation 阈值 4 rad/s，与原推理规则的 1 rad/s 对比：

| 指标 | 原推理规则 | modulation 4 rad/s |
| --- | ---: | ---: |
| 总通过 | 428/432 | 423/432 |
| gait forward / 平均频率 | 3457 / 2.92 Hz | 3513 / 2.93 Hz |
| modulation forward / 平均频率 | 15926 / 13.44 Hz | 11958 / 9.97 Hz |
| flat 的 modulation 平均频率 | 9.18 Hz | 4.95 Hz |
| gap_continuous 通过 | 71/72 | 70/72 |
| pit 通过 | 69/72 | 65/72 |
| episode 等权平均速度绝对误差 | 0.581 m/s | 0.591 m/s |
| 批量仿真墙钟 / 模拟步 | 96.00 ms | 96.52 ms |

modulation 有效 forward 数降低约 25%，但 432 个 episode 中多失败 5 个（约 1.2 个百分点），主要集中在 pit。批量仿真中的非空 modulation 网络调用批次为 533→924；降频没有带来端到端加速。各组轨迹长度变化使总模拟步数从 761 增至 1181，故比较墙钟时应按模拟步归一。真实单机推理时的耗时仍需单独测量。

单机器人 flat、1.5 m/s 的种子 321/322 配对初始状态一致，modulation forward 分别从 22→13、18→10；墙钟每模拟步分别为 82.85→82.76、83.61→81.03 ms。两个短轨迹的差异不足以证明稳定的单机端到端加速。

将已见障碍触发完全屏蔽的诊断在种子 321 上只通过 126/144；每隔一个 modulation 候选时刻复核近障，种子 321/322 合计通过 271/288，原规则为 284/288。两种诊断都通过运行时临时替换规则完成，不属于正式配置，也不应按其报告源码哈希复现。它们表明在现有固定训练频率的 checkpoint 下，直接把全地形平均频率压向 5 Hz 会明显损害连续沟壑和 pit。因此近障仍在每个候选时刻复核。本节实验时 4 rad/s 是可选阈值，默认值仍为 1 rad/s；当前默认值见下文。

在种子 321 上同时开启原有普通晃动滤波与 modulation 4 rad/s 阈值，modulation 仍为 9.99 Hz、通过 140/144，因此没有进一步降频收益。

播放使用 `play.py --task MP_full_a1 --adaptive_forward --modulation_horizontal_rate_limit 4`；评估使用 `evaluate_mp_full_gaits.py --adaptive-forward --scan-period-steps 5 --modulation-horizontal-rate-limit 4`，并补全 checkpoint、输出路径和地形矩阵等必需参数。原始报告位于 `modulation_rate4_321`、`modulation_rate4_322`、`modulation_rate4_323`。

## 几何边缘过滤平滑地形

几何边缘规则（现默认启用，播放兼容参数 `--extero_edge_trigger`，评估兼容参数 `--extero-edge-trigger`）使用已缓存的 10 Hz 高度扫描：在前方中心 ±0.1 m 走廊取每一行低/高高度中点的中位数，仅当相邻扫描行的高度跳变至少 0.035 m，且原有 0.05 m 高度异常进入速度相关的安全视距时，已见障碍才触发 forward。播放可用 `--terrain_edge_height_m` 调阈值，评估脚本为 `--terrain-edge-height-m`。未知区域、无效扫描、本体异常、命令变化和跳过上限仍立即按原规则处理。规则只读高度几何和已知掩码，不读取仿真地形类别；平滑坡面与小幅起伏不再仅因相对近处的累计高度偏差而反复触发。坡道末端若真有高度突变，仍会按障碍处理。

与上一节的 modulation 4 rad/s 配置配对比较，种子 321/322、七种地形、两档难度、三档速度、每格四次，共 336 个初始 episode；两组每个种子的初始状态哈希相同。此比较单独检验边缘过滤的增量效果：

| 指标 | 原高度偏差触发 | 加几何边缘过滤 |
| --- | ---: | ---: |
| 总通过 | 327/336 | 327/336 |
| modulation forward / 平均频率 | 10033 / 10.27 Hz | 9014 / 9.11 Hz |
| flat 平均频率 | 4.81 Hz | 4.81 Hz |
| rough_slope 平均频率 | 8.71 Hz | 5.68 Hz |
| ramp 平均频率 | 11.41 Hz | 6.56 Hz |
| stairs 通过 | 48/48 | 47/48 |
| hurdle 通过 | 47/48 | 47/48 |
| gap_continuous 通过 | 47/48 | 47/48 |
| pit 通过 | 41/48 | 42/48 |
| 批量仿真墙钟 / 模拟步 | 97.57 ms | 97.05 ms |

总成功数相同，但 stairs/pit 有一次互换；两种子不足以证明安全性或速度稳定改善。rough_slope 的 modulation 地形触发计数从 666 降到 95，ramp 从 1197 降到 198；剩余计数包含未知区域与真正的边缘。单独的 gap_single 配对试验（种子 321，24 个 episode）两组均为 24/24、412 次 modulation forward。原始报告位于 `edge_baseline_321/322`、`edge_filter_321/322`、`edge_gap_single_base_321`、`edge_gap_single_filter_321`。该几何规则在当时默认关闭，现已默认启用；在真实深度相机上需要先得到带已知掩码的局部高度图，不能直接对原始深度像素套用 0.035 m 阈值。

本节实验的显式播放参数：`play.py --task MP_full_a1 --adaptive_forward --modulation_horizontal_rate_limit 4 --extero_edge_trigger`。评估对应 `evaluate_mp_full_gaits.py --adaptive-forward --scan-period-steps 5 --modulation-horizontal-rate-limit 4 --extero-edge-trigger`，并补齐 checkpoint、输出路径和地形参数。当前这些阈值参数可省略。

## 当前推理默认规则（2026-11-03）

启用 `--adaptive_forward` 后，gait 的水平角速度模长触发阈值为 **2.0 rad/s**，modulation 为 **4.0 rad/s**，几何边缘触发默认开启。偏航角速度不再单独触发 forward；角速度出现非有限值仍触发无效观测保护。其余倾斜阈值 0.25、命令变化、近障/未知区域、最大连续跳过次数（gait 2、modulation 3）保持不变。若使用 `--filter_normal_sway`，水平角速度滤波和瞬时保护仍按各 actor 的阈值执行。仅推理阶段使用这些规则，训练继续按固定周期更新。

播放只需 `play.py --task MP_full_a1 --adaptive_forward`；评估只需 `evaluate_mp_full_gaits.py --adaptive-forward --scan-period-steps 5`，再补齐必要的 checkpoint 和地形参数。关闭几何边缘过滤可用播放参数 `--no_extero_edge_trigger` 或评估参数 `--no-extero-edge-trigger`。仍可显式指定两个角速度阈值。

使用 checkpoint `1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt`，种子 321/322、七种地形、两档难度、三档速度、每格四次，共 336 个首 episode。每个种子的配对初始状态一致。旧规则为 gait 1.0、modulation 4.0 rad/s、yaw 1.5 rad/s、几何边缘开启；新规则移除 yaw 并将 gait 提到 2.0 rad/s。新规则先用极高的 yaw 阈值作等效诊断，再以代码删除 yaw 条件并按默认参数复测。下表新规则采用代码修改后的复测结果。

| 指标 | 旧规则 | 新规则 |
| --- | ---: | ---: |
| 越障成功 | 327/336 | 328/336 |
| gait 平均 forward 频率 | 2.83 Hz | 2.29 Hz |
| modulation 平均 forward 频率 | 9.11 Hz | 8.92 Hz |
| gait 实际切换次数 | 798 | 640 |
| gait 实际切换频率 | 0.81 Hz | 0.66 Hz |

频率按各环境活跃仿真时间加权。切换次数只计首步之后 policy 输出的 gait ID 变化，和 forward 次数不同。样本量有限，不能据此断言成功率有统计改善；下降的更新与切换频率在两颗种子上均出现。等效诊断为 330/336、gait 2.30 Hz、630 次切换，与代码修改后的复测有轻微波动。批量仿真墙钟时间受最慢 episode 和运行步数影响，本组数据不证明端到端加速。原始对照报告位于 `yaw_study_base_321/322` 与 `yaw_study_rate20_321/322`；代码修改后的默认参数复测位于 `defaults_final_321/322`。

单独的 `gap_single` 配对样本（种子 321，24 个 episode）两组均通过 24/24；gait 平均频率由 3.03 降至 2.44 Hz。新报告位于 `defaults_gap_single_321`，旧报告位于 `edge_gap_single_filter_321`。

## 大跳过上限的地形视距修正

原规则直接用“剩余可跳过候选次数 × actor 候选周期”计算地形安全视距。使用 `--gait_max_skips 15 --modulation_max_skips 75` 时，两者刚开始都对应约 4.5 s；即使在平地，0.8 m/s 时预测距离也超过 3.3 m 的扫描范围。未知区域保护因而在每个候选时刻强制 forward，使大上限完全失效。种子 321 的 24 个 flat episode 中，修正前 gait/modulation 的每个候选时刻都由 `terrain` 触发，平均频率为 3.47/16.77 Hz。

现在仅在**地形安全视距**中把剩余跳过次数限制为 gait 2、modulation 3；实际允许的连续跳过次数仍分别使用命令行的 15/75。每个候选时刻仍重新检查已见障碍、未知区域和本体状态。视距还包括 10 Hz 视觉周期、当前帧龄、0.10 s 响应时间和 0.15 m 距离余量。默认 2/3 跳过上限的原有视距不变。平地同批 24 个 episode 仍通过 24/24，gait/modulation 频率降为 0.51/1.37 Hz，gait 实际切换从 47 次降至 2 次；临近下一地形时仍可能触发更新。

七地形、种子 321/322 共 336 个首 episode 的 15/75 设置为 326/336，通过数比默认 2/3 设置的 328/336 少 2 个；gait/modulation 平均频率分别为 2.06/7.21 Hz，默认设置为 2.29/8.92 Hz。单独的 `gap_single` 种子 321 样本两组均为 24/24。大上限不保证越障表现更好；这组仿真也不能证明端到端墙钟加速。报告位于 `high_skips_flat_before_321`、`high_skips_flat_capped_321`、`high_skips_terrain_capped_321/322`、`high_skips_gap_single_capped_321`。

## 每个 episode 的推理频率（2026-11-03）

`play.py` 现在在每个 `MP_full_a1` episode 结束时，按机器人打印 gait 和 modulation 的实际 forward 次数除以该 episode 的仿真时长；最后一步计入刚结束的 episode，随后分别清零计数。多机器人同时或交错结束会各自打印。按 Ctrl+C 退出时仍打印整段播放（含尚未结束的 episode）的汇总。评估脚本也在每个首 episode 完成时打印同一口径；到评估时长上限仍未完成的 episode 标为 `evaluation cutoff`，并打印截至截断时的频率。训练路径不使用这些计数或打印。

## 早期实验命令

早期原始报告在 `seed321_fixed`、`seed321_adaptive`、`seed322_fixed`、`seed322_adaptive`，每个目录包含 `policy.json` 与 `policy_initial.npz`。`smoke*` 为开发过程中的小规模试验。以下命令现按**当前**默认规则运行，不能逐项重现早期含 yaw 触发的历史结果。

```bash
for seed in 321 322; do
  for mode in fixed adaptive; do
    adaptive_arg=()
    if [ "$mode" = adaptive ]; then adaptive_arg=(--adaptive-forward); fi
    PATH=/home/ubuntu/anaconda3/envs/cpgil/bin:$PATH \
      /home/ubuntu/anaconda3/envs/cpgil/bin/python \
      legged_gym/legged_gym/scripts/evaluate_mp_full_gaits.py \
      --checkpoint legged_gym/logs/A1/1101_11-11-00_A1MPfull-gait-balanced-v1/model_1001_10000.pt \
      --output docs/terrain/a1_inference_forward_hold/seed${seed}_${mode} \
      --gait policy --terrains flat stairs hurdle gap_single gap_continuous pit \
      --difficulties 0.3 0.8 --speeds 0.8 1.5 2.2 \
      --repeats 4 --seconds 12 --seed "$seed" --terrain-seed 3 \
      --scan-period-steps 5 "${adaptive_arg[@]}"
  done
done
PATH=/home/ubuntu/anaconda3/envs/cpgil/bin:$PATH \
  /home/ubuntu/anaconda3/envs/cpgil/bin/python \
  docs/terrain/a1_inference_forward_hold/benchmark_components.py
```

播放入口通过 `play.py --task MP_full_a1 --adaptive_forward` 开启，并可用 `--gait_max_skips` 与 `--modulation_max_skips` 调整连续跳过上限。规则仅经受理想化静态高度扫描仿真；未验证真实深度相机遮挡、噪声、视觉延迟和状态估计误差。

## 验证状态

当前相关 pytest 共 33 个测试通过，包括推理跳过时的 action/reservoir 保持、训练路径仍调用全部固定候选、未知区域边界、大跳过上限的平地保持与近障触发、两 actor 独立上限、默认 CLI 参数和播放开关。现有 `test_mp_full_training_contract.py` 仍因工作区已删除的 `mp_proprio_terrain_config.py` 无法收集，未计入通过数。
