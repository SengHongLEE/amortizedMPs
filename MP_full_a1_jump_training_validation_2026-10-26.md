# MP_full_a1 跳跃原语训练验证（2026-10-26）

## 目标与实验设置

在不改动默认 `MP_full_a1`、`train.py` 和原有评测入口的情况下，使用独立任务 `MP_full_jump_a1` 把索引 4 的周期性 pronk 换成接触触发的单次 jump。前四个原语仍为 walk、trot、canter、bound；总动作维度保持 19（1 个离散选择、18 个连续调制）。jump 的预压、蹬伸、腾空、恢复由接触力和后腿相位状态机执行，是否选择 jump 由离散策略决定。

主实验从新参数开始，以 seed 1、4096 环境训练。与原 pronk 基线比较时，使用同一个评测种子、初始状态、速度与地形；每个地形含两列、两个难度（0.5、0.75），每列重复 8 次，因此每地形 32 次，共 96 次。指令速度为 1.2 m/s。首次障碍带时限 5 s，第二障碍带时限 10 s；后者的 5 s 对 hurdle 明显过短，不用于主比较。所有通过数均来自确定性 actor 的完整环境评测。

## 训练与通过率

| 策略 | 单次 gap，首次带 | 连续 gap，首次带 | hurdle，首次带 | jump 事件 |
| --- | ---: | ---: | ---: | ---: |
| 原 pronk，第 310 轮 | 28/32 | 26/32 | 23/32 | 不适用 |
| jump，第 50 轮 | 15/32 | 9/32 | 20/32 | 0 |
| jump，第 100 轮 | 20/32 | 15/32 | 20/32 | 0 |
| jump，第 150 轮 | 22/32 | 24/32 | 21/32 | 0 |

以上评测的初始状态 SHA256 均为 `59d4e1e2cb1e6ce98fb8c450d7dad288b87f4c12c092f2f1629c54d3424ce404`。第 150 轮的 96 个环境均全程只输出 canter，既没有 jump 选择，也没有预压或四足腾空事件。第 150 轮在三类障碍上仍未超过原 checkpoint。

原 pronk 基线的第二障碍带通过数为单次 gap 28/32、连续 gap 19/32、hurdle 19/32（seed 123，10 s）。seed 321 的同设置基线分别为 28/32、18/32、23/32。第 150 轮 jump 训练未做第二带比较，因为策略完全没有使用 jump；这一项由探索系数对照实验继续检验。

## 为什么总选 canter

训练时离散 actor 使用 `Categorical(logits)` 随机采样，动作会写入环境的离散 action；评测时使用 logits 的 `argmax`。原语奖励、事件 transition 与 PPO 更新链路已逐项核对，没有发现参数更新后动作未送入环境的错误。状态机也在强制选择 jump 的 8 环境物理检查中 8/8 进入预压、腾空、落地，最大竖直速度约 1.54–1.95 m/s；但该小样本的障碍通过数为 0/8。

| Checkpoint 与真实状态样本 | canter 平均概率 | 第五原语平均概率 | 确定性选择 |
| --- | ---: | ---: | --- |
| jump 第 10 轮，704 次决策 | 41.56% | jump 3.86% | 全部 canter |
| jump 第 100 轮，305 次决策 | 99.01% | jump 0.016% | 全部 canter |
| jump 第 150 轮，690 次决策 | 99.70% | jump 0.0050% | 全部 canter |
| pronk 第 310 轮，415 次决策 | 99.96% | pronk 0.0014% | 全部 canter |

所以 canter 占优不是第 5 原语替换才产生的，也不只是 `argmax` 把一个很小的概率差放大：训练本身让随机采样分布逐渐集中到 canter。真实仿真中的 32 个 reservoir 脉冲特征有 30 个发生变化，第 100 轮观察到 175 种特征组合，不能把现象归因于 reservoir 完全无输出。现有读出层没有原始观测直达输入，熵系数为 0.001；这两项是可检验的探索限制，而非已经证实的单一根因。

canter 的步长系数为 1.5，其它常规步态为 1.0。对原 pronk 最终 checkpoint 只在评测时把 canter 系数改成 1.0，完全相同的 96 个初始状态中首次障碍带总通过数从 77/96 降到 1/96。这证明已训练的 locomotion 与 canter 长步幅高度耦合；由于没有重训，此消融不能单独证明最初的原语偏好为何形成。jump 初期随机选择仍有 3.86%，之后却迅速降低；一个合理但未单独验证的解释是：jump 当下中断前进，而越障奖励稀疏、失败惩罚立即出现，使策略在训练早期压低它的概率。

该系数直接乘脚端前后摆幅：`max_step_length=0.1 m`，canter 名义半摆幅 0.15 m，其他常规原语为 0.10 m。库中的 canter 基频是 8 Hz，bound 为 10 Hz；简单的“摆幅×频率”量级在原设置下约为 1.2 对 1.0，等步长后约为 0.8 对 1.0。这个量级比较忽略支撑占比、相位、后腿前摆限幅及连续策略调制，只用于解释为什么更改步长可能改变相对运动能力。

第 150 轮 jump 平均选择概率约 0.005%，且没有任何评测环境实际跳跃。继续训练这组配置无法有效回答 jump 能否提高通过率，因此主实验在保存第 150 轮 checkpoint 后早停；**本实验没有完成原计划的 310 轮**。

## 熵系数对照

第二组独立训练仅把离散 PPO 熵系数从 0.001 改成 0.02，其余 jump 控制器、地形、canter 步长、seed 与 4096 环境保持一致，完成 100 轮。覆盖只发生在该进程，默认配置和原有脚本均不变。

第 10 轮同场景概率诊断中，高熵组 jump 为 4.10%、canter 为 45.30%，原组分别为 3.86%、41.56%；早期尚未体现明显的 jump 探索增益。

第 50 轮同初始状态首次障碍带评测：高熵组单次 gap 21/32、连续 gap 8/32、hurdle 22/32，原组同轮次为 15/32、9/32、20/32；高熵组 96 环境仍全程选择 canter，jump 事件为 0。另在同设置 48 环境真实决策中，原组 canter/jump 平均概率为 97.45%/0.062%，高熵组为 72.06%/0.138%，两组全部 argmax=canter。高熵确实减缓概率集中，但目前尚未让部署策略使用 jump，无法把通过率差异归因于跳跃。

第 100 轮高熵组转而集中到 bound：48 环境真实决策中 bound/canter/jump 平均概率为 83.79%/9.77%/0.031%，确定性选择全部为 bound，jump 事件仍为 0。首次障碍带通过数为 3/32、0/32、13/32；第二障碍带为 3/32、0/32、7/32。提高熵系数延迟了模式坍缩，却没有让 jump 被学到，最终性能更差。

## canter 步长因果对照

第三组独立训练从零参数开始，仅把 canter 的步长系数从 1.5 改成 1.0，其他四种原语保持 1.0，离散 PPO 熵系数保持原来的 0.001。seed 1、4096 环境与 jump 任务均同主实验。这组实验专门区分“已训练策略依赖长步幅”和“长步幅造成训练偏选 canter”；在第 50 轮保存 checkpoint 后因单步态坍缩早停。

第 10 轮的同初始状态诊断中，原步长组 canter/bound/jump 平均概率分别为 41.56%/34.19%/3.86%，704 次确定性选择全部为 canter；等步长组分别为 25.71%/45.31%/3.52%，701 次确定性选择全部为 bound。两组初始状态 SHA256 都为 `3c1d9b31685a914c161229f115b265ddac2b48e9e7aa44d684e6afb203b0a1f6`。这支持 canter 长步幅对早期选择偏好有因果影响，但第 5 原语的概率基本没有增加。

第 50 轮的配对评测进一步确认这种转移：等步长组 96/96 环境都选 bound，单次 gap 0/32、连续 gap 0/32、hurdle 14/32，jump 事件为 0；原步长组同轮次全部选 canter，通过数为 15/32、9/32、20/32。等步长组另 48 环境的 bound/canter/jump 平均概率为 98.11%/1.13%/0.060%，而原组的 jump 为 0.062%。取消长步幅没有解除单步态坍缩，也没有促进 jump；在当前轮次下还使 gap 能力明显退化。

该组已在保存第 50 轮 checkpoint 后早停，未完成原计划的 100 轮。这个结论限于相同训练预算下的原语偏好与中期性能，不能据此断言等步长 bound 永远无法学会越 gap。

## jump 状态机的受控物理验证

新增独立探针 `legged_gym/legged_gym/scripts/probe_mp_full_jump_oracle.py`：在已知第一障碍边缘前的指定距离，仅把离散动作切到 jump 一个环境步，随后恢复已训练的常规策略；jump 状态机仍只由足端接触力和后腿相位启动与推进。该真值位置脉冲用于测试物理动作的能力上界，不代表部署时策略已学会选择或预测障碍。

难度 0.75、每地形 8 次的先导实验中，同 24 个初始状态的基线单次 gap/连续 gap/hurdle 为 3/8、2/8、6/8；在障碍边缘前 0.35 m 脉冲选择 jump 后分别为 0/8、1/8、2/8。24 次脉冲中只有 17 次进入预压、15 次四足腾空、14 次有落地事件。单次 gap 的多数预压发生在障碍边缘之后，hurdle 的多数落地发生在边缘前约 0.15 m。

96 环境配对扫描中的 0.15 m 组已完成：基线首次障碍带总通过 67/96，oracle jump 为 54/96；按地形分别从 22/32、24/32、21/32 变成 19/32、18/32、17/32。逐例看，jump 救回 4 个原本失败样本，同时损失 17 个原本成功样本。91/96 环境实际收到脉冲，77 次进入预压、64 次四足腾空、62 次落地。三类地形的腾空位置中位数均已越过障碍前沿，说明 0.15 m 太晚；该组不能用于断言更早触发也失败。

完整 96 环境的其它触发距离也没有超过 canter 基线：0.35 m 为 17/32、13/32、9/32；0.55 m 为 11/32、6/32、7/32；0.75 m 为 0/32、0/32、7/32。0.55 m 处单次与连续 gap 的腾空水平位移中位数仅约 0.22、0.26 m，hurdle 仅约 0.05 m。把 jump 离散选择保持 15 个控制步（实际策略选项周期）后，0.35 m 为 15/32、17/32、6/32，0.55 m 为 6/32、3/32、11/32，仍全部低于基线。这是重设计 jump 的物理依据。

## V2/V3 重设计与连续策略的控制边界

V2 独立控制器尝试在接触触发后对支撑脚同时施加向后和向下足端运动，并缩短蓄力阶段。原先约 50 ms 蓄力经常错过高速 canter 的后脚接触窗；缩短到一个控制步（20 ms）后，24 环境腾空数由 20 提高到 23，竖直起跳速度中位数约 1.1–1.3 m/s。不过同样的 24 个难度 0.75 环境中，V2 最好的小样本仍只有单次 gap 1/8、连续 gap 0/8、hurdle 3/8，低于原 canter 的 3/8、2/8、6/8。减小后蹬、减小蹬伸、调整触发距离和落地伸腿时序均未使 V2 稳定通过。目标足端位置的逆运动学误差多数时刻接近零，峰值常约 2–3 cm；裁剪不是唯一失败原因。gap 落点常位于缺口内或远侧边缘，hurdle 起跳时前向速度易丢失。

更关键的结构问题是，V2 的 PUSH/FLIGHT 分支直接覆盖足端 x/z 输出；一次跳跃触发后，连续策略在这两个阶段改变原始足端目标不会改变实际足端轨迹。单元反事实测试对此给出完全相同的输出；在同一 24 环境里把 jump 期间的连续动作改为策略输出、全 0、全 +1 或全 −1，gap 通过数仍分别只有 1/16、0/16、0/16、0/16。因此不能把 V2 的飞行失败简单归咎于连续策略尚未训练：它缺少实时调制通路。

V3 仍使用 1 个离散 + 18 个连续动作，仅在隔离任务中把这些已有连续通道的摘要锁存在接触触发瞬间，决定本次跳跃轨迹。六个参数为：动作 1–4 的均值控制后蹬幅度，5–8 控制 z 蹬伸深度（4–12 cm），9–12 控制落地伸腿延时（0.1–0.3 s），13–16 控制足端前伸（基准前脚 12 cm、后脚 10 cm，各可调 ±5 cm），17 控制腾空收腿高度（9–17 cm），18 控制蹬伸轨迹时间（40–80 ms）。原动作的 canter 调制用途保留；jump 触发由接触力决定，参数只在触发时锁存，避免飞行中动作抖动。V3 还保留进入 jump 前的底层周期步态相位，避免助跑 canter 立即切换到 bound 造成支撑突变。

为直接拆开“旧连续策略没有优化”和“动力学无法响应”，在 seed 123 的同一 96 环境、同一旧 V1 checkpoint 助跑、同一 0.55 m 真值选择下，仅替换 jump 被选择期间的连续动作：

| V3 jump 连续动作来源 | 单次 gap | 连续 gap | hurdle | 合计 | 实际启动 jump |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 V1 策略输出 | 2/32 | 0/32 | 9/32 | 11/96 | 96/96 |
| 全 0 | 13/32 | 1/32 | 16/32 | 30/96 | 96/96 |
| `[1, 1, -0.5, 1, 1, -1]` | 24/32 | 15/32 | 26/32 | 65/96 | 96/96 |

三组初始状态 SHA256 相同；触发后记录的竖直蹬伸深度中位数依次约为 7.5、8、12 cm，后蹬幅度约为 6.1、6、9 cm，证明现有连续动作确实进入 V3 物理控制。旧策略的连续头只在 V1 动力学与普通 canter 上训练，其 11/96 **不能解释为 V3 原生上限**。手工向量虽然明显改善，但仍低于同场景不跳的 canter 67/96，且全地形固定使用时不稳定；这也是为何必须同时学习离散选择。

下面 96 个环境使用同一旧 V1 checkpoint 负责助跑，在距第一障碍 0.55 m 处用真值位置选择 jump 并保持 15 个控制步；V3 的混合调制向量为 `[1, 1, -0.5, 1, 1, -1]`。这**不是训练过的 V3 策略**。两组初始状态 SHA256 均为 `59d4e1e2cb1e6ce98fb8c450d7dad288b87f4c12c092f2f1629c54d3424ce404`。

| 地形 | 难度 | 原 canter | V3 真值触发 + 固定调制 | 配对净变化 |
| --- | ---: | ---: | ---: | ---: |
| 单次 gap | 0.50 | 15/16 | 13/16 | −2 |
| 单次 gap | 0.75 | 7/16 | 11/16 | +4 |
| 连续 gap | 0.50 | 15/16 | 6/16 | −9 |
| 连续 gap | 0.75 | 9/16 | 9/16 | 0 |
| hurdle | 0.50 | 14/16 | 15/16 | +1 |
| hurdle | 0.75 | 7/16 | 11/16 | +4 |

V3 在较难单次 gap 和 hurdle 上已显示物理收益，而较易连续 gap 会因强制跳跃显著退化。按难度事后选取“难度 0.75 使用 V3、难度 0.50 保留 canter”的配对结果可组合成单次 gap 26/32、连续 gap 24/32、hurdle 25/32，合计 75/96，原 canter 为 67/96。这只是依赖地形真值的条件选择估计；真实策略能否从 60 个高度观测中学到选择和连续调制，需要新训练验证。当前 16 环境/格样本也不足以声称稳定提升。

同一参数在 seed 321 的独立 96 环境复核中，初始状态 SHA256 为 `72cd051ba57a7c42c618e8a8bc9441d76316e09ac449c466699958fd76122259`。每格仍是 16 环境，真值触发共 96 次、全部启动和腾空、94 次进入落地状态。较难单次 gap 从 6/16 升到 9/16，但较难连续 gap 从 7/16 降到 4/16，较难 hurdle 维持 10/16；易连续 gap 从 16/16 降到 13/16。总体基线/V3 分别为 68/96 与 65/96（seed 123 为 67/96 与 65/96）。若把 seed 123 的“只在难地形跳”规则原样用于 seed 321，配对组合仍为 68/96，未超过 canter 基线。**难单次 gap 的局部收益可以复现，整体通过率与连续 gap 稳定性尚未改善。**

作为控制能力而非部署成绩的进一步上界估计，若事后只在较难单次 gap 和全部 hurdle 选 V3、其它场景保留 canter，则两个种子的配对组合分别为 76/96（原 67/96）与 74/96（原 68/96）。这说明六维调制空间内存在有用的起跳轨迹，主要剩余困难是何时选择；同时该规则直接使用地形类别和难度真值，当前策略只看到本体状态与局部高度测量，不能假定它可辨认完整连续 gap 模式。该组合不是一次真实无特权 rollout，也不能代表最终策略性能。

故障归因按可控性区分：V1/V2 的后蹬与落地轨迹不足、以及 V2 在关键阶段屏蔽连续动作，属于原生动力学缺陷；V3 的同场景 11→65/96 连续动作消融说明连续参数未优化是另一条独立瓶颈；较易连续 gap 在强制跳跃时退化，说明选择策略必须学会保留 canter。V3 新策略的训练结果见下节，不能把真值触发数据当作已学会的策略能力。

## V3 新策略训练：物理机会尚未转化为学到的 jump

独立任务 `MP_full_jump_v3_a1` 从零参数启动，seed 1、4096 环境，沿用原训练奖励与熵系数 0.001。原计划 150 轮；第 20 轮保存后因 jump 采样占比从第 1 轮的约 19.8% 依次降至第 9 轮 3.8%、第 20 轮 0.59%，且两个已保存 checkpoint 的确定性评测均没有 jump，提前结束本组试验。这是明确的早停结果，不能推断增加训练预算也绝无可能学会。

| 新策略 checkpoint | 单次 gap | 连续 gap | hurdle | 96 环境 jump 事件 | 确定性步态 |
| --- | ---: | ---: | ---: | ---: | --- |
| V3 第 10 轮 | 0/32 | 0/32 | 12/32 | 0 | 全 bound |
| V3 第 20 轮 | 17/32 | 4/32 | 22/32 | 0 | 全 canter |
| 旧 V1 第 150 轮 | 22/32 | 24/32 | 21/32 | 0 | 全 canter |

这三组评测使用相同 96 初始状态 SHA256 `59d4e1e2cb1e6ce98fb8c450d7dad288b87f4c12c092f2f1629c54d3424ce404`。V3 第 20 轮并未超过旧 V1 或原 pronk checkpoint；全部通过数来自普通步态，**不能说 V3 policy 已具备跳跃能力**。与真值触发在较难障碍上的收益并列看，当前瓶颈是离散选择早期坍缩和连续跳跃参数缺少有效训练样本，同时 V3 在易连续 gap 上强制跳跃仍有原生退化风险。下一步若要训练选择策略，应针对较难障碍增加有意义的 jump 探索样本，并保留易连续 gap 的 canter 负例；需要用无真值的确定性评测确认收益。

## 原始数据与复现

- 原 pronk 首次带基线：`docs/terrain/mp_full_jump_baseline_first_band_96/report.json`
- 原 pronk 第二带基线：`docs/terrain/mp_full_jump_baseline_second_band_10s_96/report.json`
- 主实验第 50、100、150 轮：`docs/terrain/mp_full_jump_iter{50,100,150}_first_band_96/report.json`
- canter 步长消融：`docs/terrain/mp_full_pronk_canter_scale1_first_band_96/report.json`
- 真实状态概率与脉冲特征：`docs/terrain/mp_full_jump_features_iter10_48/reservoir_diagnostic.json`、`docs/terrain/mp_full_jump_logits_iter100_24/primitive_diagnostic.json`、`docs/terrain/mp_full_jump_features_iter100_24/reservoir_diagnostic.json`、`docs/terrain/mp_full_jump_features_iter150_48/reservoir_diagnostic.json`；原 pronk logits 见 `docs/terrain/mp_full_pronk_logits_final_24/primitive_diagnostic.json`
- 主实验配置、代码摘要与第 150 轮权重：`legged_gym/logs/A1/1026_17-54-09_A1MPfull-v3-18D-psF_3.333e+00-mpF_1.667e+01-MF_1e+03-noResume-jump5-noPronk/`
- 熵系数对照：`legged_gym/logs/A1/1026_18-38-20_A1MPfull-v3-18D-psF_3.333e+00-mpF_1.667e+01-MF_1e+03-noResume-jump5-noPronk-entropy002/`
- canter 步长对照：`legged_gym/logs/A1/1026_18-44-52_A1MPfull-v3-18D-psF_3.333e+00-mpF_1.667e+01-MF_1e+03-noResume-jump5-noPronk-canterScale1/`
- 步长对照第10轮概率：`docs/terrain/mp_full_jump_canter1_features_iter10_48/reservoir_diagnostic.json`
- 步长对照第50轮评测与概率：`docs/terrain/mp_full_jump_canter1_iter50_first_band_96/report.json`、`docs/terrain/mp_full_jump_canter1_features_iter50_48/reservoir_diagnostic.json`
- 熵系数对照第10轮概率：`docs/terrain/mp_full_jump_entropy002_features_iter10_48/reservoir_diagnostic.json`
- 熵系数对照第50轮评测与概率：`docs/terrain/mp_full_jump_entropy002_iter50_first_band_96/report.json`、`docs/terrain/mp_full_jump_entropy002_features_iter50_48/reservoir_diagnostic.json`
- 熵系数对照第100轮评测与概率：`docs/terrain/mp_full_jump_entropy002_iter100_first_band_96/report.json`、`docs/terrain/mp_full_jump_entropy002_iter100_second_band_96/report.json`、`docs/terrain/mp_full_jump_entropy002_features_iter100_48/reservoir_diagnostic.json`
- 原组第50轮概率：`docs/terrain/mp_full_jump_features_iter50_48/reservoir_diagnostic.json`
- jump 物理先导配对：`docs/terrain/mp_full_jump_oracle_baseline_smoke_24/report.json`、`docs/terrain/mp_full_jump_oracle035_smoke_24/report.json`
- jump 物理96环境基线与0.15m触发：`docs/terrain/mp_full_jump_oracle_baseline_035_96/report.json`、`docs/terrain/mp_full_jump_oracle_oracle_015_96/report.json`
- V1 全部距离扫描：`docs/terrain/mp_full_jump_oracle_oracle_{035,055,075}_96/report.json`；15 步选择：`docs/terrain/mp_full_jump_oracle_hold15_{035,055}_96/report.json`
- V2 轨迹与逆运动学诊断：`docs/terrain/mp_full_jump_v2_oracle055_ik_24/report.json`、`docs/terrain/mp_full_jump_v2_oracle055_ik_24/trace.npz`；快速蹬伸与落地扫描见 `docs/terrain/mp_full_jump_v2_fast08_24/`、`docs/terrain/mp_full_jump_v2_delay{015,020,025}_24/`
- V3 seed 123 的 96 环境受控对照：`docs/terrain/mp_full_jump_oracle_baseline_035_96/report.json`、`docs/terrain/mp_full_jump_v3_preserve_{policy,zero,mixc,high}_96/report.json`
- V3 seed 321 的 96 环境复核：`docs/terrain/mp_full_jump_seed321_baseline_96/report.json`、`docs/terrain/mp_full_jump_seed321_v3_mixc_96/report.json`
- V3 第 10、20 轮真实策略评测：`docs/terrain/mp_full_jump_v3_iter10_first_band_96/report.json`、`docs/terrain/mp_full_jump_v3_iter20_first_band_96/report.json`；训练配置、源代码摘要、权重与 TensorBoard 数据：`legged_gym/logs/A1/1026_19-59-29_jumpV3-modulated-preserveGait-seed1/`

V1 的训练和评测入口为 `legged_gym/legged_gym/scripts/train_mp_full_jump.py`、`legged_gym/legged_gym/scripts/evaluate_mp_full_jump.py`；V3 为 `legged_gym/legged_gym/scripts/train_mp_full_jump_v3.py`、`legged_gym/legged_gym/scripts/evaluate_mp_full_jump_v3.py`。例如重新启动独立 V3 训练：

```bash
python legged_gym/legged_gym/scripts/train_mp_full_jump_v3.py \
  --task MP_full_jump_v3_a1 --headless --num_envs 4096 \
  --max_iterations 150 --seed 1
```

该命令创建新的训练，不会从现有 V1/V3 checkpoint 自动恢复。V3 的确定性评测必须调用 V3 专用入口并指定 V3 checkpoint。复现 seed 123 的受控物理试验可运行：

```bash
python legged_gym/legged_gym/scripts/probe_mp_full_jump_v3_oracle.py \
  --jump-mode oracle --arm-distance 0.55 --hold-steps 15 \
  --jump-modulation 1 1 -0.5 1 1 -1 \
  --checkpoint legged_gym/logs/A1/1026_17-54-09_A1MPfull-v3-18D-psF_3.333e+00-mpF_1.667e+01-MF_1e+03-noResume-jump5-noPronk/model_14_150.pt \
  --output docs/terrain/mp_full_jump_v3_recheck_96 \
  --terrains gap_single gap_continuous hurdle \
  --difficulties 0.5 0.75 --speeds 1.2 --repeats 8 \
  --seconds 5 --seed 123 --hazard-target first_band
```

真值触发探针使用旧 V1 权重，仅测物理机会，不能代表训练结果。相关控制器、任务隔离与调制锁存单元测试通过，语法检查通过；测试不代表障碍通过率已经整体提升。
