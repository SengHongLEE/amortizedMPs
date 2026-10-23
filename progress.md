# 进度记录

## 2026-10-21：持续目标启动
- 分类：进展（确认实际进程终止、最新权重和配置；开始建立独立基线）。
- 已调用 get_goal：目标 active，无 token 预算。
- 检查进程与 GPU：无活跃训练，最新权重已经从 730 推进到 810。
- 读取文件规划与设计技能；用户已授权系列优化、训练与验证，直接执行。
- 写入完整目标、阶段和成功证据要求，保留其他复杂地形回归要求。
- 下一步：全地形独立评估工具和 810 基线；随后实现可学习时序与摆动高度参数。

## 当前优化轮
- 新增首次episode全地形评估工具，统计侧偏走廊内的正向跨越，测试确保
  横向出界/跌倒后复位不能增加通过率。
- 810 楼梯详细复测及录像完成，报告位于 docs/terrain/equilibrium_diagnosis_0810。
- v2 动力学控制器和独立配置任务已实现；增加轨迹相位对齐开关。
- 严格网络迁移和实际 CUDA PPO smoke 已通过。
- 生成 `legged_gym/logs/A1/1021_equilibrium_v2_aligned_warmstart/warmstart.pt`；
  尚未正式训练，不能称为楼梯修复。
- 正在运行 aligned warmstart 混合评估与 verified v1 基线；下一步根据
  零样本回归结果决定对齐选项，然后开始混合地形微调。

## 第一轮微调已启动
- verified v1 基线仍为 stairs72/96、wave89/96，其余六类96/96。
- aligned warmstart：stairs73/96、wave93/96、其余六类96/96；
  8cm上阶仅1.2m/s组2/8，通过下降8cm的.8m/s组丢失1次；
  并不足以证明可靠提升，因此继续训练。
- 102个 legged_gym 测试通过，结果 `/tmp/terrain_tests_v2_verified.log`。
- 训练 PID 812245，session 33100；2048环境、60次更新、学习率3e-4固定、
  现有八类地形混合，课程初始等级0–5。
  目录 `legged_gym/logs/A1/1021_equilibrium_v2_mixed_round1`；
  日志 `/tmp/train_v2_round1.log`（结束时复制入run目录）。
- 10次更新权重正在独立评估6/7/8cm、上下阶、.5/.8/1.2m/s，各8次。
  评估session76227；目录`docs/terrain/evaluation_v2_round1_10/stairs_seed123`。
  还没有最终训练后通过率，目标仍active，禁止提前标记完成。

## 中途评估与下一步（恢复入口）
- round1 第10次更新：6cm上阶23/24，7cm上阶20/24，8cm上阶0/24；
  全部144个楼梯episode中113通过。报告已完成。
- 第30次更新：flat/slope/rough_slope/rubble各96/96，rough93/96，
  ramp95/96，wave95/96，stairs71/96。8cm上阶三个速度均0/8。
  控制器实测平均频率仍约6.9–7.4Hz，duty约.52–.55，lift约.14–.15m，
  因此当前微调只学到了很小的时序变化。没有解决上阶。
- 发现并修正评估初态偏差：不同网络尺寸在构造时消耗不同数量随机数，
  原“同seed”并不保证逐环境相同初态。现改为加载网络后重置seed并env.reset，
  保存initial_state.npz和SHA256。v1/v2实际16环境校验均为
  `a3020e87d7570c56abc3d5b42073a33dcb1807eeb1bbd7f2a260091a8286c8c9`。
  先前评估是分布层面的初步对照，不作为最终严格配对验收。
- 正在重跑v1完整配对基线：session66047，
  输出`docs/terrain/evaluation_0810/paired_seed123`，日志`/tmp/eval0810_paired.log`。
- 训练session33100仍运行，最近已保存model_4_50.pt，预计60次更新结束。
  结束后复制/tmp/train_v2_round1.log到run/train.log。
- 下一步：60次更新权重完整配对评估，核对初态SHA与paired v1完全相同。
  再用新增`--sample-low`对照确定性均值与训练Gaussian采样行为差异；
  不能直接把探索噪声大认定为根因。
- 如果60更新仍失败，需要更充分训练/不同初始化，优先检验低频、较长支撑、
  较低摆动高度的动力学先验，而非继续小幅调默认参数。
  候选先验可用lift~.11m、频率倍率~.55、duty偏移~+.15；
  这必须作为独立实验充分再训练，历史直接降频失败不否定再训练可能性。
- 还需检查frequency/duty联合范围的最短摆动时间；当前独立极值能产生很短摆动。
  后续若加入约束，必须版本/配置明确，不能在旧checkpoint评估时悄悄改变物理映射。

## 后续目标轮：配对验证、控制器容量实验、强化楼梯课程
- 上一轮分类：进展（v2实现、真实训练及失败证据改变了下一步），目标继续active。
- round1已经终止，最终`model_6_60.pt`；train.log已复制入run目录。
- v1和round1_60的完整配对初态SHA一致。各类通过数（每类96）：
  flat96→96，slope96→96，rough_slope96→95，rough94→96，
  rubble96→96，stairs75→69，ramp96→92，wave93→95。
  round1_60的8cm上阶三个速度均0/8。该候选未达到要求，不替换默认v1。
- v1 Gaussian低层采样对照完成，初态与确定性基线一致：
  stairs71/96，8cm上阶0/24，ramp73/96。确定性基线8cm上阶3/24，
  所以没有证据支持“推理用均值是主要根因”。
- 新增诊断工具`probe_dynamical_priors.py`。256组固定参数×2初态，
  平地413/512、8cm上阶0/512。不能把单纯降低频率当作已验证解决方案。
- 姿态反馈原型在诊断env中将IMU gravity XY映射成各腿平衡高度修正，
  扫描gain、支撑z速度、平衡滤波频率。完整JSON核对后是4/192次上阶成功，
  分属4个候选，各1/2；其中两组平地2/2，两组平地1/2。
  此前只读日志尾部漏掉了前两组，已纠正。尚不适合进入默认控制器。
- 当前正在测试带tilt deadband的反馈，探地extension .06m（原.02）：
  session70174，PID815025，日志/tmp/prior_search_deadband.log，
  报告目录docs/terrain/dynamical_prior_search_deadband_extension。
  这些均为诊断env修改，没有悄悄更改生产控制映射。
- 强化楼梯课程round2首次启动时发现16列+stairs权重3会丢失ramp一个方向；
  已主动SIGINT停止PID815193，确认进程终止；其日志保存在原run/train.log。
  这是有明确布局缺陷的终止，不是观测超时后重复启动。
- 修正为20列后从round1_60重新启动：
  `legged_gym/logs/A1/1021_equilibrium_v2_stair_curriculum_round2_20cols`，
  session58875，日志/tmp/train_v2_round2_20cols.log。
  2048环境，200更新，固定3e-4，课程floor6/initial8（5.5–7cm起步），
  stairs权重3；实际打印确认stairs6列、其余每类2列，方向均保留。
- 下一步：检查deadband/extension扫描；等待round2中期权重后独立回放8cm，
  再做全地形配对回归。用户目标仍未达到，不能结束目标。

## 本轮恢复入口补充
- deadband+extension .06扫描已结束：216组、432次上阶0成功，平地311/432。
  不采用；它同时改变了反馈/探地组合，不能单独归因某个参数。
- 正在验证姿态反馈最有希望的基准配置：
  gait=trot、scale=.55、duty+.1、lift=.14、amplitude=.8、bx=-.02、bz=.02，
  平衡filter3Hz、支撑z速度.08；比较gain0/1/2/4。
  session75033，`/tmp/prior_posture_validation.log`，输出
  `docs/terrain/dynamical_prior_posture_validation`。
  每组32初态，独立seed321，逐候选复制相同root/dof初态实现配对，
  不以2个样本筛选结果直接认定反馈有效。
- 同时进行低频CPG+执行器刚度20/40/60/100、阻尼.5/1/2容量扫描：
  session21907，`/tmp/prior_search_impedance.log`，
  `docs/terrain/dynamical_prior_search_impedance`。只改诊断env每环境p/d gains。
- round2_20cols PID815448、session58875确认仍运行，已开始真实PPO更新；
  后续在50/100/200更新做独立评估，不重复启动活跃训练。
- 新增/修改的3个脚本已AST检查通过，训练/评估工具通过真实运行验证。
  本轮未改变生产动力学控制器，也未替换默认checkpoint。

## 协同动作 v3 与最新权重复查
- round2_100 配对结果：flat/slope/rough_slope/rough/rubble/wave 各96/96，
  stairs74/96、ramp95/96；8cm上阶0/8、1/8、1/8，下阶24/24。
- v3 warmstart 已评估，初态SHA与v2_100一致：stairs73/96、ramp94/96，
  其余六类96/96；8cm上阶1/24。控制器/网络代数迁移测试等价，但接触仿真
  不保证浮点变换后轨迹逐位一致；未见足以中止实验的明显回归，不宣称改善。
- v3实现：每轴common/front-back/left-right/diagonal四模式，控制范围与v2相同；
  显式变换actor/critic输入、低层输出、Gaussian边缘方差。
  动作变化惩罚按原Cartesian坐标计算，避免改变奖励量纲。
- v3真实CUDA烟测含高低层PPO更新、保存重载通过；全legged_gym测试107通过。
  首次直接unittest调用因torch先于Isaac导入失败；改用import isaacgym后的发现执行通过。
- v3混合训练已启动：session5935，输出
  legged_gym/logs/A1/1021_modal_v3_mixed_round1，日志/tmp/train_modal_v3_round1.log。
  从warmstart迁移权重继续100更新、2048环境、lr1.5e-4、floor6/initial8、
  stairs权重3、20列；未重复施加探索倍率。v2 round2_20cols同时接近200更新。
- 姿态验证完成：gain0/1/2/4在同初态32次中上阶0/3/4/5，平地全32/32。
  关节阻抗扫描上阶0/512；足端阻抗扫描0/216、平地37/216，不采用。
- 诊断脚本新增实际频率、支撑比例、late-contact状态和探地伸长记录；
  构造网络后重新设seed并保存初态，旧诊断与新诊断不可直接当作相同初态。
- 下一步：v2_200全地形配对与逐帧诊断；检查v3中期训练。目标仍未达到。

### v3 环境入口裁剪缺陷（已复现并修复）
- 发现此前控制器/网络等价测试遗漏环境层：step仍逐分量裁剪模式动作，
  而迁移后的输入网络要求“旧足端动作裁剪后的反变换”。
  含饱和动作的实际step入口测试失败（4/46元素不符，最大误差.575）。
- 增加controller.bounded_actions；v1/v2保持原行为，v3在足端坐标裁剪后
  回到模式坐标，控制输入和历史观测使用同一值。新回归测试及全套108通过。
- 已SIGINT停止v3首轮PID819308，确认结束，train.log归档。
  该短训练不作为有效候选；重新从未更新warmstart开始，不能从该短训练续跑。
- 修正后CUDA/PPO烟测session65783；warmstart全地形复测session16707，
  日志/tmp/eval_modal_warmstart_bounds.log，目录evaluation_modal_v3_warmstart_bounds。
- v2 round2已完成200更新，最终model_20_200.pt，日志归档。
  全地形评估session74322，/tmp/eval_v2_round2_200.log；
  逐帧/视频诊断session45946，/tmp/diagnosis_v2_round2_200.log，
  目录docs/terrain/diagnosis_v2_round2_200。

### 最终 v2_200 失败定位与后续实验
- 配对全地形：六类各96/96，wave94/96，stairs71/96；
  8cm上阶0/24、下阶23/24。该候选拒绝，不替换默认任务。
- 最新逐帧诊断完成，0.5m/s，平地/2/4/6cm全8/8，8cm0/8、跌倒4。
  前脚至少一次踩到第一阶7/8、后脚5/8，平均最大x1.474m、侧偏1.296m。
  第一阶附近实际频率9.10Hz、duty.522，名义支撑约57ms；
  支撑后段缺接触35.8%（平地6.6%），IK投影接近零。
  完整证据 docs/terrain/diagnosis_v2_round2_200/analysis.md。
- 新增纯离线analyze_stair_trace.py，汇总首episode、侧向门槛、姿态、
  接触/相位及跟踪指标；已在旧/新trace上运行，不覆写旧诊断报告。
- 修正环境入口后的v3 warm全地形：stairs74/96、ramp95/96、rubble95/96，
  其余五类96/96；8cm上阶3/24。与v2_100初态SHA一致。
  不是逐位一致物理轨迹；无明显大幅退化，作为有限训练实验起点。
- 已重启有效v3训练：session27990，日志/tmp/train_modal_v3_round1_bounds.log，
  run=legged_gym/logs/A1/1021_modal_v3_mixed_round1_bounds，
  100更新、2048环境、lr1.5e-4，其余课程/布局同前。
  从未更新的warmstart出发，不使用被停止的错误入口短训练。
- 正在独立测试有界参考关节速度项D*(dq_ref-dq)（诊断flag
  --joint-velocity-feedforward 1），只改诊断env，生产控制不变：
  session70479，日志/tmp/diagnosis_v2_round2_200_velocity_ff.log，
  输出docs/terrain/diagnosis_v2_round2_200_velocity_ff。
  首次运行因reset_idx覆盖签名遗漏completed_episode报错，已修正并重启；
  失败日志保留/tmp/diagnosis_v2_round2_200_velocity_ff_reset_error.log。
- 下一步：检查FF对照的初态SHA、过阶/平地/跟踪变化；如果有效，做半增益
  与全地形回归后再决定是否纳入正式控制。继续检查v3_50/100候选。
  当前目标仍active，8cm尚未解决，不能完成或阻塞。

### 速度参考对照结束
- session70479完成；与原PD初态SHA一致
  7472995bd28e59f99ad94243deaffa3ab711fc5565b9c5fae0cdf14c334fc3c2。
- 两者平地/2/4/6cm均8/8，8cm仍0/8，跌倒4→8。
  第一阶附近关节跟踪误差.153→.101rad，但支撑后段缺接触35.8→54.7%，
  力矩饱和1.95→8.05%。不采用；不支持只提高轨迹跟踪精度解决上阶。
- docs/terrain/diagnosis_v2_round2_200/analysis.md已更新对照。
- 唯一长任务：v3有效训练PID820822/session27990，已至少23/100更新。
  下一轮先查该PID/日志，避免重复启动；在50/100独立评估。
- 接下来的控制器研究重点：接触事件是否真正约束相位推进和支撑，
  保留正常地形的中性映射，先做诊断对照再决定新版本。当前相位即便缺接触
  也只减速，不能等待；该结构限制明确，但尚未证明改它一定解决问题。

## 本轮：接触等待与高层任务目标
- 上一轮分类为进展：修复v3入口、完成v2_200实测和速度参考反证。
  本轮开始确认PID820822仍活跃，36/100；未重启该训练。
- 新增诊断ContactPhaseGate：mid-stance缺连续25ms支撑时等待最多80ms，
  保留原参考时钟；状态只在诊断实例，不进入训练。关闭路径、按接触/超时释放、
  部分reset共3测试通过。真实诊断session13485，
  /tmp/diagnosis_v2_round2_200_contact_wait.log，
  docs/terrain/diagnosis_v2_round2_200_contact_wait。
- v3原奖励训练50更新权重已做全地形评估：session35286，
  /tmp/eval_modal_v3_round1_50.log，
  docs/terrain/evaluation_modal_v3_round1_50/paired_seed123。
- 检查到高层目标潜在冲突：0.5m/s命令下完全停滞只损失约.02497跟踪奖励
  （dt前），其他四项总权重4。新增显式--task-aligned-selection训练对照，
  使用与低层相同的任务/安全奖励；默认不变。
- 修复高低层使用同名奖励时episode统计双计数，用selection_前缀单独统计。
  新测试先复现双计数，修复后全套112通过。
- 真实CUDA1次训练完成：1021_modal_v3_task_aligned_smoke，
  高层14个、低层15个权重张量改变且全部有限。
- 已启动100更新对齐奖励训练，与原v3相同warmstart/seed/lr/布局/课程：
  session83726，/tmp/train_modal_v3_task_aligned_round1.log，
  legged_gym/logs/A1/1021_modal_v3_task_aligned_round1。
- 当前两个长训练：原奖励PID820822/session27990和对齐奖励session83726。
  接触等待与50更新回放即将结束，先检查结果再继续控制器方向。
  设计与命令见docs/terrain/contact-and-objective-experiments.md。

### 接触门控结果与慢速初始化
- 接触等待结束，严格统计平地5/8、2cm1/8、4/6/8cm0/8，8cm跌倒7/8。
  不采用；局部冻结相位破坏原推进，不能直接进入生产或训练。
- v3原奖励50更新：flat/slope/rough_slope/rough/rubble均96/96、
  stairs73/96、ramp95/96、wave94/96；8cm上阶2/24，下阶23/24。
  仍无可靠提升，默认不变。
- 新增显式--timing-prior FREQUENCY_SCALE DUTY_OFFSET CLEARANCE：
  只重置6个时序/摆动高度输出头，前16空间头保持；不是等价迁移。
  对应测试先缺失失败后通过，检查常量目标、旧空间头不变、范围拒绝。
- 慢速先验smoke正在运行：session91272，/tmp/train_slow_prior_smoke.log，
  run1021_modal_v3_slow_prior_smoke，先验.55/+ .1/.12，任务目标对齐，
  初始课程0–1，32env、1次更新。其warmstart.pt未训练，后续长训练可从它开始。
- 当前长训练：v3原奖励已到80/100，对齐奖励约14/100。
  待原奖励结束后，再启动慢速先验的较易课程训练，避免同时堆积长GPU任务。

### 本轮当前运行入口（以此段为准）
- 原奖励v3训练100更新已结束，最终model_10_100.pt，stdout已归档train.log。
  全地形评估正在运行session24433，/tmp/eval_modal_v3_round1_100.log，
  输出docs/terrain/evaluation_modal_v3_round1_100/paired_seed123。
- 对齐奖励训练PID823001/session83726仍活跃，约46/100；
  日志/tmp/train_modal_v3_task_aligned_round1.log，下一检查点50及最终100。
- 慢速先验1更新smoke完成；未更新的warmstart用于正式200更新训练，
  PID824643/session36962，/tmp/train_modal_v3_slow_prior_round1.log，
  run=legged_gym/logs/A1/1021_modal_v3_slow_prior_round1。
  2048env、lr1.5e-4、floor0/initial1、stairs权重3、20列、目标对齐，
  没有重复重置timing-prior。后续50/100/200检查。
- 慢速warmstart的flat/stairs基线正在评估session16321，
  /tmp/eval_slow_prior_warmstart.log，
  docs/terrain/evaluation_slow_prior_warmstart/paired_seed123。
  第一次评估因保存的高层共享奖励字段被配置加载器忽略而报错；
  已在primitive_selection_scales显式声明可选字段，新增保存重载回归测试通过，
  重新跑同一权重。失败日志/tmp/eval_slow_prior_warmstart_reload_error.log。
  原奖励默认激活列表未改变；正在跑的对齐训练是直接设置字段，不受该重载问题影响。
- 新训练会正确保存更新计数：修复runner计数只在log分支局部递增、checkpoint
  iter长期为0的问题；无log时也能按请求次数结束。先复现两类失败再修复，
  rsl_rl相关4测试通过。活跃旧进程仍用旧实现，不重新启动；历史权重不改写。
- 最后全legged_gym测试114通过（/tmp/terrain_tests_contact_alignment_reload.log）。
  高低层真实PPO smoke均通过；新慢速主训练已实际更新，后续checkpoint会带真实iter。
- GPU4090 24GB，当前两个训练+两个短评估约19GB/98%利用率，均正常；
  不再叠加第三个长训练。下一轮先收集两个评估，再评估对齐奖励50/100。
  目标未完成，无阻塞。

## 后续轮：收集结果与水平碰撞识别
- 上一轮分类为进展（训练接口、重载/计数修复及实际训练推进）。
  开始时确认PID823001/824643仍活跃，约48/100、7/200，未重启。
- v3原奖励100更新完整配对结束：flat/slope/rough_slope/rough/rubble/ramp各96/96，
  wave95/96、stairs73/96；8cm上阶1/24，下阶24/24。该路线未形成可靠提升。
- 慢速未训练warmstart的flat/stairs回放结束：flat95/96、stairs46/96；
  8cm上阶1/24、下阶8/24。只说明初始化基本保留平地，不能作为合格策略。
- 读取v2_200原始trace发现：8cm第一阶附近水平力>10N且Fz<=1N的
  事件平均占足端子步1.88%，其中约86.1%的相关策略采样q_obstacle_peak<.05。
  0–6cm这组统计近零。是候选立面冲击信号，不是经过几何认证的碰撞分类。
- 新增HorizontalObstacleProbe，仅诊断实例使用：
  水平候选冲击在摆动期(.15π,π)补入q_obstacle，竖直支撑判定不改变。
  未输入水平信号时与原轨迹/观测逐位一致；late-swing触发、非假支撑、
  stance不触发及局部reset两项单元测试通过。
- 当前水平碰撞真实诊断session82812：
  /tmp/diagnosis_v2_round2_200_horizontal.log，
  docs/terrain/diagnosis_v2_round2_200_horizontal，
  同v2_200、0.5m/s、20s、8初态/高度，只有--horizontal-obstacle开启。
- 对齐奖励50更新全地形评估session42323：
  /tmp/eval_modal_v3_task_aligned_50.log，
  docs/terrain/evaluation_modal_v3_task_aligned_50/paired_seed123。
- 两个长训练仍活跃，最近对齐约69/100、慢速约28/200。
  下一步先收集上述两个评估，再评估对齐100/慢速50。

### 本轮最新结果与运行句柄
- 对齐奖励50更新完成：七类非楼梯均96/96，stairs83/96；
  8cm上阶4/8、2/8、6/8，共12/24，下阶23/24。
  通过episode在.5/.8/1.2命令下速度MAE约.282/.404/.683m/s；
  有明显降速，不能只用通过数宣称速度能力合格。
- 扩大独立初态：session95123完成，
  evaluation_modal_v3_task_aligned_50/stairs_seed321（seed321、terrain_seed3、
  每速度32初态，8cm双向），上阶4/32、13/32、12/32，共29/96，
  下阶30/32、32/32、28/32，共90/96。原8样本较乐观，远未达90%。
- 已启动v1_810同协议基线session96073，/tmp/eval0810_stairs321.log，
  docs/terrain/evaluation_0810/stairs_seed321；待核对初态SHA与分项结果。
- 对齐奖励100训练PID823001已完成并归档train.log。
  最终model_10_100.pt全地形评估session20146，
  /tmp/eval_modal_v3_task_aligned_100.log，
  docs/terrain/evaluation_modal_v3_task_aligned_100/paired_seed123。
- 慢速50全地形session49353已完成：flat/slope/rough_slope/rubble96，
  rough95、ramp87、wave95、stairs57（均/96）；暂不合格，不能替换默认。
  原慢速200训练PID824643/session36962仍活跃，最近约77/200。
  保存的model_4_50.pt内部iter确认为50，计数修复实际生效。
- 水平障碍诊断session82812完成，平地/2/4/6cm均8/8，
  8cm0/8、跌倒8/8、最大前进均值1.245m；前脚8/8踩过第一阶、后脚5/8。
  没有解决支撑问题，不采用；只增加检测不是充分修复。
- 新增训练CLI --entropy-coef、--modulation-std-scale，可用于后续精调。
  前者覆盖PPO熵系数，后者只缩放探索std，不改确定性均值；都验证有限值范围。
  尚未启动使用这两个选项的长训练；先比较对齐50/100，选较好者再续训。
  慢速训练继续，不重复启动。目标未完成。

## 本轮：精调继续、独立对照收尾及实际高度评估
- 确认原始v1_810在seed321上阶2/96、下阶94/96，对齐50更新29/96、90/96，
  初态SHA完全相同。对齐100更新原小样本上阶9/24、下阶23/24，
  其余七类96/96。保留50更新，不默认使用最新编号。
- 已启动精调PID827681/session84838，运行目录
  `1021_modal_v3_task_aligned_refine_round2`，日志`/tmp/train_modal_v3_refine_round2.log`。
  从对齐50更新继续200更新，2048env，lr1e-4，entropy .001，std乘.5，
  floor8/initial9，stairsweight3/20列。warmstart逐张量验证：除连续std恰好减半外
  所有高低网络参数相同；10更新高层14/低层15张量改变，全有限，checkpoint iter10。
- 慢速训练PID824643仍活跃，最近111/200；100更新全地形评估
  session76869，`/tmp/eval_modal_v3_slow_prior_100.log`，
  输出`docs/terrain/evaluation_modal_v3_slow_prior_100/paired_seed123`。
- 增加`evaluate_proprio_terrain.py --terrains stairs --stair-heights .06 .07 .08`，
  与difficulty参数互斥，严格检查网格可表示性。默认中难度实际4.5cm，不能代替6cm。
  三项评估测试通过，包括实际生成双向楼梯的高程图6/7/8cm检验。
  首次测试夹具误用selected_kwargs官方兼容接口，改为terrain_params后通过；
  未修改生产地形生成器。
- 目标未完成；下一步收集慢速100、精调50/100，候选需按明确高度和多种子验收。

### 当前结果和待完成进程
- 慢速100评估session76869完成，初态SHA与全地形基线一致：
  flat/slope/rough_slope/rubble/ramp各96/96，rough95、wave95、stairs72；
  8cm上阶三个速度均0/8，下阶均8/8。上阶平均频率6.08Hz、duty .599，
  未改善上阶，原计划200更新仍继续。
- 对齐50独立上阶失败分类：成功29、跌倒59、未跌倒但侧偏越界7、其余未通过1；
  原v1分别2/74/18/2。跌倒仍占主要部分，不能把失败主要解释为停止推进。
- 明确高度评估session4072仍在运行：
  `docs/terrain/evaluation_modal_v3_task_aligned_50/heights_seed321`，
  日志`/tmp/eval_modal_v3_task_aligned_50_heights321.log`；
  stairs6/7/8cm、双向、三速度、32初态，共576env。改变了布局和环境数，
  不能把其中8cm结果视为之前192env报告的相同初态重跑。
- 对齐50逐帧诊断session88003运行中，日志`/tmp/diagnosis_modal_v3_task_aligned_50.log`，
  输出`docs/terrain/diagnosis_modal_v3_task_aligned_50`，默认0.5m/s、5高度各8初态、
  20s；所有诊断干预关闭。完成后运行analyze_stair_trace.py，核对与v2诊断初态SHA。
- 最近精调PID827681约37/200，慢速PID824643约132/200，均正常。
  无需重启；先收集明确高度及轨迹，再评估精调50。

### 明确高度评估已完成
- session4072完成：对齐50更新6/7cm上、下阶各96/96；8cm上28/96，
  下90/96。8cm上三个速度6/32、13/32、9/32，56次跌倒。
  初态SHA `a345d3af9522e17276b1a75a5d05eda725e14343da5ffde7f7967ea70e8bfc30`。
  出现明确7→8cm能力下降；单种子6/7cm通过仍不能作为全部验收完成。
- 已启动原v1相同布局对照session70736，`/tmp/eval0810_heights321.log`，
  输出`docs/terrain/evaluation_0810/heights_seed321`，需完成后核对SHA。
- 对齐50诊断session88003和两个长训练继续。目标尚未完成。

## 本轮：配对轨迹与搜索伸长对照
- 上一轮分类为进展（确切高度接口、几何验证及576环境评估完成）。
  开始时确认两个训练PID824643/827681和两个诊断/评估仍活跃，未重启。
- 原v1明确高度评估session70736完成，SHA与对齐50相同：
  6cm上下96/96；7cm上90/96、下96/96；8cm上5/96、下94/96。
  对齐50的8cm上28/96是同初态改善，尚未可靠。
- 对齐50逐帧诊断session88003完成，summary生成session47665完成：
  平地及2/4/6cm各8/8，8cm3/8、3跌倒、2侧向越界；
  SHA与v2_200诊断相同。8cm附近f9.10→7.10Hz，duty .522→.601，
  late-stance缺接触35.8→30.0%，跟踪误差.153→.129rad，力矩饱和1.95→1.02%。
  详情和对比图在`diagnosis_modal_v3_task_aligned_50/analysis.md`。
- 环境32/33前脚曾踩第一阶但后脚尚未接续就侧翻。环境32左后脚2.6–3s
  最低脚中心离起始地面6.4cm（接触中心约2cm），offsetZ约-.9cm、
  搜索伸长约1.3cm（上限2cm）。描述性证据，不推断接触缺失先于侧倾。
- 新增诊断CLI `--stance-extension`，仅实例覆盖0..8cm，不改生产默认。
  报告保存请求和实际值。nan/0.081均在仿真前拒绝；第一次CLI验证漏PATH导致
  ninja依赖失败，补上cpgil PATH后通过，无代码改动来掩盖环境问题。
  同权重同初态4cm搜索诊断session66515：
  `/tmp/diagnosis_modal_v3_task_aligned_50_search4cm.log`，
  `docs/terrain/diagnosis_modal_v3_task_aligned_50_search4cm`，仍运行。
- 精调50更新全地形评估session3839正在运行：
  `/tmp/eval_modal_v3_refine_50.log`，`docs/terrain/evaluation_modal_v3_refine_50/paired_seed123`。
  最新精调约57/200、慢速152/200。无第三个长训练，目标未完成。

### 精调50与搜索对照结果
- 精调50评估session3839完成：flat/slope/rough_slope/rough/rubble各96/96，
  ramp90/96、wave93/96、stairs79/96；8cm上1/8、3/8、3/8，共7/24，
  下24/24。初态SHA与原全地形评估一致。尚不如对齐首轮50，
  且ramp下降超过5个百分点，不替换默认/最佳。训练按原计划继续200。
- 搜索4cm诊断session66515完成，summary session78639已生成：
  初态SHA与对齐50基线相同。平地/2/4cm各8/8，6cm7/8，8cm2/8、6跌倒；
  原8cm3/8、3跌倒。8cm late-stance缺接触32.1%（原30.0%），
  平地26.6%（原8.2%）；IK投影近零。拒绝该直接覆盖，生产仍2cm。
- 新增7.5cm评估session7593，仍运行：
  `/tmp/eval_modal_v3_task_aligned_50_height75.log`，
  `docs/terrain/evaluation_modal_v3_task_aligned_50/height75_seed321`，
  对齐首轮50、两方向、三速度、32初态，共192env。
  当前10行线性课程最后两级实际为7.0→8.0cm；先检查7.5cm能力，
  再决定是否细分课程，不把此假设当作已证实根因。
- 最新精调约75/200、慢速170/200，PID仍827681/824643；无阻塞，目标未完成。

### 7.5cm结果与下一步
- session7593完成：7.5cm上95/96（各速度32/32、32/32、31/32）、下93/96，
  上阶0跌倒、下阶3跌倒。初态SHA
  `1c6336dc1f16d914f43a9b02e8a15e129e299e7d0b26d7b1439134adbcd65eac`，
  布局不同于576环境明确高度测试。几何测试补充7.5cm双向并再次3项通过。
- 对齐50的摆动幅值在平地到8cm变化很小，8cm第一阶附近四腿约
  .132/.140/.119/.128m；水平候选冲击占子步约1.1%，较低高度近零，
  原障碍反射峰值均值仍<.01。仍是力信号候选，不是几何认证碰撞。
- 已启动对齐50（较低频率）的水平碰撞单参数对照session10303：
  `/tmp/diagnosis_modal_v3_task_aligned_50_horizontal.log`，
  `docs/terrain/diagnosis_modal_v3_task_aligned_50_horizontal`。
  之前该对照只在v2_200（约9Hz）上失败；这里检验约7Hz的新策略，
  所有其他诊断覆盖关闭。生产反射未改变。
- 后续可扩展诊断脚本的高度列表（当前硬编码0/2/4/6/8cm），对7.5/8cm
  的接触过程做可比较轨迹。该接口尚未实现，不应声称已经比较。
  当前未新增训练路线或新控制动作，仍等精调100/200、慢速200。
