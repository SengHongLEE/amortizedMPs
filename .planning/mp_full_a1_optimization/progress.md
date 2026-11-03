# 进度

- 2026-10-26 三项修复完成：移除 MP_full 4 维无效 lift，新策略连续输出 18 维（总动作 19）；新增 `gap_continuous` 与 `gap_single`，训练和评估共用障碍区间；相邻 ramp 行做高度对齐，并把高难连续 gap 与中难单次 gap 分配到不同列。先写的三个回归测试在旧代码分别失败，修复后 77 项相关测试通过。真实 Isaac Gym 16 环境、1 次 PPO 更新成功，actor 输出 18，obs 162，critic 169，日志区分两类 gap。旧 22 维 checkpoint 不能加载；历史实验报告已注明不兼容。全库旧测试含既存的另一个 `MP_pro_terrain_a1` 契约差异，不把它们计为本任务验证通过。

- 2026-10-26 新任务：调查 lift 删维、gap 等距及单次/连续区分、下坡 ramp 卡关。已定位 gap 放置相位在每格重置；下坡同类格之间存在向上高度跳变。受控 ramp 仿真 session 56475 运行中。

- 2026-10-26：确认上一轮取得进展：修正yaw奖励、配对评估多个checkpoint，70项相关单测与diff检查通过；未有同时改善跨格gap/hurdle的checkpoint。
- 2026-10-26：启动200次迭代长训练，session 53672，PID 2028218，日志 `/tmp/mp_full_v2_modcycle3_yaw1_long200_train.out`，从modC3 model_6_60.pt warmstart，保持双policy与0.07m后腿前摆。需持续核实进程与日志，不因观察超时重启。
- 下一步：加逐帧诊断，仅在评估中采样，不改变策略动作或训练；分析相位、足端、接触、可见高度图与出界时间顺序。
- 2026-10-26：评估器新增可选 `--trace-env-ids`，在自动重置前记录足端、触地力、CPG相位/轨迹、动作、指令、高度扫描及失败原因；不改变训练或控制。AST与diff检查通过。16环境、4秒的实际仿真smoke正在运行，session14770；通过后再采集24秒跨格轨迹。
- 2026-10-26：16环境smoke已完成，trace.npz有200步、root[200,2,13]、feet[200,2,4,3]、phase[200,2,4]，有效行均有限且报告记录正确env IDs。跨格128环境评估/16环境轨迹采集运行中，session13789，输出 `docs/terrain/mp_full_v2_modcycle3_next_tile_trace_128`。长训练至早期阶段，TensorBoard前10轮applied near-limit约1.6%、raw outside=0。
# 2026-10-26 实验汇总

- 对 200 次训练最终 checkpoint 完成 seed 123 短程与 seed 123/456 跨格 128 环境评测；对 yaw2 10 次 checkpoint 完成 seed 456 跨格复测。
- 已生成 `MP_full_a1_experiment_report_2026-10-26.md`，含原始数据链接、推荐 checkpoint、复现命令和当前失败模式。整体优化目标尚未完成，高难 gap/hurdle 跨格表现仍低。

