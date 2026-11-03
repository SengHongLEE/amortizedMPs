# 发现

- 2026-10-26，当前 18 维 model_5_60.pt 的 48 环境实验使用同一个 checkpoint SHA 与初始状态 SHA。
- baseline: gap_continuous 0/16，gap_single 3/16，hurdle 10/16；障碍场景全部选择 trot。
- 强制 bound: 0/16，0/16，9/16；强制同相的未训练探针: 0/16，0/16，1/16。
- 探针相位在 0.5 s 后同步；连续 gap 在三组中均为 16/16 底盘碰撞。
- `stance_extension` 的更新在 `Interoceptive_Reflex_Dynamics._update_reflex` 中被注释，现有垂直轨迹主要抬脚。
- 第五 primitive 候选采用与 bound 相同的 10 Hz / 0.40 duty，只改变相位，以便后续训练消融。
- smoke 模型的配置确认为 5 primitive、18 连续 action、19 总 action；旧 4 primitive checkpoint 在改库后仍可评测加载。
- 旧轨迹中 `stance_extension` 从未更新；现有四个高度动作可给可选伸腿提供逐腿幅度，不需要增加 action 维数。
- 2 cm 上限的强制 pronk 相位探针：连续 gap 0/16，单次 gap 0/16，hurdle 3/16；六条抽样轨迹实际最大伸腿约 1.4–1.5 cm。
- 4 cm 上限的同种子探针：连续 gap 0/16，单次 gap 0/16，hurdle 6/16；抽样最大伸腿约 2.9 cm。hurdle 后腿 calf 接触步比例为 0.78%，高于无伸腿探针的 0.19%。
- 工作区默认配置已经从 4 primitive 变为 5，且第五步长系数 1.5；训练入口必须避免重复追加第六项。
