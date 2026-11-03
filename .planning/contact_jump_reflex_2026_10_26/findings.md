# 发现

- 当前运行训练 PID 2107479，占用 GPU 约 6.3 GiB；不启动完整新训练。
- 现有五原语 v3 checkpoint 有 18 个连续 action；评测脚本支持同种子确定性场景、记录 trace 和 success。
- 原语选择周期 0.30 s、调制周期 0.06 s；现有 pronk 周期 0.167 s。
- 环境每 1 ms 调用 ID.intrinsic_dynamics 并传足端接触力范数；可用独立包装器实现亚控制周期状态机。
- 现有 eval 脚本的 EvaluationEnv 可通过新脚本子类和局部替换复用，无需编辑原文件。
- model_31_320.pt 的 24 环境配对评测初始状态 SHA256 完全相同，前缀 907642989b1a7664。
- 基线：gap_single 4/8，gap_continuous 4/8，hurdle 3/8。
- 0.35 m oracle arm reflex：gap_single 4/8，gap_continuous 2/8，hurdle 3/8；24/24 arm、20/24 进入预压、17/24 检测腾空。未见总体收益，连续 gap 变差。
- 连续 gap 0.35 m arm 样本的第一次无接触事件接近前沿，提示起跳时机可能偏晚；下一实验把 arm 提前至 0.55 m。
- 0.55 m arm：gap_single 4/8、gap_continuous 3/8、hurdle 1/8；24/24 arm、21/24 进入预压、20/24 检测到四足无接触事件。同一初始状态哈希，提前挂起也未提高总体通过率，hurdle 反而明显变差。
- 0.35 m arm 的连续 gap 首次四足无接触位置距障碍前沿中位数约 -0.08 m；0.55 m arm 提早到约 -0.21 m。hurdle 分别约 -0.21 m 与 -0.25 m。故时机变化已实际生效。
- Trace 中 hurdle 环境 16 的 0.55 m 组在障碍附近世界坐标竖直速度峰值约 1.13 m/s（基线约 0.75 m/s），但该样本未通过；四足无接触本身不能证明越障有效。
- 0.20 m arm：gap_single 5/8、gap_continuous 3/8、hurdle 0/8；24/24 arm、17/24 进入预压、16/24 四足无接触。hurdle 的基线成功样本全部损失。四组初始状态 SHA 全同。
- 三档总通过数为 8/24、9/24、8/24，基线 11/24；当前旧策略与该 reflex 参数组合没有整体收益。
