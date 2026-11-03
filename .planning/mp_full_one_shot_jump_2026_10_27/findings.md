# 发现

- 现有四步态为 walk/trot/canter/bound；旧 jump V3 依赖索引 4 的旧 pronk 表，当前默认四步态表不能直接构造它。
- MP_full_a1 的控制步长为 20 ms，原 primitive 每 15 步更新一次即 300 ms；连续调制每 3 步更新一次。
- ContactJumpReflexV3 已实现 10 ms 预压、约 60 ms 蹬伸、腾空和落地恢复，足端目标在 1 ms 物理子步更新。
- 旧训练 jump 选择坍缩，单纯提高离散熵未能让部署策略选择 jump；新奖励须有时机和事件信号。
