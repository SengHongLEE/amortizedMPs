# 发现

- 2026-10-30 午间当前 MP_full_a1 训练持续运行，RTX 4090 约 7.7 GiB 已占用。
- 保存配置为 `legged_gym/logs/A1/1030_10-08-49_A1MPfull-signed-dense-v1/config.json`。
- 既有 `evaluate_proprio_terrain.py` 提供逐回合评估和配对种子做法，但依赖已删除的 proprio 模块；需为 MP_full 单独接入。
- 历史 2026-10-26 的 bound 障碍探针失败，不足以判断最新 checkpoint。
- 固定 checkpoint `model_245_2450.pt` SHA256 为 `e5819b4ac8df978d482a574eba2b5f31b373a54b7fcb8c47a63b3a32c52eb95e`。
- 配对评估初步 walk：flat 32/32，stairs 16/32，hurdle 13/32，gap_single 1/32；需等其余 gait 对照。
- walk 平地 0.4 m/s 原始 CoT 平均约 7，训练的 `1-tanh(CoT²)` 已接近零，确认效率信号饱和风险。
- 第一轮四 gait x 128 episodes 的初态 SHA 与源码 SHA 完全一致。总通过率按 flat/stairs/hurdle/gap_single：walk 32/16/13/1，trot 32/20/23/17，canter 32/27/29/32，bound 32/23/31/26（每项分母32）。
- 平地0.4m/s：walk 速度绝对误差.13、selection return40.3；trot .21/30.3；canter .32/20.8；bound .23/28.3。四者均8/8成功。
- single gap：canter 32/32，bound26/32，trot17/32，walk1/32。hurdle：bound31/32、canter29/32，bound优势需更多样本核实。
- canter 实验在写完完整报告后退出时出现一次 Python/Isaac Gym 段错误；前三组报告保留且 bound 单独重跑成功。原训练进程日志显示外部 KeyboardInterrupt，已停止；评估未发停止信号。
