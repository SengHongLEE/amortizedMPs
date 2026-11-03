# 进度

- 已运行三组 Isaac Gym 评测并保存到 `docs/terrain/jump_validation_18d_*_48/`。
- 首次运行因 `ninja` 未进 PATH 失败；补入 cpgil/bin 后三组成功。
- 已给评测脚本加入 `--force-pronk-phase`，用于在旧 4 primitive checkpoint 上单独测试四足同相。
- 新增可选 `--include-pronk` 训练入口；一轮 16 环境 / 8 rollout 步训练及新旧 checkpoint 评测加载 smoke 成功。
- 修正 `intrinsic_dynamics` 对输入维度的校验（实际需要 9 列），更新步态单元测试中已经过时的 6 步态及 5 列 action 假设。
- 初步结果与限制写入 `docs/terrain/mp_full_jump_primitive_validation_2026-10-26.md`。
- 新增可选逐腿 stance_extension：只对指定步态生效，复用现有高度动作；单元测试先红后绿。
- 2 cm 配对物理评测已完成，4 cm 评测进行中；默认旧模型仍无伸腿。
- 4 cm 配对物理评测已完成；旧 checkpoint 初始状态 SHA 与先前实验相同。
- 首次可变伸腿训练 smoke 因现有默认配置已变为 5 primitive 而重复追加步长系数，导致校验失败；入口改为幂等后，1 轮训练更新和 5 primitive 评测 smoke 均成功。
