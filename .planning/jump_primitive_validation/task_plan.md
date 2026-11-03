# MP_full_a1 跳跃 primitive 验证

## 目标
在不改变默认 18 维双 policy 训练的前提下，验证四足同相候选是否改善 gap/hurdle，并提供可复现的 5 primitive 训练入口。

## 阶段
1. [x] 用当前 18 维 checkpoint 运行同种子 baseline / 强制 bound / 四足同相探针。
2. [x] 加入可选的第 5 个 pronk primitive；保持默认 4 primitive 和旧 checkpoint 行为。
3. [x] 运行单元测试、配置检查和最小训练 smoke；GPU 正运行用户训练，暂不并行进行完整重训。
4. [x] 汇总初步结果、限制和后续训练方案。
5. [x] 按用户建议加入可变 stance_extension 的可选消融，保持旧策略兼容并比较 0/2/4 cm 上限。

## 已知限制
- 当前 GPU 上已有用户启动的 train.py 和 play.py；避免并行启动耗时的新训练。
- 现有 checkpoint 仅 4 primitive；强制相位探针无法替代重新训练。
