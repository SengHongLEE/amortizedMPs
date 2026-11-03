# A1 MP 双任务整理

## 目标
仅保留 A1 的 `MP_full_a1` 与 `MP_interoceptive_a1` 两个注册入口；前者默认匹配 1028 signed-dense run，后者按日志配置恢复六种步态。所有 `logs/A1` checkpoint 能按所属 task 加载并继续训练。

## 阶段
1. [complete] 冻结日志配置、checkpoint 格式和现有工作区状态，建立失败测试。
2. [complete] 合并 full 默认配置，精简 A1 注册入口及续训选择逻辑。
3. [complete] 清理两个活动环境和 runner 中的调试副作用，保留训练行为。
4. [complete] 验证所有 checkpoint 的结构加载、两个 task 的短训练与推理，更新使用文档。

## 约束
- 保留已有未提交修改，不删除未跟踪的实验代码和日志。
- Go2 注册入口不变。
- 日志 `config.json` 是恢复每个 run 的权威配置；checkpoint 格式保持兼容。
- 用户所说复现指加载并续训，不要求重新训练得到逐字节相同权重。
- 归档未跟踪的旧实验源码与测试至 `.planning/mp_two_tasks/archived_debug_code.tar.gz`，并归档旧 full runner；活动源码目录只保留两个 A1 task 的入口。
