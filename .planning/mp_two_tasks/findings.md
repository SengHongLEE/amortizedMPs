# 发现

- `logs/A1` 有六组 1018 interoceptive（每组 41 个 checkpoint）和一组 1028 full signed-dense（201 个 checkpoint）。
- full 的日志网络为 19 actions、262 actor obs、269 critic obs、signed-dense 2×16×5；当前 `MP_full_a1` 默认仍为 legacy 60 扫描，signed-dense 另注册为 `MP_full_signed_dense_a1`。
- interoceptive 共用 5 actions、42 actor obs、49 critic obs，六组 run 的步态不同，必须加载各自 `config.json`。
- `train.py` 在建环境前未加载 run config，当前 `--resume --load_run` 对非默认步态不可靠。
- checkpoint 文件名是 `model_<序号>_<训练迭代>.pt`；`get_load_path` 显式 checkpoint 分支目前拼成 `model_<n>.pt`，无法匹配。
- interoceptive checkpoint 保存的 `iter` 为 0；文件名的第二数字反映更新数，应在续训时修复迭代计数。
- 工作区有大量既有修改与未跟踪实验文件，需要保留。
- 447 个归档 checkpoint 均通过保存配置构造的 runner 严格 `state_dict` 加载。
- full 的 `model_200_2000.pt` 和 interoceptive trot 的 `model_40_200.pt` 均通过训练 CLI 的一次续训，分别输出 `model_201_2001.pt` 和 `model_41_201.pt` 到 `/tmp`。
- walk、pace、trot、canter、bound、gallop 六步态均在真实 Isaac Gym 中加载最新 checkpoint 并执行推理。
- 旧实验 task 的单元测试依赖已删除注册入口，不能纳入双任务活动测试集。其未跟踪实验源码与测试已归档。
