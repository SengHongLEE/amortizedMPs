# 楼梯课程的立面形状突变

## 已复现的问题

原配置`horizontal_scale=0.1 m`、`vertical_scale=0.005 m`、
`slope_threshold=0.75`。Isaac Gym的高度图转网格函数仅对
相邻高度差**大于**`slope_threshold * horizontal_scale`的边进行竖直化。
因此边界为7.5 cm：较低楼梯的每个立面实际上是跨一个10 cm网格单元的小斜坡，
8 cm才变为竖直面。直接检查三角网格顶点复现了这一点：

| 请求高度 | 原网格沿行进方向的立面水平跨度 |
|---|---:|
| 2 cm | 10 cm |
| 6 cm | 10 cm |
| 7.5 cm | 10 cm |
| 8 cm | 0 cm |

这解释了为何只差5 mm却出现几何形状突变，并指出原课程不只是逐渐增加高度。
它不证明这是所有控制失败的唯一原因，也不证明修正后训练一定成功。
此前7/7.5 cm的高通过率是旧网格上的结果，不能作为真实竖直楼梯验收证据。
此前8 cm及其余地形的结果仍保留。

## 修正

`terrain_params.stairs.mesh_vertical_faces=True`及
`terrain_params.stairs_linear.mesh_vertical_faces=True`使三角网格中的楼梯
在所有高度都有真实立面，不依赖全局斜率阈值。
该参数只作用于trimesh；heightfield模式仍只生成高程数组，无法表示垂直面。

构建器从未叠加表面噪声的楼梯形状计算SDK的网格位移，只替换相应立面顶点的
XY坐标。踏面噪声、Z坐标、高程采样、出生点、三角形索引及其他地形均保留。
不能通过降低全局斜率阈值修复所有地形，否则会改变坡道和波浪的几何。

新`mp_proprio_terrain`默认配置显式开启该设置。构建器缺省值为False，
因此旧checkpoint保存的配置仍可复现；继续训练旧权重必须显式选择新几何：

```bash
python legged_gym/legged_gym/scripts/train_adaptive_equilibrium.py \
  --checkpoint SOURCE_CHECKPOINT --output NEW_RUN \
  --vertical-stairs --minimum-level 0 --initial-level 2 \
  --iterations 200 --num-envs 2048
```

以上为接口示例，不代表某次已完成的训练。实际实验参数见`progress.md`及run的
`config.json`和`provenance.json`。

`evaluate_proprio_terrain.py`和`diagnose_proprio_stairs.py`也支持
`--vertical-stairs`，可以在相同权重下检验真实楼梯。
无该选项时保留保存配置的行为；报告中有效配置及逐块参数明确记录该字段。

诊断入口另支持`--heights 0 .02 .04 .06 .075`，便于比较不同高度的轨迹。
默认仍为0/2/4/6/8 cm；高度必须递增且精确落在竖直网格上。
保存的`initial_state.npz`新增origins，可核对不同地图绝对高度下的相对初态。

## 几何验证

`test_stair_mesh.py`覆盖两种楼梯、x/y轴、两个方向及1/2/6/7.5/8 cm：
立面沿行进方向的水平跨度必须为零。
还检查旧配置网格复现、8 cm原网格逐位一致、其他地形顶点逐位一致，
以及踏面噪声不被额外竖直化。高程图高度正确本身不足以证明碰撞网格正确。

实际策略通过率需重新评估低台阶并重新训练；几何测试不等于爬楼成功。
