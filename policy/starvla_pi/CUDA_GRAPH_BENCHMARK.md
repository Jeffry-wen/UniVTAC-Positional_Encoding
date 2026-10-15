# StarVLA-PI CUDA Graph 验证

以下记录对应旧版 2048 宽度动作头及当时的精度设置，不代表当前默认 1024 宽度、FP32 动作头的速度。

2026-10-15，NVIDIA A100-SXM4-80GB，PyTorch 2.7.0+cu126。
使用预训练 Qwen3.5-2B、默认 24 层动作头、chunk 32、10 步去噪、batch 1、
head/wrist 两个视角。动作头与 VAE 为随机初始化；本测试验证计算一致性和耗时，
不评估任务成功率。该默认结构也已通过真实数据的单步训练和验证。

下面是串行复测结果。计时使用 CUDA 同步后的 wall time，交替运行 eager/Graph，
每种配置、每个计时范围采样 8 次，报告中位数；排除首次捕获。
部署入口包含观测预处理、模型计算、动作反归一化和关节动作转换，不含模拟器步进。

| 触觉模式 | eager (ms) | CUDA Graph (ms) | 加速 |
| --- | ---: | ---: | ---: |
| VAE S full / encoder | 251.58 | 101.68 | 2.47× |
| none | 247.49 | 86.12 | 2.87× |

首次捕获加首次回放约 1.59–1.63 秒。完整 VLM Graph 和完整动作 chunk Graph
均实际捕获并回放，未发生 eager 回落。三个真实 HDF5 帧、三个噪声种子、两种模式
共 18 组比较的最大输出差异为 **0**。另已验证返回输出不会被后续回放改写、
相同随机种子的 eager/Graph 随机数进度一致、正常推理每次使用新噪声。
完整报告：[cuda_graph_benchmark_a100.json](cuda_graph_benchmark_a100.json)。

额外的真实 VLM 验证使用两层动作头，以缩短触觉路径测试时间：
`none/as_image`、VAE `cls/full × encoder/decoder`，以及文本层捕获回落路径，
相同输入、相同噪声的最大输出差异均为 0。
离线部署验证仅依赖 run 内 Qwen config/processor 和 checkpoint，
将 `models_root` 指向不存在的目录仍可完成严格加载和推理。

## 复现

```bash
OMP_NUM_THREADS=4 python -m unittest \
  scripts.test_starvla_pi scripts.test_starvla_pi_cuda_graph \
  scripts.test_starvla_cuda_graph scripts.test_joint_actions scripts.test_policy_inputs

python scripts/benchmark_starvla_pi_cuda_graph.py \
  --checkpoint /path/to/starvla_pi/run \
  --episode resources/data/isaac51/lift_can/hdf5/0.hdf5 \
  --frames 0,10,20 --task lift_can --device cuda:0 \
  --iterations 20 --output /tmp/starvla_pi_graph.json
```

第二条命令测试指定训练 checkpoint；模型结构和触觉路径由 run 配置决定。
`--head-only` 仅捕获动作头，`--text-only` 使用文本层捕获替代完整 VLM 捕获。
