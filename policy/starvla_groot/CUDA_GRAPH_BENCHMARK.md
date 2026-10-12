# StarVLA CUDA Graph 实测（2026-10-12）

仅测试 VAE full、排除观测预处理和动作反归一化、四个模型部分各 100 次平均值的分析，见 [VAE_FULL_COMPONENT_BENCHMARK.md](VAE_FULL_COMPONENT_BENCHMARK.md)。本页原表格为包含预处理的部署入口中位数，口径不同。

已直接阅读 `/mnt/data_ssd/yichi.zhang/lerobot-v0.5.1-test-cuda_graph` 中的 `fast_vlm.py`、`fast_dit.py`、`fast_infer.py`，按其 FastVLM / FullChunkGraph 思路扩展了首版实现，并完成三种现有 checkpoint 的复测。该参考目录不含 `.git`，来源按文件路径和 SHA256 记录。

## 实现范围

- GR00T：参考 `FullChunkGraph`，把特征 repeat、完整 4 步 Euler 去噪、ensemble mean 和输出转 float32 放进同一张图；每次调用在 Graph 外生成新噪声，再复制到固定输入缓冲区，便于精确对照 eager 的输出和 RNG 推进。
- Qwen3.5：参考 `FastVLM`，按完整指令 tuple 和图像/触觉布局缓存文本 embedding 模板、图像 token 索引、视觉位置 embedding、M-RoPE 及 attention mask；图内执行 patchify、vision 和 text decoder。适用于 `qwen3_5_text`、default RoPE、SDPA vision；其他情形先回落文本 decoder Graph，或仅使用动作头 Graph。
- 无 encoder 触觉前缀的路径直接取 decoder 特征，省去动作预测不需要的语言 logits 和 KV cache；该项也包含在无触觉模型的整体收益中。
- 保留原 eager 路径及计算精度，不更改去噪步数、ensemble 数量或 checkpoint。返回独立输出张量，避免下次回放覆盖已排队的动作。
- 每个模块默认最多缓存 2 个 Graph；完整 prefix 缓存满后先尝试文本 decoder Graph，再回落 eager。完整 prefix 按每个样本的真实指令构建元数据，同长度的不同指令也会区分。训练、加载权重或移动模型时清理缓存。

适配时保留 UniVTAC 的真实 VLM hidden size、图像分辨率、state/action 维度、encoder/decoder 触觉融合和 padding mask；没有照搬参考入口里的固定 1024 hidden size / 256 分辨率 / 全 1 mask。视觉常量按实例保存，没有全局修改 transformers 类。保持原动作头 float32 autocast 和原 SDPA 选择，避免强制改成参考实现的 bf16 / `[FLASH, MATH]` 引入额外数值变化；本次捕获前后均实测逐元素一致。输出复制、缓存上限及训练/权重变更时的清理也保留。

本机 transformers 的 Qwen linear attention 使用 PyTorch 回退实现，尚未安装 FLA/causal-conv1d fast path。其循环有大量 kernel 提交，因此文本 decoder Graph 的收益明显；其他硬件或 fast-path 环境需重新测量。

## 测量口径

GPU 为 RTX 5090，PyTorch `2.7.0+cu128`，transformers `5.5.0`。batch=1，chunk=32，ensemble=8，去噪=4。输入来自 `.cache/control-audit/modelscope/isaac51/lift_can/hdf5/42.hdf5` 的第 0、10、20 帧，任务为 `lift_can`。

每个模式先预热 5 次，再交替执行 eager/graph、graph/eager，统计同步 GPU 后的墙钟耗时中位数。DINOv2 每模式 20 次，其余每模式 40 次。模型加载、HDF5 解码和 Graph 首次捕获不计入稳定推理时间。部署入口包含观测转换、归一化、模型推理、有效性检查和动作反归一化，不包含模拟器运行。

| Checkpoint | 动作头 eager → Graph（单独微测） | 整模型 eager → Graph | 部署入口 eager → Graph | 部署加速 |
| --- | --- | --- | --- | --- |
| DINOv2 S / cls / encoder，30000 | 42.14 → 39.17 ms | 195.11 → 78.89 ms | 199.10 → 82.85 ms | 2.40× |
| 无触觉，50000 | 40.64 → 37.60 ms | 188.63 → 68.67 ms | 191.99 → 71.15 ms | 2.70× |
| VAE S / full / encoder，90000 | 46.50 → 43.42 ms | 205.03 → 86.75 ms | 209.10 → 90.77 ms | 2.30× |

上一版仅捕获文本 decoder 和分离动作头时，三种模型的部署入口分别为 94.04 / 82.57 / 102.18 ms；本次扩展后分别为 82.85 / 71.15 / 90.77 ms（不同轮次实测，非同轮三模式测量）。`--previous-graph` 可复现上一版捕获边界。仅捕获动作头的 DINOv2 初测，部署入口为 198.31 → 195.13 ms（1.016×）；整体收益主要来自 VLM Graph。

首次完整 prefix + chunk Graph 捕获及回放分别为 1095.69 / 1086.34 / 1160.56 ms，不含模型加载。微测动作头 Graph 另行捕获，其首次调用为 406.85 / 326.96 / 354.47 ms；实际部署直接使用完整 chunk Graph，不需要额外创建该微测 Graph。上述捕获耗时均未计入表中的稳定推理时间。

## 输出变化与验证

每个 checkpoint 比较 3 帧 × 3 组初始噪声（seed=0、1、42），共 27 组完整 `[1,32,8]` 动作块：归一化动作的最大绝对误差、平均绝对误差和 RMSE 均为 **0**，反归一化关节动作的最大绝对误差也为 **0**，全部逐元素相等。比较使用相同初始噪声，避免 flow-matching 采样随机性干扰。

还验证了每次调用仍生成新噪声、CUDA RNG 推进与 eager 一致、输入变化生效、已返回动作不被后续回放覆盖。4 个 CUDA 回归测试覆盖不同样本指令、同长度新指令、padding mask、位置偏移、序列长度变化、动态触觉输入、缓存上限，以及训练/更换权重/dtype 迁移时的 Graph 清理。三种模型的完整 prefix 和完整 chunk Graph 均各捕获 1 次，真实模型推理阶段没有回退；已有观测一致性与评测结果保存测试通过。

这些结果验证了本次样本的输出一致性；本次未运行模拟器成功率对比。

## 启用和复现

```bash
bash test_policy.sh --checkpoint /path/to/run --task lift_can --cuda-graph --headless
```

部署 YAML 设置 `use_cuda_graph: true`，或直接设置 `model.config.use_cuda_graph = True`。默认关闭。`model.config.cuda_graph_vlm = False` 可只捕获动作 chunk。`cuda_graph_full_vlm=False`、`cuda_graph_full_chunk=False` 可切回上一版边界。单个实例使用单一推理线程/stream；手动修改参数或结构后调用 `model.clear_cuda_graphs()`。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/data_ssd/yichi.zhang/miniconda3/envs/univtac/bin/python \
  scripts/benchmark_starvla_cuda_graph.py \
  --checkpoint policy_results/20261001-165913-264380040_starvla_groot_multi_task_all_encode_marker_rgb_DINOv2_S_cls_encoder/checkpoint_30000.pt \
  --episode .cache/control-audit/modelscope/isaac51/lift_can/hdf5/42.hdf5 \
  --iterations 20 --output .cache/cuda-graph-benchmark/dinov2-reference-30000.json

CUDA_VISIBLE_DEVICES=3 /mnt/data_ssd/yichi.zhang/miniconda3/envs/univtac/bin/python \
  -m unittest scripts.test_starvla_cuda_graph -v
```

完整逐次计时、输出差异和捕获/回放计数位于本地：

- [DINOv2 JSON](../../.cache/cuda-graph-benchmark/dinov2-reference-30000.json)
- [无触觉 JSON](../../.cache/cuda-graph-benchmark/none-reference-50000.json)
- [VAE full JSON](../../.cache/cuda-graph-benchmark/vae-reference-90000.json)

同目录对应的 `.outputs.pt` 文件保留了各帧、各 seed 的 eager/Graph 输出。`.cache` 中的原始实验产物不纳入版本控制。

## 参考文件记录

目录：`/mnt/data_ssd/yichi.zhang/lerobot-v0.5.1-test-cuda_graph/src/lerobot/policies/starvla_groot/`

| 文件 | SHA256 |
| --- | --- |
| `fast_vlm.py` | `a9536542bcc005ed56d23f88f19311b96afad4ec71fd779e4f62283a4ceb46f8` |
| `fast_dit.py` | `3486f0fbbae0da7fb74565f8d4ab5808e91c18c10ec5753daf15ef8e864eb8f8` |
| `fast_infer.py` | `97999a73f4af2479d032cc937258b0a25e5ae14824a3d098e761f862f6111370` |
