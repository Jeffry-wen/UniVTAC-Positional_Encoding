# VAE full 四部分推理耗时（2026-10-12）

仅测试 `20261008-114326-417910781_starvla_groot_multi_task_all_encode_marker_rgb_VAE_S_full_encoder/checkpoint_90000.pt`，GPU 为 RTX 5090，PyTorch `2.7.0+cu128`，transformers `5.5.0`。任务 `lift_can`，使用 HDF5 第 0、10、20 帧循环测量，batch=1。

## 测量边界

每种模式预热 10 次后，交替执行 eager/Graph、Graph/eager，各统计 **100 次算术平均值**。每次计时前后同步 CUDA，测的是实际墙钟延迟，包括模型内部 Python 调度和 GPU 等待。首次捕获、预热均未计入，计时阶段捕获次数没有变化。

以下操作全部提前执行，不计入任何一项：HDF5/JPEG 读取、观测适配、CPU→GPU 搬运、图像及触觉 resize、归一化、图像 patchify、tokenizer、prompt/M-RoPE 准备、state 归一化、初始噪声生成。输出是归一化动作，**不做动作反归一化**。

“视觉 encoder”仅指 Qwen vision transformer；“VLM”仅指后续文本 decoder，二者没有重复计算。为独立归因，分项微测把视觉、文本分别捕获为 Graph；完整模型另行使用融合的 vision+text Graph，输入为已准备的 pixel values，仍不包含 patchify。触觉 encoder 在两种模式下均使用 eager，与当前部署一致。本次仅新增测量脚本，没有改变模型加速实现。

## 平均耗时

| 部分 | 原始 eager 平均 | 加速路径平均 | 加速比 | 节省时间 |
| --- | ---: | ---: | ---: | ---: |
| 视觉 encoder | 15.081 ms | 6.058 ms | **2.49×** | 9.024 ms |
| 触觉 encoder | 8.199 ms | 8.199 ms | **1.00×** | 0.000 ms |
| VLM 文本 decoder | 131.401 ms | 27.198 ms | **4.83×** | 104.203 ms |
| 动作 head | 49.209 ms | 45.981 ms | **1.07×** | 3.228 ms |
| **完整模型，不含预处理/反归一化** | **204.683 ms** | **88.028 ms** | **2.33×** | **116.654 ms** |

完整模型约从 4.89 次/s 提高到 11.36 次/s；这里是动作 chunk 推理频率，不是机器人控制频率。

分项平均值之和为 203.890 → 87.436 ms。独立分项同步、Graph 分界和完整流水线中的 token 合并等桥接操作，使其与另行测得的完整模型延迟不完全相等。完整模型行不是分项耗时的简单相加。

| 部分 | eager P95 | 加速路径 P95 |
| --- | ---: | ---: |
| 视觉 encoder | 15.138 ms | 6.076 ms |
| 触觉 encoder | 8.223 ms | 8.230 ms |
| VLM 文本 decoder | 133.548 ms | 28.499 ms |
| 动作 head | 49.303 ms | 46.031 ms |
| 完整模型 | 207.260 ms | 88.225 ms |

## 四部分分析

**视觉 encoder：15.08 → 6.06 ms，2.49×。** 两路 256×256 RGB 图像已在计时前转换成 `[512,1536]` pixel values，计时从 vision patch embedding 开始，经过 24 层视觉 transformer 和 merger。视觉 encoder 参数约 331.42M。Graph 复用固定图像布局的视觉位置编码、cu_seqlens 等常量，并一次提交视觉计算，避免每帧重复处理布局元数据和逐算子调度。没有把图像 resize、归一化或 patchify 的收益混入该数值。

**触觉 encoder：8.20 → 8.20 ms，1.00×。** 左右两路 `[1,3,224,224]` 输入分别经过共享的 VAE-S encoder、49-token latent 编码和输出投影，合并为 `[1,98,2048]`。参数约 25.55M，保留原 bf16 autocast。当前部署没有把触觉 encoder 纳入 CUDA Graph，因此两个测量函数执行相同代码，没有实际加速。VAE full 使用确定性的 mu token，不做训练时的随机 latent 采样。加速后这一部分约占四项独立耗时之和的 9.4%。

**VLM 文本 decoder：131.40 → 27.20 ms，4.83×，是主要收益来源。** 输入为已融合视觉、语言和 98 个触觉 token 的 `[1,244,2048]` embedding；包含 24 层 decoder，其中 18 层 linear attention、6 层 full attention。这里没有重新计算视觉 encoder，也不计算语言 logits 或 KV cache。当前环境未安装 FLA/causal-conv1d fast path，linear attention 使用带 Python 循环的 PyTorch 回退实现。Graph 复用准备好的 mask/位置元数据并消除大量逐算子提交开销；在独立分项节省的 116.45 ms 中，该部分贡献约 **89.5%**。

**动作 head：49.21 → 45.98 ms，1.07×，成为加速后的最大耗时项。** 保留 8 路 ensemble、4 步 Euler 去噪、16 层 DiT、chunk=32、action_dim=8 和原 float32 autocast。repeat、完整去噪和 ensemble mean 在同一张 Graph 中，初始噪声提前准备且两种模式完全一致。Graph 没有减少这些矩阵运算，实测只节省约 3.23 ms。相对较小的 Graph 收益表明，这一部分更接近 GPU 计算瓶颈；它占加速后四项独立耗时之和约 **52.6%**，高于文本 decoder 的 31.1%。进一步优化整体延迟，应优先研究 head；降低 ensemble 或改变精度会改变数值行为，不包含在本次测试中。

## 输出与计时有效性

三帧的视觉特征、触觉 token、文本 hidden state、动作 head 输出及完整流水线输出，在各自 eager/Graph 对比中均逐元素相等，最大绝对误差为 **0**。不含预处理的完整测量流水线还与正式 `predict_action_chunk` 的同噪声输出逐元素相等。

视觉、文本、无预处理融合 prefix、完整动作 chunk Graph 均只捕获一次，无 eager 回退；触觉明确保持 eager。所有捕获和输出验证均在稳定计时区间之外。

## 复现与原始数据

### 关闭 ensemble 的补充实测

同一 checkpoint、RTX 5090、三帧输入和测量边界，指定 `--ensemble 1` 关闭多路采样平均，仍保持 4 步去噪、16 层 DiT 和原计算精度。各模式预热 10 次后测量 100 次。仅覆盖本次测试的运行时配置，未修改部署默认值。

| 部分 | eager 平均 | CUDA Graph 路径平均 | 加速比 |
| --- | ---: | ---: | ---: |
| 视觉 encoder | 15.112 ms | 6.068 ms | 2.49× |
| 触觉 encoder | 8.133 ms | 8.155 ms | 1.00× |
| VLM 文本 decoder | 130.220 ms | 27.052 ms | 4.81× |
| 动作 head | 34.109 ms | 19.030 ms | **1.79×** |
| **完整模型，不含预处理/反归一化** | **188.688 ms** | **60.849 ms** | **3.10×** |

与此前 8 路 ensemble 相比，Graph head 从 45.981 降至 19.030 ms，降低约 58.6%；Graph 完整模型从 88.028 降至 60.849 ms，降低约 30.9%。ensemble=8 的 head 具有更大的并行批次，GPU 计算占比更高；ensemble=1 后减少调度的收益更明显，单靠 Graph 节省约 15.08 ms。这里是两次实测间的比较，视觉和 VLM 的小幅差异属于运行波动。没有用 GPU profiler 分离各内核耗时。

本次三帧的各部分及完整模型 eager/Graph 输出均逐元素一致，最大绝对误差为 0，完整流水线还与正式推理输出一致。head 已捕获并 replay，无 eager 回退。该一致性仅比较 ensemble=1 下的两种执行模式；关闭 ensemble 改变了采样平均策略，不能据此认为其输出或任务成功率与 ensemble=8 相同。

复现时将下方命令的 `--ensemble 8` 改为 `--ensemble 1`，并改用独立输出文件。当前部署已默认使用 CUDA Graph + ensemble=1；下方显式指定 8 以复现最初的测量。原始记录：[ensemble=1 JSON](../../.cache/cuda-graph-benchmark/vae-components-90000-ensemble1.json)。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/data_ssd/yichi.zhang/miniconda3/envs/univtac/bin/python -u \
  scripts/benchmark_starvla_components.py \
  --checkpoint policy_results/20261008-114326-417910781_starvla_groot_multi_task_all_encode_marker_rgb_VAE_S_full_encoder/checkpoint_90000.pt \
  --episode .cache/control-audit/modelscope/isaac51/lift_can/hdf5/42.hdf5 \
  --ensemble 8 --iterations 100 --warmup 10 \
  --output .cache/cuda-graph-benchmark/vae-components-90000.json
```

脚本：[benchmark_starvla_components.py](../../scripts/benchmark_starvla_components.py)。逐次计时、输入形状、输出差异、参数量和捕获/回放计数：[原始 JSON](../../.cache/cuda-graph-benchmark/vae-components-90000.json)。
