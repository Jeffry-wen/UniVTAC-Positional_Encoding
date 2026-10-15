# StarVLA-PI

Qwen-VL + layer-wise flow-matching DiT。实现已纳入当前仓库，运行、训练和部署均不依赖外部 qwen_pi 目录或其 LeRobot checkout。复用仓库内 StarVLA-GR00T 的 Qwen 接口、基础 DiT 组件、传感器路由和 processor。

## 模型

- 与 `starvla_groot` 相同的 `resources/pretrained_models/Qwen3.5-2B/` 预训练 VLM。
- 默认取所有文本层（当前 2B 模型为 24 层，hidden size 2048）；动作头各 block 依次 cross-attend 对应层。
- 新模型动作头默认宽度 `action_expert_hidden_size=1024`，cross-attention 的 K/V 输入维度保持 VLM 的 2048。新模型动作层数必须等于 VLM 文本层数；较少层的配置仅用于旧 checkpoint 兼容。
- 动作头训练和推理均使用 `autocast float32`，关闭 TF32；VLM 保持原有 BF16 精度。
- 动作头随机初始化，内部动作宽度 `max_action_dim=32`。真实 8D 动作先归一化再补齐；补齐维度和 `is_pad` 时间步不参与训练 loss。输出裁回真实宽度。
- State/action 使用 q01/q99 分位数归一化，多相机 RGB 图像使用 Qwen 预处理。
- 默认 chunk 50、执行 30 步、10 步 Euler 去噪、单次噪声采样。默认开启 CUDA Graph。

## 触觉

支持 `none`、`as_image`、`encode`，复用当前仓库的触觉 encoder（六种方法、cls/full、输入模态及时间窗口接口）。

- `as_image`：触觉作为额外图像视角进入 Qwen。
- `encode` + `encoder`：触觉 token 注入 Qwen 输入，经各文本层处理，逐层 hidden states 传给对应动作 block。
- `encode` + `decoder`：触觉 token 追加到每层 Qwen hidden states，并同步扩展 attention mask。

## 使用

```bash
POLICY=starvla_pi TACTILE_MODE=none TRAINING_MODE=single_task TASK=lift_can \
  CAMERAS=all EPISODES=2 STEPS=1 BATCH_SIZE=1 WORKERS=0 bash train_policy.sh

POLICY=starvla_pi TACTILE_MODE=encode TACTILE_INPUT_MODE=marker_rgb \
  ENCODER_METHOD=VAE ENCODER_SIZE=S TACTILE_TYPE=full \
  TACTILE_INSERT_LOCATION=encoder TRAINING_MODE=multi_task CAMERAS=all \
  bash train_policy.sh

bash test_policy.sh --checkpoint /path/to/run --task lift_can --offline
python scripts/test_starvla_pi.py
```

模型配置位于 `base_config.yml`。训练导出 run 内 Qwen config/processor；部署根据这些资源构建 VLM，再严格加载 checkpoint 内全部模型权重。无需复制 VLM 预训练权重到 run 目录。逐层动作头较大，训练 batch size 应根据显存实测设置。

旧 run/config 缺少 `action_expert_hidden_size` 时自动恢复原 VLM 同宽动作头（2B 为 2048），严格加载原权重；不会转换为 1024。旧的少层动作头也按原层数恢复。新 run 显式保存宽度，部署不受后续默认值变化影响。

## CUDA Graph 推理

默认 `use_cuda_graph=true`，捕获逐层动作头完整去噪、动作裁剪和输出类型转换，
以及受支持的 Qwen3.5 SDPA patchify → vision → 全部所选文本层。
触觉编码在捕获外执行；encoder/decoder 插入均支持。
完整 VLM Graph 缓存满后回落到文本层 Graph，文本或动作头缓存满后使用 eager。
不支持完整捕获的 VLM 也可使用动作头 Graph。

噪声在捕获外采样，每次推理使用新噪声；`predict_action_chunk(batch, noise=...)`
可传入固定的 `[B, chunk_size, max_action_dim]` 噪声以对照结果。
训练、加载权重、移动设备或 dtype 时会清空 Graph。手动修改模型参数或结构后调用
`model.clear_cuda_graphs()`。同一实例应由单一推理线程/stream 使用。

关闭加速可设置 `model.config.use_cuda_graph=False` 或部署参数 `use_cuda_graph: false`。
`cuda_graph_vlm=False` 仅捕获动作头；`cuda_graph_full_vlm=False` 使用文本层 Graph，
`cuda_graph_full_chunk=False` 仅捕获动作头去噪，将裁剪和输出转换留在捕获外。
默认每个缓存最多 2 个 Graph，首次调用需要预热和捕获。

```bash
python scripts/test_starvla_pi_cuda_graph.py
python scripts/benchmark_starvla_pi_cuda_graph.py \
  --checkpoint /path/to/run --episode /path/to/episode.hdf5 \
  --task lift_can --device cuda:0 --output /tmp/starvla_pi_graph.json
```

A100 默认结构的计算一致性和加速结果见 [CUDA_GRAPH_BENCHMARK.md](CUDA_GRAPH_BENCHMARK.md)。
