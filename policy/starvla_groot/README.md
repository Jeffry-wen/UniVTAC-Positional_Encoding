# StarVLA-GR00T

StarVLA-GR00T 的模型实现位于 `policy/starvla_groot/`，由 Qwen 视觉语言模型和 GR00T flow-matching DiT action head 组成。共用配置、优化器、触觉编码和 processor 在 `policy/vla_common/`。

## 环境与模型

环境安装遵循根目录 [README_yichi.md](../../README_yichi.md) 的 A100/A800 训练 Quick Start，统一使用 `requirements_train.txt`。统一训练入口默认读取 `resources/pretrained_models/Qwen3.5-2B/`。5090 模拟器测试环境和单回合 StarVLA 推理结果见 [README_5090.md](../../resources/docs/README_5090.md)。

## 训练与测试

从仓库根目录运行：

```bash
POLICY=starvla_groot TRAINING_MODE=multi_task EPISODES=0 CAMERAS=all \
  TACTILE_MODE=encode TACTILE_INPUT_MODE=marker_rgb \
  ENCODER_METHOD=ResNet ENCODER_SIZE=S TACTILE_TYPE=cls \
  TACTILE_INSERT_LOCATION=encoder bash train_policy.sh

bash test_policy.sh --checkpoint /path/to/run --inspect
bash test_policy.sh --checkpoint /path/to/run --task lift_can --offline
```

当前统一训练使用绝对 8D 关节 state/action，动作从 `t+1` 开始；默认 chunk 为 50，执行 30 步。触觉支持 `none`、`as_image`、`encode`。`encode` 使用本仓库训练的六种 encoder，支持 `cls/full` 和 `encoder/decoder` 插入位置。

多卡训练使用 `NUM_PROCESSES`，`BATCH_SIZE` 是全局 batch；四个八卡机器的批量启动脚本仍位于 `scripts/`。完整参数和 checkpoint 约定见 [TRAINING.md](../TRAINING.md)。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `configuration_starvla_groot.py` | 模型配置与输入输出契约 |
| `modeling_starvla_groot.py` | 模型、训练 loss 和动作推理 |
| `qwen_vl_interface.py` | Qwen 多模态接口 |
| `action_head/` | GR00T DiT 动作头 |
| `processor_starvla_groot.py` | processor 定义 |
| `base_config.yml` | 模型与优化器默认配置 |

该实现由 StarVTLA / starVLA 代码适配，源代码保留相关版权及许可声明。

## CUDA Graph 推理

StarVLA 默认启用 CUDA Graph，并使用单次采样（`repeated_diffusion_steps=1`，取消 ensemble），新模型默认 10 步去噪。旧 checkpoint 按保存的配置恢复去噪步数；历史 GROOT 配置未记录该字段时回退到 4 步，无需转换权重。Graph 捕获 GR00T 完整去噪，以及受支持的 Qwen3.5 patchify → vision → text decoder；如显式启用 ensemble，也捕获 repeat 和 ensemble mean。实现已对照本地 `lerobot-v0.5.1-test-cuda_graph` 的 FastVLM / FullChunkGraph 适配。

```bash
bash test_policy.sh --checkpoint /path/to/run --task lift_can --cuda-graph --headless

# 离线验证，不启动模拟器
bash test_policy.sh --checkpoint /path/to/run --task lift_can \
  --offline --hdf5 /path/to/episode.hdf5 --cuda-graph
```

不再需要传 `--cuda-graph`；该参数保留兼容。部署 YAML 可显式设置 `use_cuda_graph: false` 关闭 Graph，或设置 `repeated_diffusion_steps: 8` 恢复 8 路 ensemble；直接使用模型时可修改对应的 `model.config` 字段。只捕获动作头可另外设置 `model.config.cuda_graph_vlm = False`；设置 `cuda_graph_full_vlm=False` 和 `cuda_graph_full_chunk=False` 可使用上一版的文本 decoder / 分离动作头 Graph。新指令或输入形状首次调用会预热并捕获；每个模块默认最多保存 2 个 Graph，完整 VLM 缓存满后先尝试文本 decoder Graph，再回落 eager。切换训练、加载权重、移动模型设备或 dtype 时会清理 Graph；手动修改参数或模型结构后请调用 `model.clear_cuda_graphs()`。单个模型实例应由单一推理线程/stream 使用。

2026-10-12 对照参考代码调整后，在 RTX 5090 上，三种现有 checkpoint 的部署入口实测加速 **2.30–2.70 倍**，27 组同输入、同初始噪声对比的最大输出差异为 **0**。详细计时口径、首次捕获开销和复现命令见 [CUDA_GRAPH_BENCHMARK.md](CUDA_GRAPH_BENCHMARK.md)。

仅 VAE full 的纯模型四部分平均耗时（各 100 次，不含观测预处理/动作反归一化），见 [VAE_FULL_COMPONENT_BENCHMARK.md](VAE_FULL_COMPONENT_BENCHMARK.md)。
