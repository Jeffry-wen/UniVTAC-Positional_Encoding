# 触觉 RoPE：V1 / V2 训练与推理

两个版本均以 DINO 全局蒸馏、masked patch 蒸馏、KoLeo 和 EMA teacher 训练。
没有形变回归、接触排名或图像重建辅助监督。V1 使用双分支旋转与拼接融合；
V2 使用静止增量 `(R(d)-I)`、内容/空间/形变分组 QK 旋转和 LayerScale 残差。
细节见 [V1](DINO_TactileRoPE/README.md) 和 [V2](DINO_TactileRoPE_V2/README.md)。

## 环境

在仓库根目录，使用 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_dino.txt
```

脚本默认使用 PATH 中的 `python`，可通过 `PYTHON_BIN` 指定环境。
数据、训练权重与运行日志不包含在代码仓库中。

## 训练

Contact HDF5 放在 `resources/data/contact/<shape>/hdf5/*.hdf5`，或传 `--data-root`。

```bash
# V1 单卡
CUDA_VISIBLE_DEVICES=0 bash scripts/train_dino_tactile_rope.sh
# V1 八卡：global batch 64，编码器每卡 8 张，损失在主卡使用完整 batch
bash scripts/train_dino_tactile_rope_s_8gpu.sh
# V2 单卡
CUDA_VISIBLE_DEVICES=0 bash scripts/train_dino_tactile_rope_v2_s_1gpu.sh
# V2 多卡沿用相同 token-parallel 训练机制
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_encoder.sh \
  --method DINO_TactileRoPE_V2 --size S --input depth_deform_raw -- --multi-gpu
```

默认 100000 帧、按文件 80/20 划分、batch 64、最多 30 epochs、BF16、AdamW，
验证 kNN 连续 5 轮不改善时早停。S 学习率 1e-4。配置在各方法 `config.yml`。
`encoder.pth` 为 kNN 最优 encoder，`last.pth` 包含恢复训练需要的状态。

```bash
bash scripts/train_dino_tactile_rope_v2_s_1gpu.sh -- \
  --resume /path/to/run/last.pth --output-dir /path/to/run
```

## 推理

统一入口自动识别 V1 / V2。只加载你信任的 PyTorch checkpoint。
以下提取连续 8 帧的左触觉特征，不使用随机裁剪或 teacher 蒸馏头：

```bash
python scripts/infer_dino_tactile_rope.py \
  --checkpoint /path/to/encoder.pth \
  --hdf5 resources/data/contact/shape/hdf5/trajectory.hdf5 \
  --side left --start 0 --count 8 --device cuda \
  --output /path/to/features.npz
```

HDF5 使用 `tactile/{left,right}_tactile/depth` 与 `marker_full` 或 `marker`。
输出包含：

- `global_features`: `[帧数,384]`（S）或 `[帧数,768]`（B）；
- `patch_features`: `[帧数,196,维度]`，默认 224×224、patch 16，行优先；
- `frame_indices`：输入帧索引；`metadata`：JSON 字符串，包含模型与输入信息。

特征不做额外 L2 归一化。输出已有时会报错，避免覆盖。
`--image-size` 可指定 16 的正整数倍；不同分辨率的精度需要独立验证。

也支持 NPZ：包含 `fields`，形状为 `[5,H,W]` 或 `[B,5,H,W]`；或单帧
`depth [H,W]` 和 `markers [2,N,2]`，后者第一组为静止参考、第二组为当前坐标。

```bash
python scripts/infer_dino_tactile_rope.py --checkpoint /path/to/encoder.pth \
  --npz /path/to/observation.npz --count 1 --device cpu --output /path/to/features.npz
```

五通道 `fields` 顺序为 `[depth_mm,u_px,v_px,x_px,y_px]`，不可预先归一化，
不可传 RGB 或压缩后的形变图。NPZ 的 `--start/--count` 选择 batch 中的帧，默认一帧。
图像缩放时保留原始传感器的坐标与位移数值。Python API：

```python
from encoder.tactile_rope_inference import load_tactile_rope, prepare_observation, infer_fields
model, metadata = load_tactile_rope('/path/to/encoder.pth', device='cuda')
fields = prepare_observation(depth_mm, reference_and_current_markers)
features = infer_fields(model, fields)
```

这是触觉 encoder 推理，输出特征；不是 ACT 动作预测或图像解码器。两个版本尚未
接入 policy 通用加载器。当前默认 h=30-depth 为假定参考深度差，u/v 为像素位移，
没有静止表面与横向毫米标定，不能据此声称已经实现跨传感器物理泛化。

## 验证

```bash
python scripts/test_dino_tactile_rope.py
python scripts/test_dino_tactile_rope_v2.py
python scripts/test_dino_token_parallel.py
python scripts/test_tactile_rope_inference.py
```

单卡 A100 上完成两个版本的实际 checkpoint 推理验证；八卡仅提供启动机制，
本地没有八卡硬件，因此不声称已完成八卡实测。
