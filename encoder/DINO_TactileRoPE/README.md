# DINO触觉RoPE

代码方法名：`DINO_TactileRoPE`。本版本仅增加双分支编码器，沿用 DINO 的
CLS 自蒸馏、masked-patch/iBOT 式自蒸馏、KoLeo 和 EMA teacher；没有形变回归、
接触排序头或任何辅助监督。kNN 仍只用于原有 checkpoint 选择。

## 启动

在仓库根目录运行：

```bash
bash scripts/train_dino_tactile_rope.sh
# ViT-B
ENCODER_SIZE=B bash scripts/train_dino_tactile_rope.sh
# 短程验证；放在 .cache 中，避免被当作正式训练结果。
bash scripts/train_dino_tactile_rope.sh -- --epochs 1 --max-steps 2 \
  --max-val-steps 1 --samples 512 --workers 2 \
  --probe-per-class-train 4 --probe-per-class-val 4 \
  --output-dir .cache/dino-tactile-rope-smoke
```

环境默认为 `python`，可用 `PYTHON_BIN` 覆盖。
依赖与 `requirements_dino.txt` 相同。默认 S、100000 帧、batch 64、BF16、最多
30 轮、验证 kNN 连续 5 轮不改善则早停。
结果保存在 `encoder_results/depth_deform_raw/univtac/DINO_TactileRoPE/<S或B>/`。

## 输入和单位

输入模式必须为 `depth_deform_raw`，不能使用 JPEG/RGB 或截断后的 deformation 图。
HDF5 中读取 `depth` 和 `marker_full`（优先）/`marker`，后者包含静止和当前坐标。
先在参考 marker 坐标上 TPS 插值 `current-reference`，保留未截断的位移。
打包五通道 `[camera_depth_mm, u_px, v_px, x_px, y_px]`。额外 x/y 只供坐标使用，
不进入 patch embedding。所有通道一起裁剪/缩放，原始坐标值和位移值不因缩放而改变。
没有亮度增强、翻转、旋转、数值裁剪或逐帧最大值归一化。

配置 `model.rope`：

- `reference_depth_mm: 30.0`：沿用仓库现有假设，h = reference_depth - camera_depth。
  保留负值和大形变，不截断；并非重新测得的静止标定。
- `mm_per_pixel: null`：当前没有标定，x/y/u/v 明确使用像素，h 使用毫米。
- `spatial_scales: [160, 120]`：x/y 的固定尺度，当前单位为像素。
- `deformation_scales: [20, 20, 5]`：u/v/h 固定尺度，当前单位为像素/像素/毫米。
- `min_frequency: 0.01`, `max_frequency: 1.0`：归一化坐标的多尺度旋转频率。

将来有标定时，可设置 `mm_per_pixel: [sx, sy]`，这会把 x/y/u/v 转为毫米；
同时将 spatial/deformation scales 改成希望跨传感器共用的毫米尺度。
不要把每个传感器各自的最大形变当作归一化尺度。本版本支持单个全局线性标定，
不自动解决不同左右传感器、非线性投影、曲面坐标或材料差异。
现有 depth 在图像网格上采样、切向位移在参考 marker 网格上插值，是仓库传感器
表示的近似，尚未做完整三维材料点追踪。不能宣称已验证跨传感器物理泛化。

## 每层计算

三通道 patch embedding 输入为归一化的 `[u,v,h]`；每个 patch 的坐标为
`[x/Lx, y/Ly, mean(u)/Du, mean(v)/Dv, mean(h)/Dh]`。均值不代替完整 patch 特征。

```
f = LayerNorm(x)
s = RoPE_uvh(Linear(f), d - 0)          # 静态参照分支
q, k, v = QKV(f)
c = Attention(RoPE_xyuvh(q), RoPE_xyuvh(k), v)  # 当前区域比较
x = x + Linear(concat(s, c))
x = x + MLP(LayerNorm(x))
```

每层、每个 head 独立按通道对分配轴，旋转相位使用 float32 计算。
没有绝对位置表、参考 token 或额外幅值 MLP。S 为 384 维/6 heads，B 为
768 维/12 heads，均 12 层。全局表示取最终 patch 特征均值；接口将其放在
第 0 token 供原有 DINO 全局 head 使用，随后是 196 个 patch tokens。
原版 DINO 的可学习 CLS 与本版平均池化不同，因此比较结果时需考虑该架构差异。

学生 masked patch 在 embedding 前清零形变并替换为 mask token，两种旋转的
u/v/h 坐标也置零；保留空间 x/y。teacher 接收同一裁剪的完整输入。
避免 masked patch 通过侧路直接获得其形变。测试验证改变被遮挡 patch 的原始
形变不会改变学生输出。

## Checkpoint 和恢复

`encoder.pth` 是选中的编码器，`last.pth` 包括完整模型、teacher、优化器。
二者保存 `architecture=tactile_rope_v1` 和 `model_config`（包括标定和频率）。
恢复时拒绝 model_config 不匹配：

```bash
bash scripts/train_dino_tactile_rope.sh -- --resume /absolute/path/last.pth \
  --output-dir /absolute/path/existing-run
```

使用不同配置时在 `--` 前传 `--config /path/to/config.yml`。模型结构与旧版 ViT
权重不同，不支持直接加载旧 DINO 权重；本次只接入 encoder 训练，未接入 policy
的通用编码器加载器。独立推理使用显式加载函数：

```python
import sys
sys.path.insert(0, 'encoder')
from DINO_TactileRoPE.encoder import load_encoder
model = load_encoder('/absolute/path/encoder.pth', device='cuda')
# raw: [B, 5, 224, 224]，遵守上述通道、坐标、单位约定
# features = model(raw.cuda())
```

## 检查

```bash
python scripts/test_dino_tactile_rope.py
```

这些检查验证计算、裁剪、遮挡和梯度，不证明训练收敛或接触强弱判断。

## 单机八卡、保持全局 batch 64

```bash
bash scripts/train_dino_tactile_rope_s_8gpu.sh
# 查看命令，不启动训练，也不要求本机存在八张卡：
bash scripts/train_dino_tactile_rope_s_8gpu.sh --dry-run
```

脚本固定 S、raw 输入、全局 batch 64、默认 CUDA 0–7；配置继续读取本目录的
`config.yml`。默认 lr=1e-4、100000 帧、最多 30 轮、早停 patience=5、seed=42，
不因八卡放大学习率或全局 batch。可以用 `PYTHON_BIN` 指定八卡机器上的环境。

这是单进程的 encoder-token DataParallel，不使用 torchrun。每个视图的 64 张图
分到八卡，各卡 8 张；Student/Teacher 编码特征汇总到主卡后计算全局 DINO head、
Sinkhorn、KoLeo 和 loss，避免各卡分别在 8 个样本上计算导致训练目标变化。
EMA 在每次 optimizer.step 后由原模型统一更新。并行适配器不改变 checkpoint
参数名，可以在单卡/八卡间加载。kNN 探针仍使用主卡。

全局 heads、损失和数据读取仍可能成为瓶颈，不保证八倍加速。浮点归约与随机
增强的数值结果不保证跨设备逐位一致。脚本会检查恰有八张可见 CUDA GPU 且支持
BF16；不足八卡时直接退出，不会静默降为单卡或启动八个重复训练进程。

```bash
python scripts/test_dino_token_parallel.py
```

测试验证分片的 mask token 梯度、完整 batch 损失和 checkpoint 参数名。CUDA 测试
使用当前机器的可见 GPU；只有在八卡机器上执行才能验证真实八卡通信。
