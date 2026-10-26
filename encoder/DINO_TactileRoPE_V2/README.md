# DINO触觉RoPE V2

V2 retains V1's raw five-channel input and DINO/iBOT/KoLeo training objectives.
No reconstruction decoder, deformation/ranking head, or auxiliary loss is added.

Each block uses a rest increment `(R(u,v,h)-I) W_s LN(x)`, a bias-free static
output projection, and partial rotary Q/K attention. Each 64-dimensional head
allocates 24 channels to unrotated content, 16 to x/y and 24 to u/v/h. Static,
context and MLP updates have independent learnable per-channel LayerScale
parameters initialized to 0.001. All twelve blocks and mean global pooling are
retained. Masked student tokens hide both deformation and validity information.

This first run deliberately preserves V1 preprocessing: h = 30 mm - depth,
u/v/x/y are pixels, and normalization scales are fixed. The 30 mm reference is
an assumption, not measured indentation; no physical cross-sensor claim follows.
No verified dataset-wide surface-validity mask or measured rest depth map is
available. `invalid_depth_ge_mm: null` therefore preserves finite background
values instead of inventing a validity threshold. The encoder supports an
explicit upper depth threshold and missing measurements, valid-weighted patch
means and a missingness embedding. A future calibrated preprocessing comparison
must rerun V1 with the identical correction. Static reference maps are not part
of this five-channel input version.

Run on one visible BF16-capable GPU:

```bash
bash scripts/train_dino_tactile_rope_v2_s_1gpu.sh
```

Defaults: ViT-S/16, global batch 64, 100,000 sampled frames, file-disjoint 80/20
split, seed 42, 30 epochs, AdamW lr 1e-4, weight decay .04, BF16, one warmup epoch,
EMA .996, kNN checkpoint selection and patience 5, matching V1. Patience can stop
before epoch 30. The single GPU processes the entire global batch; no accumulation
or distributed Sinkhorn approximation is used. Results go to a distinct
`encoder_results/depth_deform_raw/univtac/DINO_TactileRoPE_V2/S/<timestamp>`.
`encoder.pth` is the best encoder; `last.pth` contains optimizer/teacher state.

Resume using the original output directory:

```bash
bash scripts/train_dino_tactile_rope_v2_s_1gpu.sh -- --resume /absolute/run/last.pth --output-dir /absolute/run
```

Architecture metadata is `tactile_rope_v2`; V1 checkpoints are not compatible.
Load with `DINO_TactileRoPE_V2.encoder.load_encoder` with `encoder/` on PYTHONPATH.
Tests: `python scripts/test_dino_tactile_rope_v2.py`, plus existing V1 and token
parallel tests. V2 tests cover static identity, relative attention algebra,
masked-value/missingness leakage, gradients, resolution changes and checkpoint
round trips. Low reconstruction error alone does not establish useful learned
representations; evaluate identical frozen probes and downstream ACT separately.
