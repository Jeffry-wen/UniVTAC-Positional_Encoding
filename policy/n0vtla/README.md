# N0-VTLA in UniVTAC

This backend trains and serves from UniVTAC alone. It does not import or read the adjacent N0-VTLA checkout. Adapted source and attribution live in `native/` and [PROVENANCE.md](PROVENANCE.md).

Use the existing training environment (`requirements_train.txt`, Transformers 5.5); do not install upstream Transformers replacements or the upstream JAX/LeRobot training stack.

Required initialization assets:

- `resources/pretrained_models/n0-vtla-base/model.safetensors`: full base checkpoint, including DINOv2, predictor, projection and the learned `z_gate`.
- `resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer/`: local PaliGemma fast tokenizer. Override with `N0_TOKENIZER_PATH` if stored elsewhere.

```bash
POLICY=n0vtla TRAINING_MODE=single_task TASK=lift_can \
  ACTION_MODE=relative TACTILE_MODE=native TACTILE_INPUT_MODE=rgb_only \
  CHUNK_SIZE=50 BATCH_SIZE=8 STEPS=20000 bash train_policy.sh

POLICY=n0vtla TRAINING_MODE=multi_task ACTION_MODE=absolute \
  BATCH_SIZE=16 NUM_PROCESSES=2 bash train_policy.sh
```

N0 defaults to relative joint actions, native RGB tactile, and chunk 50 in the Bash launcher. Direct Python invocations must explicitly select `--action-mode relative --tactile-mode native --tactile-input-mode rgb_only`. `N0_PRETRAINED_PATH` overrides the base checkpoint directory. Native tactile accepts `rgb_only`, `marker_rgb`, `marker_only`, and `depth_deform_80`, `depth_deform_160`, `depth_deform_240`. The current and episode-first frames use the same selected representation before subtraction. `TACTILE_MODE=none` is a missing-touch control using the same model structure and false tactile masks; it does not remove the predictor.

State is eight absolute joint values. Action mode uses the unified absolute/relative conversion, with an absolute gripper. Normalization uses the current training set's quantiles, before padding to the native 32-dimensional model width. The base's EEF semantics must be adapted by post-training; it is not itself a joint-space policy.

Head/wrist map to the native base/left-wrist RGB slots; the absent right-wrist slot is masked. Left/right tactile sensors map to the two left-hand finger slots; the two absent right-hand slots are masked. Camera and tactile RGB images are resized with aspect-preserving black padding to 224; marker rasters and depth/deformation fields use the shared UniVTAC conversion at 224. Native model images are in [-1,1]; the tactile branch forms the current-minus-first-frame float difference, without additional ImageNet normalization. DINOv2 is frozen; its projection, predictor and action expert train through the action objective. The learned base gate is loaded unchanged.

Training strictly loads the full base (including the shared language embedding), then builds the optimizer. `base_config.yml` sets the learning rate, tactile LR multiplier and native prefix cache. Loss averages only the first eight dimensions and valid time steps. Validation disables image augmentation. The training file split, task balancing, DDP and checkpoint format are shared with the other unified policies.

Saved runs include the full model state (including DINOv2), normalization/action-mode metadata, native architecture settings and a bundled tokenizer. Deployment reconstructs the model offline and strictly loads the run; it needs neither the base checkpoint nor external backbone downloads.

```bash
bash test_policy.sh --checkpoint /path/to/run --inspect
bash test_policy.sh --checkpoint /path/to/run --task lift_can --offline
```

Online inference captures a tactile baseline after each reset and retains it for the episode. Offline HDF5 inference explicitly reads frame 0 even when predicting a later frame. Predicted relative chunks are restored against their original observation before execution.

Validated on A100 with the local base: one-step relative training, validation, checkpoint save/reload and 50x8 inference; one-step two-GPU absolute training on lift_can + insert_hole. An audited inference run blocked reads of the sibling N0-VTLA checkout, the original base checkpoint and original tokenizer directories; loading and inference succeeded from the saved run alone. These checks validate integration, not policy convergence or simulator task success.

CPU regressions: `python scripts/test_n0vtla.py`, `python scripts/test_joint_actions.py`, and `python scripts/test_policy_inputs.py`.
