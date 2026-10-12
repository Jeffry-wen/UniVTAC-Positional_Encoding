# N0-VTLA source provenance

The files in `native/` were adapted from the local NeoteAI N0-VTLA checkout at
commit `03a0ce4d7091ca2354864796770715aa212601b7`, corresponding to the
[N0-VTLA project](https://github.com/neoteai/N0-VTLA).
Runtime code uses repository-local imports only; the original checkout is not required.

Copied sources: `n0vtla/models_pytorch/{n0vtla_policy,pi0_pytorch,gemma_pytorch,tactile_encoder,tactile_predictor,preprocessing_pytorch}.py`.
`native/image_tools.py` extracts the Torch resize helper from `n0vtla/shared/image_tools.py`;
`native/gemma_config.py` extracts the Big Vision/OpenPI model dimensions.

Changes for UniVTAC:

- Replace upstream imports and JAX-dependent configuration/data utilities with local Torch-only equivalents.
- Use existing `policy.vla_common.pi_gemma` AdaRMS classes with Transformers 5.5, without replacing any installed Transformers files.
- Restore tied language embeddings and keep native image features unscaled by calling the vision tower and projector directly.
- Preserve the upstream SigLIP FP32 embedding/BF16 encoder transition, and use native per-layer prefix tensors as read-only denoising caches instead of the mutable HF cache API.
- Construct DINOv2-base from its offline architecture (`image_size=518`, matching the 1370-position weight tensor); load all backbone weights from the full checkpoint.
- Respect model train/eval mode for augmentation. Preserve the native tactile difference, token pooling, predictor, gate, and flow-matching sampler.
- The unified wrapper adds valid-time/8D action loss masking and omits frozen parameters from the optimizer.

The upstream LICENSE and NOTICE and the Apache-2.0 text are retained in `native/`.
The runtime integration does not change the model weights' Gemma terms, recorded in the local model README.
