from dataclasses import dataclass


@dataclass(frozen=True)
class NativePiConfig:
    """Torch-only subset of upstream Pi0Config; avoids the JAX training stack."""
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    pi05: bool = True
    discrete_state_input: bool = True
    pytorch_compile_mode: str | None = None
    use_tactile: bool = False
    tactile_single_token: bool = False
    tactile_image_keys: tuple[str, ...] = (
        "left_wrist_left_tactile", "left_wrist_right_tactile",
        "right_wrist_left_tactile", "right_wrist_right_tactile",
    )


@dataclass
class N0PolicyConfig:
    chunk_size: int = 50
    n_action_steps: int = 16
    num_inference_steps: int = 10
    tokenizer_max_length: int = 200
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 2e-6
    predictor_lr_scale: float = 0.1
    use_prefix_cache: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.chunk_size < 1 or not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("N0 requires 1 <= n_action_steps <= chunk_size")
        if self.num_inference_steps < 1 or self.tokenizer_max_length < 1:
            raise ValueError("N0 inference steps and tokenizer length must be positive")
        if self.predictor_lr_scale <= 0:
            raise ValueError("predictor_lr_scale must be positive")

    def native_config(self):
        from .native.n0vtla_policy import N0VTLAConfig
        return N0VTLAConfig(action_horizon=self.chunk_size, max_token_len=self.tokenizer_max_length,
                            tactile_predictor_enabled=True, tactile_mode="latent", n_latent=5,
                            predictor_arch="tactile_kv", z_gate_zero_init=True,
                            vl_dropout_prob=0.0, predictor_loss_weight=0.0,
                            use_prefix_cache=self.use_prefix_cache)

    def get_scheduler_preset(self):
        from policy.vla_common.engine.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        return CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps, peak_lr=self.optimizer_lr, decay_lr=self.scheduler_decay_lr)
