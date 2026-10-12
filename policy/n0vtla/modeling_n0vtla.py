"""Unified policy wrapper around the native N0 tactile latent pathway."""
import torch
from torch import nn
import torch.nn.functional as F

from .native.n0vtla_policy import N0VTLAPolicy
from .observation_adapter import make_observation, masked_action_loss


class N0Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Every runtime construction is followed by a strict base/run load.
        # Avoid repeatedly random-initializing the replaced HF backbones.
        from transformers.initialization import no_init_weights
        with no_init_weights():
            self.model = N0VTLAPolicy(config.native_config())
        self.model.paligemma_with_expert.gradient_checkpointing = config.gradient_checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def load_base(self, path):
        from safetensors.torch import load_model
        missing, unexpected = load_model(self.model, str(path / "model.safetensors"), strict=True)
        if missing or unexpected:
            raise ValueError(f"N0 base mismatch: missing={missing}, unexpected={unexpected}")
        print(f"Loaded N0 pretrained weights strictly: {path / 'model.safetensors'}; "
              f"z_gate={self.model.z_gate.detach().float().item():.9f}", flush=True)

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def optimizer_groups(self):
        base, tactile = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = tactile if name.startswith(("tactile_predictor.", "tactile_encoder.", "z_proj.")) else base
            group.append(parameter)
        return [{"params": base}, {"params": tactile, "lr": self.config.optimizer_lr * self.config.predictor_lr_scale}]

    def forward(self, batch):
        observation = make_observation(batch)
        actions = F.pad(batch["action"], (0, 24))
        # Native augmentation follows train/eval mode, so validation is unaugmented.
        loss = self.model(observation, actions)
        loss = masked_action_loss(loss, batch["is_pad"])
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.model.sample_actions(batch["observation.state"].device, make_observation(batch),
            num_steps=self.config.num_inference_steps)[..., :8]

    def reset(self):
        # Native per-call tactile stashes are refreshed by preprocessing.
        pass
