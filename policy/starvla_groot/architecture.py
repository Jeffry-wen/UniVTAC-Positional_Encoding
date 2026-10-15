"""Restore architecture defaults used by checkpoints predating expert widths."""


def restore_starvla_architecture(saved, policy):
    restored = dict(saved)
    if policy in ("starvla_groot", "starvla_pi"):
        # These are checkpoint-read fallbacks, not defaults for new training.
        restored.setdefault("chunk_size", 32)
        restored.setdefault("n_action_steps", 16)
        restored.setdefault("num_inference_timesteps", 4 if policy == "starvla_groot" else 10)
    if policy in ("starvla_groot", "starvla_pi") and "action_expert_hidden_size" not in restored:
        # None retains DiT-B/L's original width, or the PI backbone's width.
        restored["action_expert_hidden_size"] = None
        if policy == "starvla_pi":
            restored["legacy_layerwise_layout"] = True
    return restored
