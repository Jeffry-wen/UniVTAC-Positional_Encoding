"""Shared PI05 prompt encoding for training and deployment, without trainer imports."""
import numpy as np


def make_pi_tokens(batch, tokenizer, max_length):
    state = batch["observation.state"].detach().cpu().numpy()
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state, bins=bins) - 1
    prompts = [f"Task: {task}, State: {' '.join(map(str, row))};\nAction: "
               for task, row in zip(batch["task"], discretized)]
    encoded = tokenizer(prompts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    batch["observation.language.tokens"] = encoded["input_ids"]
    batch["observation.language.attention_mask"] = encoded["attention_mask"].bool()
