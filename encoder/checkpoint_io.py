"""Publish complete encoder files atomically for concurrent policy discovery."""
import os
from pathlib import Path
import torch


def save_checkpoint(payload, path):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
