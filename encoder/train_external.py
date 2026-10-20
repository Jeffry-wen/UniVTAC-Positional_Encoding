"""Export tactile initializers or adapt official vision encoders on UniVTAC contact data."""
import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'encoder'))
from encoder.external import ExternalEncoder, resize_frames
from encoder.pretrained_registry import result_root, validate_identity
from encoder.linear_depth_deform import DEPTH_DEFORM_SPECS, depth_deform_spec
from encoder.modality import INPUT_CHANNELS
from encoder.checkpoint_io import save_checkpoint
from train_common import TactileFrames


class VisionAdaptation(nn.Module):
    """Masked teacher-feature distillation + tactile patch reconstruction.

    These are UniVTAC adaptation objectives, not reproductions of upstream training.
    Wan uses its pretrained decoder with reconstruction + KL instead.
    """
    def __init__(self, encoder, mask_ratio=.5, kl_weight=1e-6):
        super().__init__()
        self.encoder, self.mask_ratio, self.kl_weight = encoder, mask_ratio, kl_weight
        if encoder.method == 'Wan2.2-VAE':
            self.decoder = encoder._initial_decoder
            self.post_quant_conv = encoder._initial_post_quant_conv
            from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
            self.decoder_convs = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            from encoder.wan_recompute import enable_wan_recomputation
            enable_wan_recomputation(encoder.model.encoder)
            enable_wan_recomputation(self.decoder)
        else:
            self.teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
            dim = encoder.embed_dim
            self.predictor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.reconstruct = nn.Linear(dim, 3)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum=.996):
        if not hasattr(self, 'teacher'):
            return
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.lerp_(student, 1-momentum)

    def forward(self, images):
        # DataParallel worker threads need an explicit bf16 context of their own.
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            loss, metrics = self._forward(images)
        return loss, dict(metrics, _sample_count=loss.new_tensor(len(images)))

    def _forward(self, images):
        if self.encoder.method == 'Wan2.2-VAE':
            # DataParallel shallow-copies Python attributes. Each replica owns its
            # cache counters before encode/decode mutate them on separate threads.
            self.encoder.model._cached_conv_counts = dict(self.encoder.model._cached_conv_counts)
            posterior = self.encoder.posterior(images)
            model = self.encoder.model
            # Decoder is registered on the training objective, never on the exported encoder.
            model.decoder, model.post_quant_conv = self.decoder, self.post_quant_conv
            model._cached_conv_counts['decoder'] = self.decoder_convs
            try:
                decoded = model.decode(posterior.sample()).sample
                if self.encoder.num_frames == 4:
                    decoded = decoded[:, :, 1:5]
            finally:
                model.decoder, model.post_quant_conv = None, None
                model._cached_conv_counts['decoder'] = 0
            target = resize_frames(images, self.encoder.hw).transpose(1, 2)*2-1
            reconstruction = F.mse_loss(decoded.float(), target.float())
            kl = posterior.kl().mean() / posterior.mean[0].numel()
            return reconstruction + self.kl_weight*kl, dict(reconstruction=reconstruction.detach(), kl=kl.detach())
        h, w = images.shape[-2:]
        mask = torch.rand(images.shape[0], 1, 14, 14, device=images.device) < self.mask_ratio
        mask = F.interpolate(mask.float(), (h, w), mode='nearest')
        if images.ndim == 5:
            mask = mask[:, None]
        masked = images*(1-mask) + .5*mask
        tokens = self.encoder.patch_tokens(masked)
        with torch.no_grad():
            target = self.teacher.patch_tokens(images)
        feature_loss = F.mse_loss(F.layer_norm(self.predictor(tokens).float(), (tokens.shape[-1],)),
                                 F.layer_norm(target.float(), (target.shape[-1],)))
        # Align target RGB patch means with the backbone's spatial and temporal lattice.
        patch = 16
        gh, gw = self.encoder.hw[0]//patch, self.encoder.hw[1]//patch
        gt = tokens.shape[1]//(gh*gw)
        source = images.transpose(1, 2) if images.ndim == 5 else images[:, :, None]
        pixels = F.adaptive_avg_pool3d(source, (gt, gh, gw)).flatten(2).transpose(1, 2)
        reconstruction = F.mse_loss(self.reconstruct(tokens).float(), pixels.float())
        loss = feature_loss + reconstruction
        return loss, dict(feature=feature_loss.detach(), reconstruction=reconstruction.detach())


def reduce_replica_loss(loss, metrics):
    """Sample-weighted reduction also handles unequal validation replica sizes."""
    counts = metrics.pop('_sample_count')
    weights = counts / counts.sum()
    return (loss * weights).sum(), {key: (value * weights).sum() for key, value in metrics.items()}


def run(args):
    validate_identity(args.pretrained_mode, args.method, args.size)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if saved:
        for key, value in [('pretrained_mode', args.pretrained_mode), ('method', args.method),
                           ('size', args.size), ('input_mode', args.input_mode)]:
            if saved.get(key) != value:
                raise ValueError(f'Resume checkpoint {key} mismatch')
        if 'training_model' not in saved:
            raise ValueError('Resume requires last.pth with the full training state')
        if args.input_mode in DEPTH_DEFORM_SPECS and saved.get('depth_deform_encoding') != depth_deform_spec(args.input_mode):
            raise ValueError('Resume representation version mismatch')
    encoder = ExternalEncoder(args.pretrained_mode, args.method, args.size, metadata=saved,
                              pretrained=saved is None, with_decoder=True)
    out = args.output_dir or (args.resume.parent if args.resume else None) or result_root(args.input_mode, args.pretrained_mode, args.method, args.size) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(encoder.metadata, input_mode=args.input_mode, format_version=3)
    if args.input_mode in DEPTH_DEFORM_SPECS:
        metadata['depth_deform_encoding'] = depth_deform_spec(args.input_mode)
    if args.pretrained_mode == 'tactile':
        save_checkpoint(dict(metadata, encoder=encoder.state_dict()), out / 'encoder.pth')
        (out / 'config.json').write_text(json.dumps(metadata, indent=2))
        print(f'Exported trainable tactile initializer: {out / "encoder.pth"}', flush=True)
        return out
    files = sorted(args.data_root.glob('*/hdf5/*.hdf5'))
    if len(files) < 2:
        raise ValueError('Need at least two contact trajectories for file-level train/validation split')
    random.Random(args.seed).shuffle(files)
    nval = max(1, min(len(files)-1, round(len(files)*.2)))
    val_files, train_files = files[:nval], files[nval:]
    frames = encoder.num_frames
    train_ds = TactileFrames(train_files, round(args.samples*.8), seed=args.seed,
                            input_mode=args.input_mode, num_frames=frames)
    val_ds = TactileFrames(val_files, max(1, round(args.samples*.2)), seed=args.seed+1,
                          input_mode=args.input_mode, num_frames=frames)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed), **options)
    val_loader = DataLoader(val_ds, shuffle=False, **options)
    if not len(train_loader):
        raise ValueError('Encoder training requires at least 64 training anchors')
    core = VisionAdaptation(encoder, args.mask_ratio, args.kl_weight).to(args.device)
    model = core
    if args.multi_gpu:
        if not str(args.device).startswith('cuda') or torch.cuda.device_count() < 2:
            raise ValueError('--multi-gpu requires at least two visible CUDA devices')
        model = nn.DataParallel(core)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(metadata, train_files=list(map(str, train_files)), val_files=list(map(str, val_files)),
                  split_strategy='file', objective='reconstruction_kl' if args.method == 'Wan2.2-VAE' else 'masked_latent_and_patch_rgb',
                  train_count=len(train_ds), val_count=len(val_ds))
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    best, stale, step, start_epoch = float('inf'), 0, 0, 0
    if saved:
        core.load_state_dict(saved['training_model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        best, stale, step, start_epoch = saved['best_val_loss'], saved.get('stale_epochs', 0), saved['step'], saved['epoch']
        if 'rng_state' in saved:
            torch.set_rng_state(saved['rng_state'])
            train_loader.generator.set_state(saved['loader_rng_state'])
            if str(args.device).startswith('cuda'):
                torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
    epoch_steps = min(len(train_loader), args.max_steps or len(train_loader))
    total_steps = epoch_steps * args.epochs
    warmup = min(epoch_steps, max(1, total_steps//10))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, images in enumerate(train_loader):
            if args.max_steps and i >= args.max_steps:
                break
            started = time.monotonic()
            images = images.to(args.device)
            factor = (step+1)/warmup if step < warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(args.device).startswith('cuda')):
                loss, metrics = model(images)
                loss, metrics = reduce_replica_loss(loss, metrics)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite adaptation loss: {loss}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            core.update_teacher()
            step += 1
            record = dict(epoch=epoch+1, step=step, loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'],
                          batch_size=len(images), gpu_count=torch.cuda.device_count() if args.multi_gpu else 1,
                          seconds=time.monotonic()-started,
                          **{k: float(v) for k, v in metrics.items()})
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if step % 10 == 0 or step == 1:
                print(json.dumps(record), flush=True)
        model.eval()
        seen, total = 0, 0.
        rng_devices = (list(range(torch.cuda.device_count())) if args.multi_gpu else
                       [torch.device(args.device).index or 0]) if str(args.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            torch.manual_seed(args.seed+100000)
            for i, images in enumerate(val_loader):
                if args.max_val_steps and i >= args.max_val_steps:
                    break
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(args.device).startswith('cuda')):
                    loss, metrics = model(images.to(args.device))
                    loss, _ = reduce_replica_loss(loss, metrics)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                total += float(loss)*len(images)
                seen += len(images)
        value = total/seen
        improved = value < best
        best, stale = min(best, value), 0 if improved else stale+1
        state = dict(metadata, encoder=encoder.state_dict(), epoch=epoch+1, step=step, val_loss=value, best_val_loss=best)
        if improved:
            save_checkpoint(state, out / 'encoder.pth')
        if not args.no_training_state:
            save_checkpoint(dict(state, training_model=core.state_dict(), optimizer=optimizer.state_dict(),
                            training_config=config, stale_epochs=stale, rng_state=torch.get_rng_state(),
                            loader_rng_state=train_loader.generator.get_state(),
                            cuda_rng_states=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else []), out / 'last.pth')
        record = dict(epoch=epoch+1, step=step, val_loss=value, best_val_loss=best, best=improved)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if args.early_stop_patience and stale >= args.early_stop_patience:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretrained-mode', required=True, choices=['tactile', 'vision'])
    p.add_argument('--method', required=True)
    p.add_argument('--size', required=True)
    p.add_argument('--input-mode', choices=INPUT_CHANNELS, default='depth_deform_nonlinear_radial')
    p.add_argument('--data-root', type=Path, default=ROOT / 'resources/data/contact')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int)
    p.add_argument('--samples', type=int, default=100000)
    p.add_argument('--batch-size', type=int, choices=[64])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--mask-ratio', type=float, default=.5)
    p.add_argument('--kl-weight', type=float, default=1e-6)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--max-val-steps', type=int)
    p.add_argument('--early-stop-patience', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--multi-gpu', action=argparse.BooleanOptionalAction, default=None,
                   help='Split the fixed global batch of 64 across visible CUDA devices')
    p.add_argument('--no-training-state', action='store_true')
    args = p.parse_args()
    defaults = dict(epochs=5, lr=1e-5, weight_decay=.05, batch_size=64, multi_gpu=False)
    config_path = args.config or ROOT / 'encoder/vision_config.yml'
    if config_path.exists():
        defaults.update(yaml.safe_load(config_path.read_text()).get(args.method, {}))
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True).get('training_config', {})
        defaults.update({key: previous[key] for key in defaults if key in previous})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.batch_size != 64:
        p.error('All encoder training batches must be 64, including config/resume overrides')
    if args.epochs < 1 or args.batch_size < 1 or args.samples < 2:
        p.error('epochs/batch size must be positive and samples must be >=2')
    run(args)


if __name__ == '__main__':
    main()
