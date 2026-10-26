#!/usr/bin/env python3
"""Extract V1/V2 tactile features from a contact HDF5 or a NumPy observation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from encoder.modality import read_tactile
from encoder.tactile_rope_inference import load_tactile_rope, infer_fields, prepare_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted encoder.pth or last.pth')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--hdf5', type=Path)
    source.add_argument('--npz', type=Path, help='fields [5,H,W]/[B,5,H,W] or depth + markers [2,N,2]')
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.batch_size < 1 or args.image_size < 16 or args.image_size % 16:
        parser.error('Require start >= 0, positive count/batch-size, image-size a positive multiple of 16')
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    model, metadata = load_tactile_rope(args.checkpoint, args.device)
    outputs = {'global_features': [], 'patch_features': []}

    def consume(batch):
        for key, value in infer_fields(model, batch).items():
            outputs[key].append(value.numpy())

    if args.hdf5:
        with h5py.File(args.hdf5, 'r') as data:
            length = len(data[f'tactile/{args.side}_tactile/depth'])
            if args.start + args.count > length:
                parser.error(f'Requested frames exceed trajectory length {length}')
            for start in range(args.start, args.start + args.count, args.batch_size):
                consume(torch.stack([read_tactile(data, args.side, i, 'depth_deform_raw', args.image_size)
                                     for i in range(start, min(start + args.batch_size, args.start + args.count))]))
        indices = np.arange(args.start, args.start + args.count)
    else:
        with np.load(args.npz, allow_pickle=False) as data:
            if 'fields' in data:
                fields = torch.from_numpy(data['fields'].copy())
                if fields.ndim == 3:
                    fields = fields.unsqueeze(0)
            else:
                fields = prepare_observation(data['depth'], data['markers'], args.image_size).unsqueeze(0)
        if args.start + args.count > len(fields):
            parser.error(f'Requested frames exceed NPZ batch length {len(fields)}')
        for start in range(args.start, args.start + args.count, args.batch_size):
            consume(fields[start:min(start + args.batch_size, args.start + args.count)])
        indices = np.arange(args.start, args.start + args.count)
    metadata.update(source=str(args.hdf5 or args.npz), checkpoint=str(args.checkpoint),
                    side=args.side if args.hdf5 else None, feature_normalization='none',
                    patch_order='row-major', units='depth mm; u/v/x/y native pixels before configured calibration')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **{k: np.concatenate(v) for k, v in outputs.items()},
                            frame_indices=indices, metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps({'output': str(args.output), 'method': metadata['method'],
                      'shapes': {k: list(np.concatenate(v).shape) for k, v in outputs.items()}}))


if __name__ == '__main__':
    main()
