"""Migrate legacy standalone encoders into input/univtac/method/size/run.

V-JEPA standalone checkpoints are removed because its input architecture changed.
Policy-contained weights are deliberately untouched.
"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'encoder_results')
    p.add_argument('--apply', action='store_true', help='Apply migration; default prints operations only')
    args = p.parse_args()
    operations = []
    for representation in sorted(args.root.iterdir()):
        if not representation.is_dir():
            continue
        for method in ('ResNet', 'VAE', 'MAE', 'DINOv2', 'I-JEPA', 'V-JEPA'):
            source = representation / method
            if not source.is_dir():
                continue
            target = representation / 'univtac' / method
            if method == 'V-JEPA':
                operations.append(dict(delete=str(source), checkpoints=[str(p) for p in source.rglob('*.pth')]))
                if args.apply:
                    shutil.rmtree(source)
            else:
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite {target}')
                operations.append(dict(source=str(source), target=str(target)))
                if args.apply:
                    target.parent.mkdir(exist_ok=True)
                    source.rename(target)
                    for config in target.rglob('config.json'):
                        data = json.loads(config.read_text())
                        data['pretrained_mode'] = 'univtac'
                        for key in ('output_dir', 'resume'):
                            if data.get(key):
                                data[key] = data[key].replace(str(source), str(target))
                        config.write_text(json.dumps(data, indent=2))
    print(json.dumps(operations, indent=2))


if __name__ == '__main__':
    main()
