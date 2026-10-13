"""Train VAE-S at three deform ranges for 30 epochs, selecting best val loss."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def train(limit, gpu, run_id):
    mode = f'depth_deform_{limit}'
    output = ROOT / 'encoder_results' / mode / 'VAE' / 'S' / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHON_BIN=sys.executable)
    command = ['bash', str(ROOT / 'train_encoder.sh'), '--method', 'VAE',
               '--size', 'S', '--input', mode, '--', '--epochs', '30',
               '--early-stop-patience', '0', '--checkpoint-metric', 'val_loss',
               '--device', 'cuda', '--output-dir', str(output)]
    status = dict(method='VAE', size='S', input_mode=mode, gpu=gpu,
                  state='running', output=str(output), command=command)
    started = time.time()
    with (output / 'driver.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        status['pid'] = process.pid
        (output / 'status.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        code = process.wait()
    status.update(state='completed' if code == 0 else 'failed', exit_code=code,
                  seconds=round(time.time() - started, 1))
    metrics = output / 'metrics.jsonl'
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        validation = [row for row in rows if 'val_loss' in row]
        if validation:
            best = min(validation, key=lambda row: row['val_loss'])
            status.update(epochs_completed=validation[-1]['epoch'],
                          best_epoch=best['epoch'], best_val_loss=best['val_loss'])
    (output / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    return code


if __name__ == '__main__':
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(train, limit, gpu, run_id)
                for limit, gpu in ((80, 0), (160, 1), (240, 0))]
        failed = any([job.result() != 0 for job in jobs])
    raise SystemExit(int(failed))
