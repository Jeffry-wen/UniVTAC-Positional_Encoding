"""Install the pinned official Sparsh source used by the tactile encoder adapter."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.pretrained_registry import ROOT, SPARSH_REVISION


def ensure_sparsh():
    path = ROOT / 'resources/third_party/sparsh'
    if not path.exists():
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/sparsh.git', str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', SPARSH_REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SPARSH_REVISION:
        raise RuntimeError(f'Sparsh source version mismatch: {actual}, expected {SPARSH_REVISION}')
    return path


if __name__ == '__main__':
    print(ensure_sparsh())
