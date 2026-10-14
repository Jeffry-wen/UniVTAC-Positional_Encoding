"""Compare historical BGR model inputs with the corrected RGB pipeline."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from encoder.image_io import LEGACY_ENCODING, image_encoding, read_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=ROOT / "resources/data/isaac51/grasp_classify/hdf5/0.hdf5")
    parser.add_argument("--frame", type=int, help="Defaults to the middle of the episode")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rgb_pipeline_comparison.png")
    args = parser.parse_args()
    keys = [
        ("Head camera", "observation/head/rgb"),
        ("Wrist camera", "observation/wrist/rgb"),
        ("Left tactile: rgb_only", "tactile/left_tactile/rgb"),
        ("Left tactile: marker_rgb", "tactile/left_tactile/rgb_marker"),
    ]
    rows = []
    metadata = {"episode": str(args.episode.resolve()), "before": "BGR", "after": "RGB", "streams": {}}
    with h5py.File(args.episode, "r") as h5:
        frame = args.frame if args.frame is not None else len(h5["embodiment/joint"]) // 2
        metadata["frame"] = frame
        for title, key in keys:
            dataset = h5[key]
            if image_encoding(dataset) != LEGACY_ENCODING:
                raise ValueError("Before/after comparison requires a historical UniVTAC episode")
            # Reproduce the pre-refactor reader: unconditional BGR2RGB after
            # OpenCV decoding. Render both model channel arrays as RGB.
            decoded = cv2.imdecode(np.frombuffer(dataset[frame], np.uint8), cv2.IMREAD_COLOR)
            before = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            after = read_rgb(dataset, frame)
            rows.append((title, before, after))
            metadata["streams"][key] = {"encoding": image_encoding(dataset),
                "before_channel_mean": before.mean((0, 1)).tolist(),
                "after_channel_mean": after.mean((0, 1)).tolist()}

    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [240, 80, 20]], np.uint8)
    swatches = np.repeat(np.repeat(colors[None], 60, axis=0), 100, axis=1)
    rows.append(("Synthetic reference: red | green | blue | orange", swatches[..., ::-1], swatches))
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 13),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, .35]})
    fig.patch.set_facecolor("#f3f5f7")
    for i, (title, before, after) in enumerate(rows):
        for j, image in enumerate((before, after)):
            axes[i, j].imshow(image, interpolation="nearest")
            axes[i, j].set_axis_off()
            axes[i, j].set_title(title, fontsize=11, pad=6)
    fig.suptitle("Visual + tactile model input: before / after\n"
                 "Before: BGR channels shown as RGB          After: correct RGB", fontsize=16, y=.985)
    fig.text(.5, .015, f"{args.episode.relative_to(ROOT) if args.episode.is_relative_to(ROOT) else args.episode} | frame {frame}\n"
             "Real decoded images before normalization; same frame and scale. Synthetic swatches are labeled separately.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .055, 1, .94), h_pad=1.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
