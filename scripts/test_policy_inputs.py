"""Check collection -> training and live/offline deployment input parity."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from encoder.image_io import ENCODING_ATTR, STANDARD_ENCODING, read_rgb, require_rgb_model
from encoder.modality import read_tactile
from policy.n0vtla.observation_adapter import read_native_tactile
from policy.train_unified import UniVTACDataset
from policy.unified_deploy import Policy
from scripts.test_policy import hdf5_observation

# Load the data writer without importing Isaac Sim's environment utilities.
spec = importlib.util.spec_from_file_location("collection_data", ROOT / "resources/envs/utils/data.py")
collection_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection_data)


class PolicyInputTests(unittest.TestCase):
    def test_training_live_and_offline_parity(self):
        for kind, image_size in (("act", 256), ("pi05", 224), ("starvla_groot", 224), ("n0vtla", 224)):
            for mode in ("marker_rgb", "rgb_only"):
                with self.subTest(policy=kind, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    self.check_parity(Path(tmp) / "demo.hdf5", kind, image_size, mode)

    def test_primary_color_channel_order(self):
        for kind, image_size in (("act", 256), ("pi05", 224), ("starvla_groot", 224), ("n0vtla", 224)):
            with self.subTest(policy=kind), tempfile.TemporaryDirectory() as tmp:
                self.check_parity(Path(tmp) / "primary.hdf5", kind, image_size, "marker_rgb",
                                  colors=([255, 0, 0], [0, 255, 0], [0, 0, 255], [240, 80, 20]))

    def test_historical_data_is_rgb(self):
        for kind, size in (("act", 256), ("n0vtla", 224)):
            for mode in ("rgb_only", "marker_rgb"):
                with self.subTest(policy=kind, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    self.check_parity(Path(tmp) / "legacy.hdf5", kind, size, mode, legacy=True)

    def test_unversioned_and_bgr_models_are_rejected(self):
        for metadata in ({}, {"color_order": "BGR"}):
            with self.assertRaises(ValueError):
                require_rgb_model(metadata)
        require_rgb_model({"color_order": "RGB"})

    def check_parity(self, path, kind, image_size, mode, colors=None, legacy=False):
        cameras = ["head", "wrist"]
        camera_keys = ["observation.images.cam_high", "observation.images.cam_wrist"]
        tactile_key = "rgb_marker" if mode == "marker_rgb" else "rgb"
        colors = colors or ([220, 80, 20], [30, 120, 240], [50, 180, 100], [160, 40, 200])
        images = [np.full((24, 32, 3), color, np.uint8) for color in colors]
        joints = np.arange(27, dtype=np.float32).reshape(3, 9) / 100
        observation = {
            "embodiment": {"joint": torch.from_numpy(joints[0])},
            "observation": {name: {"rgb": torch.from_numpy(images[i])} for i, name in enumerate(cameras)},
            "tactile": {side + "_tactile": {tactile_key: torch.from_numpy(images[i+2])}
                        for i, side in enumerate(("left", "right"))},
        }
        collected = {
            "embodiment": {"joint": joints},
            "observation": {name: {"rgb": [images[i]] * 3} for i, name in enumerate(cameras)},
            "tactile": {side + "_tactile": {tactile_key: [images[i+2]] * 3}
                        for i, side in enumerate(("left", "right"))},
        }
        with h5py.File(path, "w") as h5:
            collection_data.HDF5Handler().dict_to_hdf5(h5, collected)
            streams = [(f"observation/{name}/rgb", images[i])
                       for i, name in enumerate(cameras)]
            streams += [(f"tactile/{side}_tactile/{tactile_key}", images[i + 2])
                        for i, side in enumerate(("left", "right"))]
            for prefix, image in streams:
                if legacy:
                    del h5[prefix]
                    _, encoded = cv2.imencode(".jpg", image)
                    h5.create_dataset(prefix, data=[encoded.tobytes()] * 3, dtype=f"S{len(encoded)}")
                else:
                    self.assertEqual(h5[prefix].attrs[ENCODING_ATTR], STANDARD_ENCODING)
                np.testing.assert_allclose(read_rgb(h5[prefix], 0)[0, 0], image[0, 0], atol=1)
                decoded = collection_data.HDF5Handler().gather_hdf5(h5, [(prefix, None)])
                np.testing.assert_allclose(decoded[prefix][0, 0, 0], image[0, 0], atol=1)
                if not legacy:
                    bgr = cv2.imdecode(np.frombuffer(h5[prefix][0], np.uint8), cv2.IMREAD_COLOR)
                    np.testing.assert_allclose(bgr[0, 0], image[0, 0, ::-1], atol=1)
            for side in ("left", "right"):
                expected = torch.tensor(colors[2 if side == "left" else 3], dtype=torch.float32) / 255
                for tensor in (read_tactile(h5, side, 0, mode), read_native_tactile(h5, side, 0, mode)):
                    torch.testing.assert_close(tensor[:, 112, 112], expected, rtol=0, atol=1.01 / 255)
        stats = {name: {"mean": [0.2] * 8, "std": [0.3] * 8, "q01": [0.] * 8, "q99": [1.] * 8} for name in ("state", "action")}
        dataset = UniVTACDataset({"lift_can": [path]}, cameras=cameras, tactile_mode="native" if kind == "n0vtla" else "encode",
                                tactile_input_mode=mode, chunk_size=2, stats=stats,
                                policy=kind, image_size=image_size)
        try:
            policy = Policy.__new__(Policy)
            policy.kind, policy.image_size, policy.stats = kind, image_size, stats
            policy.task_name, policy.camera_keys = "lift_can", camera_keys
            policy.tactile_mode, policy.tactile_input_mode = ("native" if kind == "n0vtla" else "encode"), mode
            policy.tactile_baselines = {}
            policy.tokenizer, policy.device = None, torch.device("cpu")
            run = {"camera_keys": camera_keys,
                   "options": {"tactile_mode": policy.tactile_mode, "tactile_input_mode": mode}}
            train = dataset[0]
            live = policy.encode_obs(observation)
            policy.tactile_baselines = {}
            offline = policy.encode_obs(hdf5_observation(path, 0, run))
            self.assertEqual(live["task"], [train["task"]])
            for key, expected in train.items():
                if not key.startswith("observation."):
                    continue
                torch.testing.assert_close(offline[key][0], expected, rtol=0, atol=0)
                # Online images skip JPEG compression, which can shift each
                # channel of these constant-color patches by one intensity.
                tolerance = 1.01 / 255 if ".images." in key else 0
                torch.testing.assert_close(live[key][0], expected, rtol=0, atol=tolerance)
        finally:
            for handle in dataset._files.values():
                handle.close()


if __name__ == "__main__":
    unittest.main()
