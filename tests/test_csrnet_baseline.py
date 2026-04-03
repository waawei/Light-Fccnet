import importlib
import sys
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path


TORCH_AVAILABLE = find_spec("torch") is not None
YAML_AVAILABLE = find_spec("yaml") is not None


@contextmanager
def _pack_train_module():
    try:
        train_module = importlib.reload(importlib.import_module("pack.train"))
        yield train_module
    finally:
        sys.modules.pop("pack.train", None)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for CSRNet baseline tests")
class CSRNetRegistryTests(unittest.TestCase):
    def test_build_model_supports_csrnet(self):
        from pack.models import build_model

        cfg = {
            "model": {
                "name": "csrnet",
                "in_channels": 3,
                "frontend_channels": [64, 64, 128, 128, 256, 256, 256],
                "backend_channels": [256, 128, 64, 64],
            }
        }

        model = build_model(cfg)

        self.assertEqual(model.__class__.__name__, "CSRNet")
        self.assertTrue(callable(getattr(model, "predict_count", None)))


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for CSRNet forward tests")
class CSRNetForwardContractTests(unittest.TestCase):
    def test_forward_returns_training_compatible_triplet(self):
        import torch

        from pack.models import build_model

        cfg = {
            "model": {
                "name": "csrnet",
                "in_channels": 3,
                "input_size": [128, 128],
            }
        }

        model = build_model(cfg)
        x = torch.randn(2, 3, 128, 128)

        final_density, attention, raw_density = model(x)

        self.assertEqual(tuple(final_density.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(attention.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(raw_density.shape), (2, 1, 128, 128))
        self.assertTrue(torch.all(attention == 1))

    def test_predict_count_returns_batch_vector(self):
        import torch

        from pack.models import build_model

        model = build_model({"model": {"name": "csrnet", "input_size": [128, 128]}})
        x = torch.randn(2, 3, 128, 128)

        counts = model.predict_count(x)

        self.assertEqual(tuple(counts.shape), (2,))


@unittest.skipUnless(TORCH_AVAILABLE and YAML_AVAILABLE, "torch and yaml are required for CSRNet config tests")
class CSRNetConfigTests(unittest.TestCase):
    def test_dataset_specific_configs_build_csrnet(self):
        import yaml

        from pack.models import build_model

        config_paths = [
            Path("pack/config/gwhd/config_gwhd_csrnet.yaml"),
            Path("pack/config/mtc/config_mtc_csrnet.yaml"),
            Path("pack/config/urc/config_urc_csrnet.yaml"),
        ]

        for config_path in config_paths:
            with self.subTest(config_path=str(config_path)):
                self.assertTrue(config_path.exists(), f"missing config: {config_path}")
                with config_path.open("r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle)
                model = build_model(cfg)
                self.assertEqual(model.__class__.__name__, "CSRNet")


@unittest.skipUnless(TORCH_AVAILABLE and YAML_AVAILABLE, "torch and yaml are required for CSRNet train compatibility tests")
class CSRNetTrainCompatibilityTests(unittest.TestCase):
    def test_csrnet_runs_one_forward_loss_pass_under_training_contract(self):
        import torch

        with _pack_train_module() as train_module:
            cfg = {
                "model": {
                    "name": "csrnet",
                    "in_channels": 3,
                    "input_size": [64, 64],
                },
                "training": {
                    "loss_type": "light_fcc",
                    "use_p3_loss": False,
                },
            }

            model = train_module.build_model(cfg)
            criterion = train_module.build_criterion(cfg)

            image = torch.randn(2, 3, 64, 64)
            gt_density = torch.zeros(2, 1, 64, 64)
            gt_count = torch.zeros(2)

            final_density, attention, raw_density = model(image)
            loss, terms = criterion(final_density, gt_density, gt_count=gt_count)

            self.assertEqual(tuple(attention.shape), (2, 1, 64, 64))
            self.assertEqual(tuple(raw_density.shape), (2, 1, 64, 64))
            self.assertTrue(torch.is_tensor(loss))
            self.assertIn("count", terms)
            self.assertIn("l2", terms)


if __name__ == "__main__":
    unittest.main()
