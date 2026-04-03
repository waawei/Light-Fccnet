import unittest
import importlib
import sys
import tempfile
import warnings
from unittest import mock
from pathlib import Path
from contextlib import contextmanager

import torch
import yaml
from PIL import Image

from pack.data import GWHDDataset
from pack.data.point_supervision import apply_transform_with_points
from pack.models import build_model
from pack.models.light_fccnet import LightFCCNet
from pack.utils import LightFCCLoss
from pack.utils.metrics import cal_mape
from pack.utils.light_ldms import compute_ldms_scales, compute_match_thresholds
from pack.models.modules import (
    LightChannelAttention,
    LightDensityHead,
    LightPyramidFeatureAggregation,
    LightSpatialAttention,
    LightweightConvBlock,
)


@contextmanager
def _pack_train_module():
    try:
        train_module = importlib.reload(importlib.import_module("pack.train"))
        yield train_module
    finally:
        sys.modules.pop("pack.train", None)


class LightFCCNetBuildTests(unittest.TestCase):
    def test_build_model_supports_light_fccnet(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "in_channels": 3,
                "input_size": [128, 128],
                "stage_channels": [32, 64, 96, 128],
            }
        }

        model = build_model(cfg)

        self.assertEqual(model.__class__.__name__, "LightFCCNet")

    def test_direct_light_fccnet_defaults_match_build_model_baseline_contract(self):
        model = LightFCCNet()

        self.assertFalse(model.use_p1)
        self.assertFalse(model.use_p2)

    def test_direct_light_fccnet_warns_when_p2_is_requested_without_p1(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LightFCCNet(use_p1=False, use_p2=True)

        self.assertFalse(model.use_p2)
        self.assertTrue(any("use_p2" in str(item.message) and "use_p1" in str(item.message) for item in caught))

    def test_light_fccnet_forward_returns_three_maps(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "in_channels": 3,
                "input_size": [128, 128],
                "stage_channels": [32, 64, 96, 128],
            }
        }
        model = build_model(cfg)
        x = torch.randn(2, 3, 128, 128)

        final_density, attention, density = model(x)

        self.assertEqual(tuple(final_density.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(attention.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(density.shape), (2, 1, 128, 128))

    def test_build_model_passes_head_init_bias_to_density_head(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "in_channels": 3,
                "input_size": [128, 128],
                "stage_channels": [32, 64, 96, 128],
                "head_init_bias": -5.5,
            }
        }

        model = build_model(cfg)

        self.assertAlmostEqual(float(model.density_head.out_conv.bias.item()), -5.5, places=5)

    def test_build_model_migrates_legacy_model_use_p3_to_training_use_p3_loss(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "use_p3": True,
            },
            "training": {
                "loss_type": "light_fcc",
            },
        }

        build_model(cfg)

        self.assertTrue(cfg["training"]["use_p3_loss"])
        self.assertNotIn("use_p3_loss", cfg["model"])

    def test_build_model_preserves_explicit_training_use_p3_loss(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "use_p3": False,
            },
            "training": {
                "loss_type": "light_fcc",
                "use_p3_loss": False,
            },
        }

        build_model(cfg)

        self.assertFalse(cfg["training"]["use_p3_loss"])

    def test_build_model_treats_p2_without_p1_as_disabled(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "use_p1": False,
                "use_p2": True,
                "use_p3": False,
            }
        }

        model = build_model(cfg)

        self.assertFalse(model.use_p1)
        self.assertFalse(model.use_p2)

    def test_build_model_warns_when_p2_is_requested_without_p1(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "use_p1": False,
                "use_p2": True,
            }
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = build_model(cfg)

        self.assertFalse(model.use_p2)
        self.assertTrue(any("use_p2" in str(item.message) and "use_p1" in str(item.message) for item in caught))

    def test_build_model_ignores_ablation_mode_and_uses_explicit_flags(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "ablation_mode": "p1_p2_p3",
                "use_p1": False,
                "use_p2": False,
            }
        }

        model = build_model(cfg)

        self.assertFalse(model.use_p1)
        self.assertFalse(model.use_p2)

    def test_light_fccnet_baseline_forward_uses_single_scale_backbone_path(self):
        cfg = {"model": {"name": "light_fccnet", "use_p1": False, "use_p2": False}}
        model = build_model(cfg)
        x = torch.randn(2, 3, 64, 64)

        model.backbone.forward = mock.Mock(side_effect=AssertionError("full pyramid path should not run for baseline"))
        tracked_single_scale = model.backbone.forward_single_scale
        model.backbone.forward_single_scale = mock.Mock(wraps=tracked_single_scale)

        final_density, _, _ = model(x)

        self.assertEqual(tuple(final_density.shape), (2, 1, 64, 64))
        model.backbone.forward_single_scale.assert_called_once()

    def test_light_fccnet_p1_only_forward_returns_shape_compatible_attention_tensor(self):
        cfg = {"model": {"name": "light_fccnet", "use_p1": True, "use_p2": False}}
        model = build_model(cfg)
        x = torch.randn(2, 3, 64, 64)

        final_density, attention, raw_density = model(x)

        self.assertEqual(tuple(final_density.shape), (2, 1, 64, 64))
        self.assertIsNotNone(attention)
        self.assertEqual(tuple(attention.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(raw_density.shape), (2, 1, 64, 64))

    def test_build_model_rejects_removed_legacy_models(self):
        for legacy_name in ("mpad_net", "fccnet_paper_1to1", "fccnet_paper_strict", "mpcount_adapter"):
            with self.subTest(model_name=legacy_name):
                with self.assertRaises(ValueError):
                    build_model({"model": {"name": legacy_name}})


class LightFCCLossTests(unittest.TestCase):
    def test_baseline_counting_loss_returns_density_and_count_terms_without_ssim(self):
        losses_module = importlib.import_module("pack.utils.losses")
        criterion = losses_module.BaselineCountingLoss()
        pred = torch.ones(2, 1, 8, 8)
        gt = torch.zeros(2, 1, 8, 8)

        total, terms = criterion(pred, gt)

        self.assertTrue(torch.is_tensor(total))
        self.assertIn("l2", terms)
        self.assertIn("count", terms)
        self.assertIn("loss", terms)
        self.assertNotIn("ssim", terms)
        self.assertEqual(total.ndim, 0)

    def test_baseline_counting_loss_scales_explicit_gt_count(self):
        losses_module = importlib.import_module("pack.utils.losses")
        criterion = losses_module.BaselineCountingLoss(density_scale=2.0)
        pred = torch.zeros(1, 1, 4, 4)
        gt = torch.zeros(1, 1, 4, 4)

        _, terms = criterion(pred, gt, gt_count=torch.tensor([3.0]))

        self.assertAlmostEqual(float(terms["count"].item()), 36.0, places=5)

    def test_light_fcc_loss_is_near_zero_when_prediction_matches_target(self):
        criterion = LightFCCLoss(alpha=0.1)
        gt = torch.rand(2, 1, 8, 8)

        total, terms = criterion(gt, gt)

        self.assertLess(float(total.item()), 1e-4)
        self.assertLess(float(terms["l2"].item()), 1e-6)
        self.assertLess(float(terms["count"].item()), 1e-6)

    def test_light_fcc_loss_returns_breakdown(self):
        criterion = LightFCCLoss(alpha=0.1)
        pred = torch.ones(2, 1, 8, 8)
        gt = torch.zeros(2, 1, 8, 8)

        total, terms = criterion(pred, gt)

        self.assertTrue(torch.is_tensor(total))
        self.assertIn("l2", terms)
        self.assertIn("count", terms)
        self.assertIn("ssim", terms)
        self.assertIn("loss", terms)
        self.assertEqual(total.ndim, 0)

    def test_light_fcc_loss_uses_explicit_gt_count_when_provided(self):
        criterion = LightFCCLoss(alpha=0.0, density_scale=2.0)
        pred = torch.ones(1, 1, 4, 4)
        gt = torch.ones(1, 1, 4, 4)

        _, terms = criterion(pred, gt, gt_count=torch.tensor([10.0]))

        self.assertAlmostEqual(float(terms["count"].item()), 16.0, places=5)

    def test_light_fcc_loss_supports_single_ssim_constant(self):
        criterion = LightFCCLoss(alpha=0.1, ssim_c=0.25)
        self.assertAlmostEqual(criterion.ssim_c, 0.25)

    def test_light_fcc_loss_does_not_mix_ldms_into_loss_terms(self):
        criterion = LightFCCLoss(alpha=0.1)
        pred = torch.zeros(1, 1, 8, 8)
        gt = torch.zeros(1, 1, 8, 8)
        points = [torch.tensor([[1.0, 1.0], [6.0, 6.0]])]
        with self.assertRaises(TypeError):
            criterion(pred, gt, points=points, image_shape=(8, 8))

    def test_train_build_criterion_supports_light_fcc_loss(self):
        with _pack_train_module() as train_module:
            criterion = train_module.build_criterion(
                {"training": {"loss_type": "light_fcc", "alpha": 0.2, "use_p3_loss": True}}
            )

        self.assertIsInstance(criterion, LightFCCLoss)

    def test_train_build_criterion_accepts_single_ssim_constant(self):
        with _pack_train_module() as train_module:
            criterion = train_module.build_criterion(
                {"training": {"loss_type": "light_fcc", "ssim_c": 0.2, "use_p3_loss": True}}
            )

        self.assertIsInstance(criterion, LightFCCLoss)
        self.assertAlmostEqual(criterion.ssim_c, 0.2)

    def test_build_criterion_returns_baseline_counting_loss_when_use_p3_loss_is_false(self):
        with _pack_train_module() as train_module:
            criterion = train_module.build_criterion({"training": {"loss_type": "light_fcc", "use_p3_loss": False}})

        self.assertEqual(criterion.__class__.__name__, "BaselineCountingLoss")

    def test_build_criterion_returns_light_fcc_loss_when_use_p3_loss_is_true(self):
        with _pack_train_module() as train_module:
            criterion = train_module.build_criterion({"training": {"loss_type": "light_fcc", "use_p3_loss": True}})

        self.assertIsInstance(criterion, LightFCCLoss)

    def test_train_build_criterion_rejects_removed_legacy_loss_types(self):
        with _pack_train_module() as train_module:
            for loss_type in ("mse", "mse_bce", "multitask", "density_mse_bce"):
                with self.subTest(loss_type=loss_type):
                    with self.assertRaises(ValueError):
                        train_module.build_criterion({"training": {"loss_type": loss_type}})

    def test_train_build_criterion_rejects_removed_light_fcc_aliases(self):
        with _pack_train_module() as train_module:
            for loss_type in ("lfcc", "light_fcc_loss"):
                with self.subTest(loss_type=loss_type):
                    with self.assertRaises(ValueError):
                        train_module.build_criterion({"training": {"loss_type": loss_type}})

    def test_build_criterion_distinguishes_use_p3_loss_true_from_false(self):
        with _pack_train_module() as train_module:
            paper = train_module.build_criterion({"training": {"loss_type": "light_fcc", "use_p3_loss": True}})
            baseline = train_module.build_criterion({"training": {"loss_type": "light_fcc", "use_p3_loss": False}})

            pred = torch.ones(1, 1, 4, 4)
            gt = torch.zeros(1, 1, 4, 4)
            paper_total, paper_terms = paper(pred, gt)
            baseline_total, baseline_terms = baseline(pred, gt)

        self.assertIsInstance(paper, LightFCCLoss)
        self.assertEqual(baseline.__class__.__name__, "BaselineCountingLoss")
        self.assertIn("ssim", paper_terms)
        self.assertNotIn("ssim", baseline_terms)
        self.assertNotAlmostEqual(float(paper_total.item()), float(baseline_total.item()), places=6)

    def test_train_one_epoch_with_light_fcc_loss_smoke(self):
        with _pack_train_module() as train_module:
            cfg = {
                "model": {
                    "name": "light_fccnet",
                    "in_channels": 3,
                    "input_size": [64, 64],
                    "stage_channels": [16, 24, 32, 48],
                    "fusion_channels": 32,
                    "use_p1": True,
                    "use_p2": True,
                },
                "training": {
                    "loss_type": "light_fcc",
                    "alpha": 0.1,
                    "use_p3_loss": True,
                },
            }
            model = build_model(cfg)
            criterion = train_module.build_criterion(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            batch = {
                "image": torch.randn(2, 3, 64, 64),
                "density": torch.rand(2, 1, 64, 64),
                "count": torch.rand(2),
            }

            log = train_module.train_one_epoch(model, [batch], criterion, optimizer, torch.device("cpu"), scaler=None)

        self.assertIn("loss", log)
        self.assertTrue(log["loss"] >= 0.0)

    def test_train_one_epoch_updates_attention_head_when_attention_supervision_enabled(self):
        with _pack_train_module() as train_module:
            torch.manual_seed(0)
            cfg = {
                "model": {
                    "name": "light_fccnet",
                    "in_channels": 3,
                    "input_size": [64, 64],
                    "stage_channels": [16, 24, 32, 48],
                    "fusion_channels": 32,
                    "use_p1": True,
                    "use_p2": True,
                },
                "training": {
                    "loss_type": "light_fcc",
                    "alpha": 0.1,
                    "use_p3_loss": True,
                    "attention_loss_weight": 1.0,
                },
            }
            model = build_model(cfg)
            criterion = train_module.build_criterion(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            before = model.attention_fusion.attention_head[3].weight.detach().clone()
            batch = {
                "image": torch.randn(2, 3, 64, 64),
                "density": torch.rand(2, 1, 64, 64),
                "attention_mask": torch.rand(2, 1, 64, 64),
                "count": torch.rand(2),
            }

            train_module.train_one_epoch(model, [batch], criterion, optimizer, torch.device("cpu"), scaler=None)

            after = model.attention_fusion.attention_head[3].weight.detach()
        self.assertFalse(torch.allclose(before, after))

    def test_train_one_epoch_skips_attention_bce_when_p2_is_disabled(self):
        with _pack_train_module() as train_module:
            torch.manual_seed(0)
            base_cfg = {
                "model": {
                    "name": "light_fccnet",
                    "in_channels": 3,
                    "input_size": [64, 64],
                    "stage_channels": [16, 24, 32, 48],
                    "fusion_channels": 32,
                    "use_p1": True,
                    "use_p2": False,
                },
                "training": {
                    "loss_type": "light_fcc",
                    "alpha": 0.1,
                    "use_p3_loss": False,
                    "attention_loss_weight": 0.0,
                },
            }
            batch = {
                "image": torch.randn(2, 3, 64, 64),
                "density": torch.rand(2, 1, 64, 64),
                "attention_mask": torch.rand(2, 1, 64, 64),
                "count": torch.rand(2),
            }

            torch.manual_seed(0)
            model_no_att = build_model(base_cfg)
            criterion_no_att = train_module.build_criterion(base_cfg)
            optimizer_no_att = torch.optim.Adam(model_no_att.parameters(), lr=0.0)
            log_no_att = train_module.train_one_epoch(
                model_no_att, [batch], criterion_no_att, optimizer_no_att, torch.device("cpu"), scaler=None
            )

            weighted_cfg = {
                "model": dict(base_cfg["model"]),
                "training": dict(base_cfg["training"], attention_loss_weight=1.0),
            }
            torch.manual_seed(0)
            model_weighted = build_model(weighted_cfg)
            criterion_weighted = train_module.build_criterion(weighted_cfg)
            optimizer_weighted = torch.optim.Adam(model_weighted.parameters(), lr=0.0)
            log_weighted = train_module.train_one_epoch(
                model_weighted, [batch], criterion_weighted, optimizer_weighted, torch.device("cpu"), scaler=None
            )

        self.assertAlmostEqual(log_no_att["loss"], log_weighted["loss"], places=6)

    def test_validate_reports_mape(self):
        with _pack_train_module() as train_module:
            class DummyModel(torch.nn.Module):
                def forward(self, image):
                    batch = image.shape[0]
                    density = torch.zeros(batch, 1, 4, 4)
                    density[:, :, 0, 0] = 2.0
                    return density, density, density

            loader = [
                {
                    "image": torch.randn(1, 3, 4, 4),
                    "count": torch.tensor([1.0]),
                }
            ]

            metrics = train_module.validate(DummyModel(), loader, torch.device("cpu"), density_scale=1.0)

        self.assertIn("MAE", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertAlmostEqual(metrics["MAE"], 1.0, places=5)
        self.assertAlmostEqual(metrics["MSE"], 1.0, places=5)
        self.assertAlmostEqual(metrics["MAPE"], 100.0, places=3)


class LightLDMSTests(unittest.TestCase):
    def test_compute_ldms_scales_returns_one_scale_per_point(self):
        points = torch.tensor([[10.0, 10.0], [20.0, 20.0], [40.0, 40.0]])
        scales = compute_ldms_scales(points, image_shape=(100, 100), k=1, factor=1.0, max_ratio=0.05)

        self.assertEqual(tuple(scales.shape), (3,))
        self.assertTrue(torch.all(scales > 0))
        self.assertTrue(torch.all(scales <= 5.0))

    def test_compute_match_thresholds_uses_fixed_box_shape(self):
        dx, dy = compute_match_thresholds(box_width=4.0, box_height=8.0)

        self.assertAlmostEqual(dx, 2.0)
        self.assertAlmostEqual(dy, (4.0**2 + 8.0**2) ** 0.5 / 2.0)


class LightMetricsTests(unittest.TestCase):
    def test_cal_mape_ignores_zero_count_targets(self):
        pred = torch.tensor([2.0, 5.0], dtype=torch.float32)
        gt = torch.tensor([0.0, 4.0], dtype=torch.float32)

        mape = cal_mape(pred, gt)

        self.assertAlmostEqual(mape, 25.0, places=5)

    def test_cal_mape_returns_zero_when_all_targets_are_zero(self):
        pred = torch.tensor([0.0, 3.0], dtype=torch.float32)
        gt = torch.tensor([0.0, 0.0], dtype=torch.float32)

        mape = cal_mape(pred, gt)

        self.assertEqual(mape, 0.0)


class LightAttentionTests(unittest.TestCase):
    def test_light_spatial_attention_accepts_token_budget(self):
        module = LightSpatialAttention(channels=32, max_tokens=64)
        x = torch.randn(2, 32, 32, 32)

        y = module(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(module.max_tokens, 64)

    def test_light_spatial_attention_keeps_full_channel_projections(self):
        module = LightSpatialAttention(channels=32, max_tokens=64)

        self.assertEqual(module.query.out_channels, 32)
        self.assertEqual(module.key.out_channels, 32)
        self.assertEqual(module.value.out_channels, 32)

    def test_light_channel_attention_has_fusion_branch(self):
        module = LightChannelAttention(channels=32)
        x = torch.randn(2, 32, 8, 8)

        y = module(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(hasattr(module, "fusion"))


class LightBackboneTests(unittest.TestCase):
    def test_lightweight_conv_block_has_no_input_residual(self):
        block = LightweightConvBlock(channels=4)
        for param in block.parameters():
            param.data.zero_()
        block.eval()
        x = torch.ones(1, 4, 8, 8)

        with torch.no_grad():
            y = block(x)

        self.assertTrue(torch.allclose(y, torch.zeros_like(y), atol=1e-6))

    def test_light_pyramid_feature_aggregation_matches_simulated_paper_scales(self):
        backbone = LightPyramidFeatureAggregation(in_channels=3, stage_channels=(8, 12, 16, 20))
        backbone.eval()
        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            feats = backbone(x)

        self.assertEqual(len(feats), 4)
        self.assertEqual(tuple(feats[0].shape[-2:]), (64, 64))
        self.assertEqual(tuple(feats[1].shape[-2:]), (16, 16))
        self.assertEqual(tuple(feats[2].shape[-2:]), (4, 4))
        self.assertEqual(tuple(feats[3].shape[-2:]), (1, 1))

    def test_light_density_head_starts_with_small_non_negative_outputs(self):
        head = LightDensityHead(in_channels=16, hidden_channels=16)
        x = torch.randn(2, 16, 32, 32)

        with torch.no_grad():
            y = head(x)

        self.assertTrue(torch.isfinite(y).all().item())
        self.assertGreaterEqual(float(y.min().item()), 0.0)
        self.assertLess(float(y.max().item()), 1e-3)

    def test_light_fccnet_initial_predicted_count_is_bounded(self):
        cfg = {
            "model": {
                "name": "light_fccnet",
                "in_channels": 3,
                "input_size": [64, 64],
                "stage_channels": [16, 24, 32, 48],
                "fusion_channels": 32,
                "use_p1": True,
                "use_p2": True,
            }
        }
        model = build_model(cfg)
        x = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            final_density, _, _ = model(x)
            count = final_density.sum(dim=(1, 2, 3))

        self.assertTrue(torch.isfinite(count).all().item())
        self.assertLess(float(count.max().item()), 8.0)

    def test_light_fccnet_first_backward_reaches_fusion_layers(self):
        torch.manual_seed(0)
        cfg = {
            "model": {
                "name": "light_fccnet",
                "in_channels": 3,
                "input_size": [64, 64],
                "stage_channels": [16, 24, 32, 48],
                "fusion_channels": 32,
                "use_p1": True,
                "use_p2": True,
            }
        }
        model = build_model(cfg)
        criterion = LightFCCLoss(alpha=0.1)
        image = torch.randn(2, 3, 64, 64)
        gt_density = torch.rand(2, 1, 64, 64)
        gt_count = torch.rand(2)

        final_density, _, _ = model(image)
        loss, _ = criterion(final_density, gt_density, gt_count=gt_count)
        loss.backward()

        self.assertGreater(float(model.density_head.conv1.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(model.attention_fusion.fuse[0].weight.grad.abs().sum().item()), 0.0)


class LightDataPipelineTests(unittest.TestCase):
    def test_apply_transform_with_points_clamps_boundary_keypoints(self):
        import albumentations as A
        import numpy as np

        image = np.zeros((384, 384, 3), dtype=np.uint8)
        points = np.array([[276.0, 384.0]], dtype=np.float32)
        transform = A.Compose([], keypoint_params=A.KeypointParams(format="xy", remove_invisible=True))

        _, aug_points = apply_transform_with_points(image, points, transform)

        self.assertEqual(tuple(aug_points.shape), (1, 2))
        self.assertGreaterEqual(float(aug_points[0, 0]), 0.0)
        self.assertGreaterEqual(float(aug_points[0, 1]), 0.0)
        self.assertLess(float(aug_points[0, 0]), 384.0)
        self.assertLess(float(aug_points[0, 1]), 384.0)

    def test_gwhd_dataset_returns_transformed_points(self):
        import albumentations as A
        import numpy as np
        import pandas as pd
        from albumentations.pytorch import ToTensorV2

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            image_path = images_dir / "sample.jpg"
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(image_path)

            csv_path = root / "labels.csv"
            pd.DataFrame(
                [
                    {
                        "image_name": "sample.jpg",
                        "BoxesString": "1 2 3 4",
                        "domain": "test",
                    }
                ]
            ).to_csv(csv_path, index=False)

            transform = A.Compose(
                [
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
            )

            dataset = GWHDDataset(
                str(csv_path),
                str(images_dir),
                transform=transform,
                target_size=(10, 10),
                sigma=1,
                attention_radius=0,
            )

            sample = dataset[0]

            self.assertIn("points", sample)
            self.assertEqual(tuple(sample["points"].shape), (1, 2))
            self.assertAlmostEqual(float(sample["count"].item()), 1.0)
            self.assertAlmostEqual(float(sample["points"][0, 0].item()), 7.0, places=4)
            self.assertAlmostEqual(float(sample["points"][0, 1].item()), 3.0, places=4)

    def test_light_fcc_collate_fn_preserves_variable_length_points(self):
        with _pack_train_module() as train_module:
            batch = train_module.counting_collate_fn(
                [
                    {
                        "image": torch.randn(3, 8, 8),
                        "density": torch.randn(1, 8, 8),
                        "attention_mask": torch.randn(1, 8, 8),
                        "count": torch.tensor(1.0),
                        "points": torch.tensor([[1.0, 2.0]]),
                        "image_name": "a.jpg",
                    },
                    {
                        "image": torch.randn(3, 8, 8),
                        "density": torch.randn(1, 8, 8),
                        "attention_mask": torch.randn(1, 8, 8),
                        "count": torch.tensor(2.0),
                        "points": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                        "image_name": "b.jpg",
                    },
                ]
            )

        self.assertEqual(tuple(batch["image"].shape), (2, 3, 8, 8))
        self.assertEqual(len(batch["points"]), 2)
        self.assertEqual(tuple(batch["points"][0].shape), (1, 2))
        self.assertEqual(tuple(batch["points"][1].shape), (2, 2))
        self.assertEqual(batch["image_name"], ["a.jpg", "b.jpg"])


class LightResultToolTests(unittest.TestCase):
    @staticmethod
    def _canonical_config_expectations():
        return {
            "gwhd": {
                "config_gwhd_light_baseline.yaml": (False, False, False),
                "config_gwhd_light_baseline_p1.yaml": (True, False, False),
                "config_gwhd_light_baseline_p1_p2.yaml": (True, True, False),
                "config_gwhd_light_full.yaml": (True, True, True),
            },
            "mtc": {
                "config_mtc_light_baseline.yaml": (False, False, False),
                "config_mtc_light_baseline_p1.yaml": (True, False, False),
                "config_mtc_light_baseline_p1_p2.yaml": (True, True, False),
                "config_mtc_light_full.yaml": (True, True, True),
            },
            "urc": {
                "config_urc_light_baseline.yaml": (False, False, False),
                "config_urc_light_baseline_p1.yaml": (True, False, False),
                "config_urc_light_baseline_p1_p2.yaml": (True, True, False),
                "config_urc_light_full.yaml": (True, True, True),
            },
        }

    def test_extract_best_results_includes_best_metrics(self):
        import pack.tools.extract_best_results as extract_best_results

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "gwhd_light_full"
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = run_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": 12,
                    "best_mae": 10.5,
                    "best_metrics": {"MAE": 10.5, "MSE": 121.0, "MAPE": 8.8},
                    "config": {
                        "model": {"name": "light_fccnet", "input_size": [256, 256]},
                        "data": {"gwhd_train_csv": "/root/autodl-tmp/datasets/gwhd_2021/competition_train.csv"},
                    },
                },
                ckpt_path,
            )

            row = extract_best_results.extract_one(str(run_dir))

            self.assertIsNotNone(row)
            self.assertEqual(row["model_name"], "light_fccnet")
            self.assertEqual(row["best_mse"], 121.0)
            self.assertEqual(row["best_mape"], 8.8)

    def test_extract_best_results_reads_use_p3_loss_from_training_semantics(self):
        import pack.tools.extract_best_results as extract_best_results

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "gwhd_light_full"
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = run_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": 5,
                    "best_mae": 3.0,
                    "best_metrics": {"MAE": 3.0, "MSE": 9.0, "MAPE": 1.2},
                    "config": {
                        "model": {
                            "name": "light_fccnet",
                            "input_size": [256, 256],
                            "use_p1": True,
                            "use_p2": True,
                            "use_p3": False,
                        },
                        "training": {"use_p3_loss": True},
                        "data": {"gwhd_train_csv": "/root/autodl-tmp/datasets/gwhd_2021/competition_train.csv"},
                    },
                },
                ckpt_path,
            )

            row = extract_best_results.extract_one(str(run_dir))

            self.assertIsNotNone(row)
            self.assertEqual((row["use_p1"], row["use_p2"], row["use_p3"]), (1, 1, 1))

    def test_extract_best_results_falls_back_to_legacy_model_use_p3(self):
        import pack.tools.extract_best_results as extract_best_results

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "legacy_light_full"
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = run_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": 5,
                    "best_mae": 4.0,
                    "best_metrics": {"MAE": 4.0, "MSE": 16.0, "MAPE": 2.0},
                    "config": {
                        "model": {
                            "name": "light_fccnet",
                            "input_size": [256, 256],
                            "use_p1": True,
                            "use_p2": True,
                            "use_p3": True,
                        },
                        "data": {"gwhd_train_csv": "/root/autodl-tmp/datasets/gwhd_2021/competition_train.csv"},
                    },
                },
                ckpt_path,
            )

            row = extract_best_results.extract_one(str(run_dir))

            self.assertIsNotNone(row)
            self.assertEqual((row["use_p1"], row["use_p2"], row["use_p3"]), (1, 1, 1))

    def test_canonical_light_training_configs_parse(self):
        config_paths = [
            Path("pack/config") / dataset_name / filename
            for dataset_name, files in self._canonical_config_expectations().items()
            for filename in files
        ]
        self.assertTrue(config_paths)

        for path in config_paths:
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["model"]["name"], "light_fccnet")
            self.assertEqual(cfg["training"]["loss_type"], "light_fcc")
            self.assertIn("MAPE", cfg["eval"]["metrics"])
            dataset_name = path.parent.name
            if dataset_name == "gwhd":
                self.assertNotIn("head_init_bias", cfg["model"])
            elif dataset_name == "mtc":
                self.assertAlmostEqual(float(cfg["model"]["head_init_bias"]), -7.2, places=5)
                self.assertAlmostEqual(float(cfg["training"]["learning_rate"]), 5.0e-5, places=10)
            elif dataset_name == "urc":
                self.assertAlmostEqual(float(cfg["model"]["head_init_bias"]), -5.5, places=5)
                self.assertAlmostEqual(float(cfg["training"]["learning_rate"]), 5.0e-5, places=10)
            else:
                self.fail(f"Unexpected config dataset directory: {dataset_name}")

    def test_canonical_configs_follow_paper_ablation_ladder(self):
        for dataset_name, files in self._canonical_config_expectations().items():
            config_dir = Path("pack/config") / dataset_name
            for filename, expected in files.items():
                with self.subTest(config=filename):
                    with (config_dir / filename).open("r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)

                    self.assertEqual(cfg["model"].get("use_p1"), expected[0])
                    self.assertEqual(cfg["model"].get("use_p2"), expected[1])
                    self.assertEqual(cfg["training"].get("use_p3_loss"), expected[2])
                    self.assertNotIn("use_p3", cfg["model"])

                    model = build_model(cfg)
                    observed = (
                        model.use_p1,
                        model.use_p2,
                        cfg.get("training", {}).get("use_p3_loss"),
                    )
                    self.assertEqual(observed, expected)


class LightTrainingEntrypointTests(unittest.TestCase):
    def test_train_default_config_points_to_light_full(self):
        with _pack_train_module() as train_module:
            original_argv = sys.argv
            try:
                sys.argv = ["train.py"]
                args = train_module.parse_args()
            finally:
                sys.argv = original_argv

        self.assertEqual(args.config, "config/gwhd/config_gwhd_light_full.yaml")


if __name__ == "__main__":
    unittest.main()
