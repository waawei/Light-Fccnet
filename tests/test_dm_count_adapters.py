import unittest
from unittest import mock
import tempfile
from pathlib import Path

import torch


class _FakeCountingDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 32, 40, dtype=torch.float32),
            "count": torch.tensor(2.0),
            "points": torch.tensor([[4.0, 8.0], [20.0, 12.0]], dtype=torch.float32),
            "image_name": "sample_001",
        }


class _FakeEmptyPointDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 24, 24, dtype=torch.float32),
            "count": torch.tensor(0.0),
            "points": torch.zeros((0, 2), dtype=torch.float32),
            "image_name": "empty_case",
        }


class _FakeValDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.zeros(3, 32, 40, dtype=torch.float32)
        count = float(index + 1)
        name = f"sample_{index}"
        return image, count, name


class DMCountDiscreteMapTests(unittest.TestCase):
    def test_downsampled_discrete_map_preserves_point_count(self):
        from external.baselines.dm_count.local_adapters.discrete_map import generate_downsampled_discrete_map

        points = torch.tensor([[4.0, 8.0], [20.0, 12.0]], dtype=torch.float32)

        discrete = generate_downsampled_discrete_map(points, image_shape=(32, 40), downsample_ratio=8)

        self.assertEqual(tuple(discrete.shape), (1, 4, 5))
        self.assertAlmostEqual(float(discrete.sum().item()), 2.0, places=5)

    def test_downsampled_discrete_map_handles_empty_points(self):
        from external.baselines.dm_count.local_adapters.discrete_map import generate_downsampled_discrete_map

        discrete = generate_downsampled_discrete_map(torch.zeros((0, 2)), image_shape=(24, 24), downsample_ratio=8)

        self.assertEqual(tuple(discrete.shape), (1, 3, 3))
        self.assertAlmostEqual(float(discrete.sum().item()), 0.0, places=5)


class DMCountDatasetAdapterTests(unittest.TestCase):
    def test_train_adapter_returns_upstream_training_tuple(self):
        from external.baselines.dm_count.local_adapters.datasets import DMCountDatasetAdapter

        adapter = DMCountDatasetAdapter(_FakeCountingDataset(), split="train", downsample_ratio=8)

        image, points, gt_discrete = adapter[0]

        self.assertEqual(tuple(image.shape), (3, 32, 40))
        self.assertEqual(tuple(points.shape), (2, 2))
        self.assertEqual(tuple(gt_discrete.shape), (1, 4, 5))
        self.assertAlmostEqual(float(gt_discrete.sum().item()), 2.0, places=5)

    def test_val_adapter_returns_upstream_validation_tuple(self):
        from external.baselines.dm_count.local_adapters.datasets import DMCountDatasetAdapter

        adapter = DMCountDatasetAdapter(_FakeCountingDataset(), split="val", downsample_ratio=8)

        image, count, name = adapter[0]

        self.assertEqual(tuple(image.shape), (3, 32, 40))
        self.assertAlmostEqual(float(count), 2.0, places=5)
        self.assertEqual(name, "sample_001")

    def test_train_adapter_supports_empty_points(self):
        from external.baselines.dm_count.local_adapters.datasets import DMCountDatasetAdapter

        adapter = DMCountDatasetAdapter(_FakeEmptyPointDataset(), split="train", downsample_ratio=8)

        image, points, gt_discrete = adapter[0]

        self.assertEqual(tuple(image.shape), (3, 24, 24))
        self.assertEqual(tuple(points.shape), (0, 2))
        self.assertEqual(tuple(gt_discrete.shape), (1, 3, 3))
        self.assertAlmostEqual(float(gt_discrete.sum().item()), 0.0, places=5)


class DMCountDatasetBuilderTests(unittest.TestCase):
    def test_build_dmcount_datasets_uses_gwhd_source_and_wraps_splits(self):
        import external.baselines.dm_count.local_adapters.datasets as dm_datasets

        cfg = {
            "model": {"input_size": [256, 256]},
            "data": {
                "sigma": 6,
                "gwhd_train_csv": "train.csv",
                "gwhd_val_csv": "val.csv",
                "gwhd_images_dir": "images",
            },
        }

        with mock.patch.object(dm_datasets, "GWHDDataset", side_effect=[_FakeCountingDataset(), _FakeCountingDataset()]) as dataset_cls:
            with mock.patch.object(dm_datasets, "get_train_transforms", return_value="train_tf"):
                with mock.patch.object(dm_datasets, "get_val_transforms", return_value="val_tf"):
                    datasets = dm_datasets.build_dmcount_datasets(cfg, dataset_name="gwhd")

        self.assertIn("train", datasets)
        self.assertIn("val", datasets)
        self.assertEqual(dataset_cls.call_count, 2)
        train_sample = datasets["train"][0]
        val_sample = datasets["val"][0]
        self.assertEqual(len(train_sample), 3)
        self.assertEqual(len(val_sample), 3)

    def test_build_dmcount_datasets_uses_mtc_split_files(self):
        import external.baselines.dm_count.local_adapters.datasets as dm_datasets

        cfg = {
            "model": {"input_size": [256, 256]},
            "data": {
                "sigma": 8,
                "mtc_root": "mtc_root",
                "mtc_train_split_file": "train.txt",
                "mtc_val_split_file": "val.txt",
            },
        }

        with mock.patch.object(dm_datasets, "MTCDataset", side_effect=[_FakeCountingDataset(), _FakeCountingDataset()]) as dataset_cls:
            with mock.patch.object(dm_datasets, "get_train_transforms", return_value="train_tf"):
                with mock.patch.object(dm_datasets, "get_val_transforms", return_value="val_tf"):
                    dm_datasets.build_dmcount_datasets(cfg, dataset_name="mtc")

        self.assertEqual(dataset_cls.call_count, 2)
        self.assertEqual(dataset_cls.call_args_list[0].kwargs["split"], "train")
        self.assertEqual(dataset_cls.call_args_list[1].kwargs["split"], "val")

    def test_build_dmcount_datasets_rejects_unknown_dataset_name(self):
        import external.baselines.dm_count.local_adapters.datasets as dm_datasets

        with self.assertRaises(ValueError):
            dm_datasets.build_dmcount_datasets({"model": {"input_size": [256, 256]}, "data": {}}, dataset_name="unknown")


class DMCountEvalExportTests(unittest.TestCase):
    def test_compute_count_metrics_returns_unified_project_fields(self):
        from external.baselines.dm_count.local_adapters.eval import compute_count_metrics

        metrics = compute_count_metrics(pred_counts=[10, 12, 9], gt_counts=[8, 10, 10])

        self.assertEqual(set(metrics.keys()), {"mae", "mse", "mape"})
        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["mse"], 0.0)
        self.assertGreaterEqual(metrics["mape"], 0.0)

    def test_build_result_row_keeps_paper_table_fields(self):
        from external.baselines.dm_count.local_adapters.eval import build_result_row

        row = build_result_row(
            dataset="GWHD",
            metrics={"mae": 1.0, "mse": 2.0, "mape": 3.0},
            result_type="adapted reproduction",
            params=100,
            flops=200,
            checkpoint_path="ckpt.pth",
        )

        self.assertEqual(row["method"], "DM-Count")
        self.assertEqual(row["dataset"], "GWHD")
        self.assertEqual(row["result_type"], "adapted reproduction")
        self.assertEqual(row["params"], 100)
        self.assertEqual(row["flops"], 200)
        self.assertEqual(row["checkpoint_path"], "ckpt.pth")

    def test_export_helpers_write_json_and_csv_rows(self):
        from external.baselines.dm_count.local_adapters.export_results import append_result_row_csv, save_result_row_json

        row = {
            "method": "DM-Count",
            "dataset": "GWHD",
            "mae": 1.0,
            "mse": 2.0,
            "mape": 3.0,
            "params": 100,
            "flops": 200,
            "result_type": "adapted reproduction",
            "checkpoint_path": "ckpt.pth",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "result.json"
            csv_path = Path(tmp_dir) / "result.csv"

            save_result_row_json(row, json_path)
            append_result_row_csv(row, csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("DM-Count", json_path.read_text(encoding="utf-8"))
            self.assertIn("dataset", csv_path.read_text(encoding="utf-8"))


class DMCountComplexityWrapperTests(unittest.TestCase):
    def test_build_dmcount_model_for_complexity_skips_weight_download(self):
        from external.baselines.dm_count.local_adapters.measure_complexity import build_dmcount_model_for_complexity

        model = build_dmcount_model_for_complexity()

        self.assertTrue(hasattr(model, "forward"))

    def test_measure_dmcount_complexity_returns_positive_counts(self):
        from external.baselines.dm_count.local_adapters.measure_complexity import measure_dmcount_complexity

        report = measure_dmcount_complexity(input_shape=(1, 3, 64, 64))

        self.assertEqual(report["model_name"], "dm_count")
        self.assertEqual(tuple(report["input_shape"]), (1, 3, 64, 64))
        self.assertGreater(report["params"], 0)
        self.assertGreater(report["flops"], 0)


class DMCountBridgeTests(unittest.TestCase):
    def test_train_collate_matches_upstream_tuple_contract(self):
        from external.baselines.dm_count.local_adapters.runner import dmcount_train_collate

        batch = [
            (
                torch.zeros(3, 32, 40, dtype=torch.float32),
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                torch.zeros(1, 4, 5, dtype=torch.float32),
            ),
            (
                torch.ones(3, 32, 40, dtype=torch.float32),
                torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
                torch.ones(1, 4, 5, dtype=torch.float32),
            ),
        ]

        images, points, gt_discrete = dmcount_train_collate(batch)

        self.assertEqual(tuple(images.shape), (2, 3, 32, 40))
        self.assertEqual(len(points), 2)
        self.assertEqual(tuple(gt_discrete.shape), (2, 1, 4, 5))

    def test_evaluate_model_returns_unified_metrics_and_predictions(self):
        from external.baselines.dm_count.local_adapters.runner import evaluate_dmcount_model

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                density = torch.ones(batch, 1, 4, 5, dtype=x.dtype, device=x.device) * 0.1
                return density, density

        dataloader = torch.utils.data.DataLoader(_FakeValDataset(), batch_size=1, shuffle=False)
        report = evaluate_dmcount_model(FakeModel(), dataloader, device="cpu")

        self.assertIn("metrics", report)
        self.assertIn("rows", report)
        self.assertEqual(len(report["rows"]), 2)
        self.assertEqual(set(report["metrics"].keys()), {"mae", "mse", "mape"})


class DMCountTrainBridgeTests(unittest.TestCase):
    def test_build_dmcount_dataloaders_returns_train_and_val(self):
        import external.baselines.dm_count.local_adapters.train_bridge as bridge
        from external.baselines.dm_count.local_adapters.datasets import DMCountDatasetAdapter

        cfg = {"training": {"batch_size": 2, "num_workers": 0}}

        fake_sets = {
            "train": DMCountDatasetAdapter(_FakeCountingDataset(), split="train", downsample_ratio=8),
            "val": DMCountDatasetAdapter(_FakeCountingDataset(), split="val", downsample_ratio=8),
        }

        with mock.patch.object(bridge, "build_dmcount_datasets", return_value=fake_sets):
            dataloaders = bridge.build_dmcount_dataloaders(cfg, dataset_name="gwhd")

        self.assertIn("train", dataloaders)
        self.assertIn("val", dataloaders)

    def test_run_train_batch_returns_expected_loss_terms(self):
        import external.baselines.dm_count.local_adapters.train_bridge as bridge

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

            def forward(self, x):
                out = torch.relu(self.conv(x))
                out = torch.nn.functional.interpolate(out, size=(4, 4), mode="bilinear", align_corners=False)
                normed = out / (out.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)
                return out, normed

        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mae_loss = torch.nn.L1Loss()
        tv_loss = torch.nn.L1Loss(reduction="none")

        batch = (
            torch.randn(2, 3, 32, 32),
            [torch.tensor([[4.0, 8.0]], dtype=torch.float32), torch.tensor([[8.0, 12.0]], dtype=torch.float32)],
            torch.ones(2, 1, 4, 4),
        )

        class FakeOTLoss:
            def __call__(self, normed_density, unnormed_density, points):
                total = unnormed_density.sum() * 0.0
                return total, 0.0, total

        loss_dict = bridge.run_train_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            ot_loss=FakeOTLoss(),
            tv_loss=tv_loss,
            mae_loss=mae_loss,
            wot=0.1,
            wtv=0.01,
            device="cpu",
        )

        self.assertEqual(set(loss_dict.keys()), {"loss", "ot_loss", "count_loss", "tv_loss", "mae", "mse"})


class DMCountRunnerCliTests(unittest.TestCase):
    def test_parse_args_accepts_config_and_dataset(self):
        from external.baselines.dm_count.local_adapters.run_local import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--config", "pack/config/gwhd/config_gwhd_light_full.yaml", "--dataset-name", "gwhd"])

        self.assertEqual(args.dataset_name, "gwhd")
        self.assertEqual(args.config, "pack/config/gwhd/config_gwhd_light_full.yaml")


if __name__ == "__main__":
    unittest.main()
