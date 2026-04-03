import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


class _FakeCountingDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 32, 40, dtype=torch.float32),
            "density": torch.ones(1, 32, 40, dtype=torch.float32) / 640.0,
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
            "density": torch.zeros(1, 24, 24, dtype=torch.float32),
            "count": torch.tensor(0.0),
            "points": torch.zeros((0, 2), dtype=torch.float32),
            "image_name": "empty_case",
        }


class SASNetDensityTargetTests(unittest.TestCase):
    def test_build_density_target_preserves_mass(self):
        from external.baselines.sasnet.local_adapters.density_targets import build_sasnet_density_target

        density = torch.zeros(1, 32, 40, dtype=torch.float32)
        density[0, 8, 4] = 1.0
        density[0, 12, 20] = 1.0

        target = build_sasnet_density_target(density=density, points=None, image_shape=(32, 40), sigma=4)

        self.assertEqual(tuple(target.shape), (1, 32, 40))
        self.assertAlmostEqual(float(target.sum().item()), 2.0, places=4)

    def test_build_density_target_handles_empty_target(self):
        from external.baselines.sasnet.local_adapters.density_targets import build_sasnet_density_target

        target = build_sasnet_density_target(density=None, points=torch.zeros((0, 2)), image_shape=(24, 24), sigma=4)

        self.assertEqual(tuple(target.shape), (1, 24, 24))
        self.assertAlmostEqual(float(target.sum().item()), 0.0, places=5)


class SASNetDatasetAdapterTests(unittest.TestCase):
    def test_train_adapter_returns_upstream_training_tuple(self):
        from external.baselines.sasnet.local_adapters.datasets import SASNetDatasetAdapter

        adapter = SASNetDatasetAdapter(_FakeCountingDataset(), split="train", sigma=4)

        image, density = adapter[0]

        self.assertEqual(tuple(image.shape), (3, 32, 40))
        self.assertEqual(tuple(density.shape), (1, 32, 40))
        self.assertGreaterEqual(float(density.sum().item()), 0.0)

    def test_val_adapter_returns_upstream_validation_tuple(self):
        from external.baselines.sasnet.local_adapters.datasets import SASNetDatasetAdapter

        adapter = SASNetDatasetAdapter(_FakeCountingDataset(), split="val", sigma=4)

        image, count, name = adapter[0]

        self.assertEqual(tuple(image.shape), (3, 32, 40))
        self.assertAlmostEqual(float(count), 2.0, places=5)
        self.assertEqual(name, "sample_001")

    def test_train_adapter_supports_empty_points(self):
        from external.baselines.sasnet.local_adapters.datasets import SASNetDatasetAdapter

        adapter = SASNetDatasetAdapter(_FakeEmptyPointDataset(), split="train", sigma=4)

        image, density = adapter[0]

        self.assertEqual(tuple(image.shape), (3, 24, 24))
        self.assertEqual(tuple(density.shape), (1, 24, 24))
        self.assertAlmostEqual(float(density.sum().item()), 0.0, places=5)


class SASNetDatasetBuilderTests(unittest.TestCase):
    def test_build_sasnet_datasets_uses_gwhd_source_and_wraps_splits(self):
        import external.baselines.sasnet.local_adapters.datasets as sas_datasets

        cfg = {
            "model": {"input_size": [256, 256]},
            "data": {
                "sigma": 4,
                "gwhd_train_csv": "train.csv",
                "gwhd_val_csv": "val.csv",
                "gwhd_images_dir": "images",
            },
        }

        with mock.patch.object(sas_datasets, "GWHDDataset", side_effect=[_FakeCountingDataset(), _FakeCountingDataset()]) as dataset_cls:
            with mock.patch.object(sas_datasets, "get_train_transforms", return_value="train_tf"):
                with mock.patch.object(sas_datasets, "get_val_transforms", return_value="val_tf"):
                    datasets = sas_datasets.build_sasnet_datasets(cfg, dataset_name="gwhd")

        self.assertIn("train", datasets)
        self.assertIn("val", datasets)
        self.assertEqual(dataset_cls.call_count, 2)
        self.assertEqual(len(datasets["train"][0]), 2)
        self.assertEqual(len(datasets["val"][0]), 3)

    def test_build_sasnet_datasets_uses_mtc_split_files(self):
        import external.baselines.sasnet.local_adapters.datasets as sas_datasets

        cfg = {
            "model": {"input_size": [256, 256]},
            "data": {
                "sigma": 4,
                "mtc_root": "mtc_root",
                "mtc_train_split_file": "train.txt",
                "mtc_val_split_file": "val.txt",
            },
        }

        with mock.patch.object(sas_datasets, "MTCDataset", side_effect=[_FakeCountingDataset(), _FakeCountingDataset()]) as dataset_cls:
            with mock.patch.object(sas_datasets, "get_train_transforms", return_value="train_tf"):
                with mock.patch.object(sas_datasets, "get_val_transforms", return_value="val_tf"):
                    sas_datasets.build_sasnet_datasets(cfg, dataset_name="mtc")

        self.assertEqual(dataset_cls.call_count, 2)
        self.assertEqual(dataset_cls.call_args_list[0].kwargs["split"], "train")
        self.assertEqual(dataset_cls.call_args_list[1].kwargs["split"], "val")

    def test_build_sasnet_datasets_rejects_unknown_dataset_name(self):
        import external.baselines.sasnet.local_adapters.datasets as sas_datasets

        with self.assertRaises(ValueError):
            sas_datasets.build_sasnet_datasets({"model": {"input_size": [256, 256]}, "data": {}}, dataset_name="unknown")


class SASNetEvalExportTests(unittest.TestCase):
    def test_compute_count_metrics_returns_unified_project_fields(self):
        from external.baselines.sasnet.local_adapters.eval import compute_count_metrics

        metrics = compute_count_metrics(pred_counts=[10, 12, 9], gt_counts=[8, 10, 10])

        self.assertEqual(set(metrics.keys()), {"mae", "mse", "mape"})
        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["mse"], 0.0)
        self.assertGreaterEqual(metrics["mape"], 0.0)

    def test_build_result_row_keeps_paper_table_fields(self):
        from external.baselines.sasnet.local_adapters.eval import build_result_row

        row = build_result_row(
            dataset="GWHD",
            metrics={"mae": 1.0, "mse": 2.0, "mape": 3.0},
            result_type="adapted reproduction",
            params=100,
            flops=200,
            checkpoint_path="ckpt.pth",
        )

        self.assertEqual(row["method"], "SASNet")
        self.assertEqual(row["dataset"], "GWHD")
        self.assertEqual(row["result_type"], "adapted reproduction")
        self.assertEqual(row["params"], 100)
        self.assertEqual(row["flops"], 200)
        self.assertEqual(row["checkpoint_path"], "ckpt.pth")

    def test_export_helpers_write_json_and_csv_rows(self):
        from external.baselines.sasnet.local_adapters.export_results import append_result_row_csv, save_result_row_json

        row = {
            "method": "SASNet",
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
            self.assertIn("SASNet", json_path.read_text(encoding="utf-8"))
            self.assertIn("dataset", csv_path.read_text(encoding="utf-8"))


class SASNetComplexityWrapperTests(unittest.TestCase):
    def test_build_sasnet_model_for_complexity_skips_weight_download(self):
        from external.baselines.sasnet.local_adapters.measure_complexity import build_sasnet_model_for_complexity

        model = build_sasnet_model_for_complexity()

        self.assertTrue(hasattr(model, "forward"))

    def test_measure_sasnet_complexity_returns_positive_counts(self):
        from external.baselines.sasnet.local_adapters.measure_complexity import measure_sasnet_complexity

        report = measure_sasnet_complexity(input_shape=(1, 3, 64, 64))

        self.assertEqual(report["model_name"], "sasnet")
        self.assertEqual(tuple(report["input_shape"]), (1, 3, 64, 64))
        self.assertGreater(report["params"], 0)
        self.assertGreater(report["flops"], 0)


if __name__ == "__main__":
    unittest.main()
