import unittest
from pathlib import Path
from importlib.util import find_spec


class BaselineComplexitySmokeTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch") is not None, "torch is required for model complexity measurement")
    def test_measurement_supports_light_fccnet_config(self):
        from pack.tools.measure_model_complexity import measure_from_config

        config_path = Path("pack/config/gwhd/config_gwhd_light_full.yaml")
        report = measure_from_config(config_path, input_shape=(1, 3, 256, 256))

        self.assertEqual(report["model_name"], "light_fccnet")
        self.assertEqual(tuple(report["input_shape"]), (1, 3, 256, 256))
        self.assertGreater(report["params"], 0)
        self.assertGreater(report["flops"], 0)

    @unittest.skipUnless(find_spec("torch") is not None, "torch is required for model complexity measurement")
    def test_measurement_supports_csrnet_config(self):
        from pack.tools.measure_model_complexity import measure_from_config

        config_path = Path("pack/config/gwhd/config_gwhd_csrnet.yaml")
        report = measure_from_config(config_path, input_shape=(1, 3, 256, 256))

        self.assertEqual(report["model_name"], "csrnet")
        self.assertEqual(tuple(report["input_shape"]), (1, 3, 256, 256))
        self.assertGreater(report["params"], 0)
        self.assertGreater(report["flops"], 0)

    def test_cli_shape_parser_accepts_four_integer_values(self):
        from pack.tools.measure_model_complexity import parse_input_shape

        parsed = parse_input_shape(["1", "3", "1080", "1920"])

        self.assertEqual(parsed, (1, 3, 1080, 1920))

    def test_cli_shape_parser_rejects_non_4d_shape(self):
        from pack.tools.measure_model_complexity import parse_input_shape

        with self.assertRaises(ValueError):
            parse_input_shape(["3", "1080", "1920"])


if __name__ == "__main__":
    unittest.main()
