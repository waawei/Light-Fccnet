import warnings

from .csrnet import CSRNet
from .light_fccnet import LightFCCNet

__all__ = ["CSRNet", "LightFCCNet"]


def build_model(config: dict):
    model_cfg = config.get("model", {})
    training_cfg = config.setdefault("training", {})
    model_name = model_cfg.get("name", "light_fccnet")

    use_p1 = bool(model_cfg.get("use_p1", False))
    requested_use_p2 = bool(model_cfg.get("use_p2", False))
    use_p2 = requested_use_p2 and use_p1
    legacy_use_p3 = model_cfg.get("use_p3")

    if requested_use_p2 and not use_p1:
        warnings.warn(
            "model.use_p2=True requires model.use_p1=True; disabling use_p2 for paper-aligned semantics.",
            UserWarning,
            stacklevel=2,
        )

    if "use_p3_loss" not in training_cfg and legacy_use_p3 is not None:
        training_cfg["use_p3_loss"] = bool(legacy_use_p3)

    if model_name == "light_fccnet":
        return LightFCCNet(
            in_channels=model_cfg.get("in_channels", 3),
            input_size=tuple(model_cfg.get("input_size", [256, 256])),
            stage_channels=tuple(model_cfg.get("stage_channels", [32, 64, 96, 128])),
            fusion_channels=int(model_cfg.get("fusion_channels", 96)),
            spatial_max_tokens=int(model_cfg.get("spatial_max_tokens", 1024)),
            head_init_bias=float(model_cfg.get("head_init_bias", -9.0)),
            use_p1=use_p1,
            use_p2=use_p2,
        )
    if model_name == "csrnet":
        return CSRNet(
            in_channels=model_cfg.get("in_channels", 3),
            input_size=tuple(model_cfg.get("input_size", [256, 256])),
            frontend_channels=tuple(
                model_cfg.get("frontend_channels", [64, 64, 128, 128, 256, 256, 256, 512, 512, 512])
            ),
            frontend_pool_indices=tuple(model_cfg.get("frontend_pool_indices", [1, 3, 6])),
            backend_channels=tuple(model_cfg.get("backend_channels", [512, 512, 512, 256, 128, 64])),
            dilation=int(model_cfg.get("dilation", 2)),
        )

    raise ValueError(
        f"Unsupported model.name: {model_name}. Expected: light_fccnet or csrnet"
    )
