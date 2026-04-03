"""Helpers for checkpoint initialization across related model variants."""


def _shape_of(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def filter_compatible_state_dict(model_state_dict: dict, loaded_state_dict: dict):
    """
    Keep only keys that exist in both dicts and have matching shapes.

    Returns:
        matched: subset safe to load with strict=False
        skipped: mapping from key to skip reason
    """
    matched = {}
    skipped = {}

    for key, value in loaded_state_dict.items():
        if key not in model_state_dict:
            skipped[key] = "missing_in_model"
            continue

        model_shape = _shape_of(model_state_dict[key])
        loaded_shape = _shape_of(value)
        if model_shape != loaded_shape:
            skipped[key] = f"shape_mismatch:{loaded_shape}->{model_shape}"
            continue

        matched[key] = value

    return matched, skipped
