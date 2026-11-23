"""Stub for the compiled `chamfer` extension.

`extensions/chamfer_dist/__init__.py` does `import chamfer` at import time, and both
model files import it, but Chamfer distance is only used for pre-training. Classification
never calls it; if something does, fail loudly instead of silently.
"""
SHIM = True


def _unavailable(*_a, **_k):
    raise RuntimeError("chamfer extension not built (classification-only env); "
                       "build extensions/chamfer_dist for pre-training")


forward = backward = _unavailable
