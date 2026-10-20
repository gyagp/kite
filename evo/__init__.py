"""ONNX export recipes and evidence for ORT WebGPU."""

__version__ = "0.2.0"


def __getattr__(name):
    # Keep the version import usable while the pipeline imports package metadata.
    if name == "build":
        from evo.api import build

        return build
    raise AttributeError(name)


__all__ = ["build"]
