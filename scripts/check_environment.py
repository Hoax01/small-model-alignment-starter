"""Print required package versions and accelerator visibility."""

from __future__ import annotations

import importlib.metadata
import json

import torch


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main():
    report = {
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    report["two_gpus_visible"] = report["gpu_count"] >= 2
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()