from __future__ import annotations

import importlib.util
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


def _load_version_module(root: str):
    path = Path(root) / "kitpaw" / "version.py"
    spec = importlib.util.spec_from_file_location("kitpaw_build_version", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load version helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GitDescribeMetadataHook(MetadataHookInterface):
    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        version_module = _load_version_module(self.root)
        metadata["version"] = version_module.get_version()


def get_metadata_hook():
    return GitDescribeMetadataHook
