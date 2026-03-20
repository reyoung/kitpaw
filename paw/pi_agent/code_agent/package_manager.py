from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class InstalledPackage:
    source: str
    scope: str
    path: str


class PackageManager:
    def __init__(self, cwd: str, agent_dir: str, settings_manager) -> None:
        self.cwd = Path(cwd).resolve()
        self.agent_dir = Path(agent_dir).resolve()
        self.settings_manager = settings_manager
        self.packages_dir = self.agent_dir / "packages"
        self.packages_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, scope: str) -> Path:
        base = self.agent_dir if scope == "user" else self.cwd / ".pi"
        base.mkdir(parents=True, exist_ok=True)
        return base / "packages.json"

    def _read_manifest(self, scope: str) -> list[dict[str, str]]:
        path = self._manifest_path(scope)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, scope: str, entries: list[dict[str, str]]) -> None:
        path = self._manifest_path(scope)
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def _key_for_source(self, source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def get_installed_path(self, source: str, scope: str = "user") -> str | None:
        for entry in self._read_manifest(scope):
            if entry["source"] == source:
                return entry["path"]
        return None

    def list(self) -> list[InstalledPackage]:
        results: list[InstalledPackage] = []
        for scope in ("user", "project"):
            for entry in self._read_manifest(scope):
                results.append(InstalledPackage(source=entry["source"], scope=scope, path=entry["path"]))
        return results

    def install(self, source: str, local: bool = False) -> str:
        scope = "project" if local else "user"
        target = self.packages_dir / self._key_for_source(source)
        if target.exists():
            shutil.rmtree(target)
        if source.startswith("git:"):
            repo = source[4:]
            subprocess.run(["git", "clone", repo, str(target)], check=True, cwd=str(self.cwd))
        else:
            src = Path(source).expanduser()
            if not src.is_absolute():
                src = (self.cwd / src).resolve()
            if not src.exists():
                raise ValueError(f"Package source not found: {source}")
            if src.is_dir():
                shutil.copytree(src, target)
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target / src.name)
        entries = [entry for entry in self._read_manifest(scope) if entry["source"] != source]
        entries.append({"source": source, "path": str(target)})
        self._write_manifest(scope, entries)
        return str(target)

    def remove(self, source: str, local: bool = False) -> bool:
        scope = "project" if local else "user"
        entries = self._read_manifest(scope)
        kept: list[dict[str, str]] = []
        removed_path: str | None = None
        for entry in entries:
            if entry["source"] == source:
                removed_path = entry["path"]
                continue
            kept.append(entry)
        if removed_path is None:
            return False
        self._write_manifest(scope, kept)
        shutil.rmtree(removed_path, ignore_errors=True)
        return True

    def update(self, source: str | None = None) -> list[str]:
        updated: list[str] = []
        for package in self.list():
            if source is not None and package.source != source:
                continue
            if package.source.startswith("git:"):
                subprocess.run(["git", "-C", package.path, "pull", "--ff-only"], check=True, cwd=str(self.cwd))
                updated.append(package.source)
            elif Path(package.source).exists():
                self.install(package.source, local=package.scope == "project")
                updated.append(package.source)
        return updated
