from __future__ import annotations

import json
import os
import errno
import tempfile
from pathlib import Path


class CorruptManifestError(RuntimeError):
    """The manifest exists but cannot be parsed as the current schema."""


class MissingRevisionError(RuntimeError):
    """The manifest points to a revision that cannot be read safely."""


class RevisionManifestStore:
    """Atomic, versioned manifest storage for source and index revisions."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, source: str | None):
        self.root = Path(root)
        self.source = source
        self.path = self.root / "index_manifest.json" if source is None else self.root / source / "manifest.json"

    def read(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
            raise CorruptManifestError(f"manifest for {self.source!r} is corrupt") from error
        self._validate_schema(data)
        return data

    def write(self, *, profile: dict, active_revision: str | None, cursor: str | None) -> None:
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "profile": profile,
            "active_revision": active_revision,
            "cursor": cursor,
        }
        self._validate_schema(data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary_path, self.path)
            self._fsync_directory(self.path.parent)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Best-effort fsync of a directory entry after rename.

        Guarantees the rename is durable across power loss / kernel crash, not
        just visible to the running process. Skipped where the platform does not
        support directory fsync (e.g. some non-POSIX FS) rather than raising.
        """
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError as error:
            if error.errno in (errno.ENOTSUP, errno.EINVAL, errno.EBADF, errno.EACCES):
                return
            raise
        finally:
            os.close(directory_fd)

    def active_revision(self) -> str | None:
        manifest = self.read()
        if manifest is None or manifest["active_revision"] is None:
            return None
        self._validate_active_revision(manifest["active_revision"])
        return manifest["active_revision"]

    def validate(self) -> None:
        active_revision = self.read()
        if active_revision is not None and active_revision["active_revision"] is not None:
            self._validate_active_revision(active_revision["active_revision"])

    @classmethod
    def _validate_schema(cls, data: object) -> None:
        if not isinstance(data, dict):
            raise CorruptManifestError("manifest must be a JSON object")
        expected = {"schema_version", "profile", "active_revision", "cursor"}
        if set(data) != expected or data["schema_version"] != cls.SCHEMA_VERSION:
            raise CorruptManifestError("manifest schema is unsupported")
        if not isinstance(data["profile"], dict):
            raise CorruptManifestError("manifest profile must be an object")
        if data["active_revision"] is not None and not isinstance(data["active_revision"], str):
            raise CorruptManifestError("manifest active_revision must be a string or null")
        if data["cursor"] is not None and not isinstance(data["cursor"], str):
            raise CorruptManifestError("manifest cursor must be a string or null")

    @staticmethod
    def _validate_active_revision(active_revision: str) -> None:
        revision_path = Path(active_revision)
        docs_file = revision_path / "docs.jsonl"
        try:
            if not revision_path.is_dir() or not docs_file.is_file():
                raise OSError("revision directory or docs.jsonl is missing")
            with docs_file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
            raise MissingRevisionError(f"active revision is unavailable: {active_revision}") from error
