"""ZIP upload handling and safe extraction.

This module exists to turn an uploaded archive into a real repository folder
without exposing the app to path traversal attacks or Windows file-locking
problems. The extraction logic is intentionally defensive because users may
upload malformed or partially locked archives.
"""

from __future__ import annotations

import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path

from config import settings


logger = logging.getLogger(__name__)


class ZipHandlerError(Exception):
    """Raised when ZIP handling fails in a user-visible way."""


class ZipHandler:
    """Manage repository upload extraction.

    The handler owns the upload destination and makes sure the extracted
    repository is always placed under the configured project folder.
    """

    def __init__(self, upload_root: Path | None = None) -> None:
        # Store uploaded repositories under the configured project folder.
        self.upload_root = upload_root or settings.upload_path
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def extract_zip(self, zip_path: Path) -> Path:
        # Validate the file type first so the extractor never treats random
        # uploads as archives.
        if not zipfile.is_zipfile(zip_path):
            raise ZipHandlerError("Invalid ZIP file.")

        repo_name = zip_path.stem
        destination = self.upload_root / repo_name
        # Best-effort cleanup so rerunning the upload flow does not leave stale
        # files behind.
        if destination.exists():
            try:
                shutil.rmtree(destination, ignore_errors=True)
            except Exception:
                pass
        if destination.exists():
            # If Windows keeps the previous folder locked, fall back to a
            # timestamped folder name instead of failing the upload.
            destination = self.upload_root / f"{repo_name}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        destination.mkdir(parents=True, exist_ok=True)

        # Extract member-by-member so one bad entry does not abort the entire
        # upload process.
        with zipfile.ZipFile(zip_path) as archive:
            self._safe_extract(archive, destination)
        return destination

    def _safe_extract(self, archive: zipfile.ZipFile, destination: Path) -> None:
        # Reject path traversal attacks before writing anything to disk.
        dest_resolved = destination.resolve()
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if dest_resolved not in member_path.parents and member_path != dest_resolved:
                raise ZipHandlerError("ZIP contains unsafe paths.")
        # Once validated, extract each entry with permission-aware error
        # handling.
        for member in archive.infolist():
            self._extract_member(archive, member, destination)

    def _extract_member(self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> None:
        target_path = destination / member.filename
        try:
            # Preserve directory structure from the uploaded archive so the
            # repository looks the same after extraction.
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                return

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        except PermissionError as exc:
            logger.warning("Skipping locked ZIP member %s: %s", member.filename, exc)
        except OSError as exc:
            logger.warning("Skipping ZIP member %s due to filesystem error: %s", member.filename, exc)
