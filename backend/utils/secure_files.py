from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

logger = logging.getLogger(__name__)


def secure_delete_file(file_path: Path | str | None) -> bool:
    """Safely removes a file from disk if it exists."""
    if not file_path:
        return False
    path = Path(file_path)
    if path.exists() and path.is_file():
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            logger.warning("Failed to remove temporary file %s: %s", path, exc)
    return False


@contextmanager
def temporary_upload_file(
    file_bytes: bytes,
    suffix: str = ".tmp",
    prefix: str = "autoform_",
) -> Generator[Path, None, None]:
    """
    Saves binary bytes into a temporary file on disk and guarantees its
    immediate removal upon exiting the context block, even on exceptions.
    """
    # NamedTemporaryFile creates files with restrictive permissions (0600 on
    # POSIX). Keep the file private while it exists; unlinking is not a promise
    # of forensic erasure on SSDs or journaled filesystems.
    temp_file = NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(file_bytes)
        temp_file.flush()
        temp_file.close()
        yield temp_path
    finally:
        secure_delete_file(temp_path)
