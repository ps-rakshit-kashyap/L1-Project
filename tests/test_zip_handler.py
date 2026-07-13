"""Smoke test for ZIP handler construction.

This test makes sure the upload root can be overridden during testing so the
extractor stays isolated from the real project directory.
"""

from pathlib import Path

from utils.zip_handler import ZipHandler


def test_zip_handler_init(tmp_path: Path):
    handler = ZipHandler(tmp_path)
    assert handler.upload_root == tmp_path
