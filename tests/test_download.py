"""Exercise the real downloader function without importing the GIS/model pipeline."""
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest


class Response:
    text = "directory"
    def __init__(self, *, fail=False):
        self.fail = fail
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, **kwargs):
        yield b"first part"
        if self.fail:
            raise IOError("interrupted GRIB")
        yield b" complete"


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.destination = self.directory / "T_2M.grib"
        self.destination.write_bytes(b"previous bytes")
        self.run = datetime(2026, 11, 5, tzinfo=timezone.utc)

    def tearDown(self):
        self.temporary.cleanup()

    def downloader(self, *, missing=False, interrupted=False):
        module = ast.parse((Path(__file__).resolve().parents[1] / "first_guess_icon_v4git.py").read_text(encoding="utf-8"))
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "download_icon_data")
        requests = []
        def get(url, **kwargs):
            requests.append(url)
            return Response(fail=interrupted and kwargs.get("stream", False))
        soup = SimpleNamespace(find_all=lambda *_: [] if missing else [{"href": "model.grib"}])
        namespace = {"DATA_DIR": str(self.directory), "VARIABLES": ["T_2M"], "os": os, "shutil": shutil,
                     "requests": SimpleNamespace(get=get), "BeautifulSoup": lambda *_: soup}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "downloader-under-test", "exec"), namespace)
        return namespace["download_icon_data"], requests

    def test_empty_listing_cannot_reuse_an_existing_grib(self):
        download, _ = self.downloader(missing=True)
        with self.assertRaises(RuntimeError):
            download(self.run)
        self.assertEqual(b"previous bytes", self.destination.read_bytes())

    def test_interrupted_download_cannot_replace_the_previous_file(self):
        download, _ = self.downloader(interrupted=True)
        with self.assertRaises(IOError):
            download(self.run)
        self.assertEqual(b"previous bytes", self.destination.read_bytes())
        self.assertFalse(self.destination.with_suffix(".grib.part").exists())

    def test_complete_download_replaces_bytes_and_uses_the_explicit_run(self):
        download, requests = self.downloader()
        download(self.run)
        self.assertEqual(b"first part complete", self.destination.read_bytes())
        self.assertTrue(all("/2026110500/" in url for url in requests))
