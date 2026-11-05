import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import unittest

from bulletin_publication import (candidate_runs, expected_dates, publish_bulletins,
                                  validate_dataset_runs, write_output_marker)


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.fail_on_asset = None
        self.corrupt_reads = False

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject()
        content = self.objects[Key]
        if self.corrupt_reads and "/assets/" in Key:
            content += b"corrupt"
        return {"Body": io.BytesIO(content)}

    def put_object(self, *, Bucket, Key, Body, **metadata):
        if self.fail_on_asset and self.fail_on_asset in Key:
            raise IOError("simulated upload failure")
        self.objects[Key] = Body
        self.writes.append((Key, metadata))


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.run = datetime(2026, 11, 5, tzinfo=timezone.utc)
        self.client = FakeS3()

    def images(self, run=None, suffix=b""):
        return {day: day.encode() + suffix for day in expected_dates(run or self.run)}

    def test_complete_run_commits_after_all_verified_assets(self):
        manifest = publish_bulletins(self.client, "test", self.run, self.images())
        self.assertEqual(3, len(manifest["assets"]))
        keys = [key for key, _ in self.client.writes]
        marker = keys.index("bulletins/v1/runs/2026110500.json")
        self.assertEqual(3, marker)
        self.assertTrue(all("/assets/2026110500/" in key for key in keys[:marker]))
        self.assertTrue(self.client.writes[0][1]["CacheControl"].endswith("immutable"))

    def test_missing_image_never_begins_publication(self):
        images = self.images()
        images.pop(next(iter(images)))
        with self.assertRaises(ValueError):
            publish_bulletins(self.client, "test", self.run, images)
        self.assertFalse(self.client.writes)

    def test_partial_upload_keeps_previous_manifest_and_immutable_bytes(self):
        old = publish_bulletins(self.client, "test", self.run, self.images())
        key = "bulletins/v1/runs/2026110500.json"
        saved = self.client.objects[key]
        self.client.fail_on_asset = "06-11-2026"
        with self.assertRaises(IOError):
            publish_bulletins(self.client, "test", self.run, self.images(suffix=b"revision"))
        self.assertEqual(saved, self.client.objects[key])
        for asset in old["assets"]:
            self.assertEqual(self.images()[asset["date"]], self.client.objects[asset["key"]])

    def test_checksum_failure_does_not_advertise_the_run(self):
        self.client.corrupt_reads = True
        with self.assertRaises(IOError):
            publish_bulletins(self.client, "test", self.run, self.images())
        self.assertNotIn("bulletins/v1/runs/2026110500.json", self.client.objects)

    def test_identical_rerun_keeps_revision_and_changed_run_increments(self):
        first = publish_bulletins(self.client, "test", self.run, self.images())
        self.assertEqual(first, publish_bulletins(self.client, "test", self.run, self.images()))
        changed = publish_bulletins(self.client, "test", self.run, self.images(suffix=b"changed"))
        self.assertEqual(2, changed["revision"])
        self.assertNotEqual(first["assets"][0]["key"], changed["assets"][0]["key"])

    def test_old_run_cannot_replace_new_run_manifest(self):
        later = self.run.replace(hour=12)
        new = publish_bulletins(self.client, "test", later, self.images(later))
        publish_bulletins(self.client, "test", self.run, self.images())
        self.assertEqual(new, json.loads(self.client.objects["bulletins/v1/runs/2026110512.json"]))
        self.assertEqual(["2026-11-06", "2026-11-07"], expected_dates(later))

    def test_candidates_follow_utc_and_cross_month_boundary(self):
        now = datetime(2026, 11, 1, 0, 1, tzinfo=timezone.utc)
        runs = candidate_runs(now)
        self.assertEqual(datetime(2026, 10, 31, 12, tzinfo=timezone.utc), runs[1])
        self.assertEqual(12, candidate_runs(now.replace(hour=12))[0].hour)

    def test_marker_uses_actual_run_date_instead_of_job_date(self):
        run = self.run - timedelta(days=1)
        with tempfile.TemporaryDirectory() as directory:
            write_output_marker(directory, run, expected_dates(run))
            self.assertEqual("2026_11_04", json.loads((Path(directory) / "publication.json").read_text())["runFolder"])

    def test_mixed_or_missing_grib_run_is_rejected(self):
        class Coordinate:
            values = "2026-11-04T12:00:00.000000000"
        class Dataset:
            coords = {"time": Coordinate()}
        with self.assertRaises(ValueError):
            validate_dataset_runs({"T_2M": Dataset()}, self.run, ["T_2M"])
        with self.assertRaises(ValueError):
            validate_dataset_runs({}, self.run, ["T_2M"])
        Coordinate.values = "2026-11-05T00:00:00.000000000"
        validate_dataset_runs({"T_2M": Dataset()}, self.run, ["T_2M"])


if __name__ == "__main__":
    unittest.main()
