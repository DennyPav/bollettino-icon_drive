"""Versioned bulletin publication. No network or scientific imports at module import time."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


PREFIX = "bulletins/v1"


def run_id(run):
    if run.tzinfo is None:
        raise ValueError("The forecast run must include its UTC timezone")
    utc = run.astimezone(timezone.utc)
    if utc.hour not in (0, 12) or utc.minute or utc.second or utc.microsecond:
        raise ValueError("Expected a 00/12 UTC forecast run")
    return utc.strftime("%Y%m%d%H")


def expected_dates(run):
    run_id(run)
    run = run.astimezone(timezone.utc)
    day = run.astimezone(timezone.utc).date()
    return [(day + timedelta(days=d)).isoformat() for d in (range(3) if run.hour == 0 else (1, 2))]


def candidate_runs(now, count=4):
    if now.tzinfo is None:
        raise ValueError("UTC time required")
    utc = now.astimezone(timezone.utc)
    latest = utc.replace(hour=12 if utc.hour >= 12 else 0, minute=0, second=0, microsecond=0)
    return [latest - timedelta(hours=12 * i) for i in range(count)]


def validate_dataset_runs(data, run, variables):
    """Static terrain is exempt; every meteorological GRIB must declare this exact run."""
    expected = run.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    for name in variables:
        if name not in data:
            raise ValueError(f"Missing required dataset: {name}")
        if name == "HSURF":
            continue
        dataset = data[name]
        if "time" not in dataset.coords:
            raise ValueError(f"Missing GRIB run identity: {name}")
        # xarray exposes numpy datetime64; str also handles a scalar Python datetime.
        actual = str(dataset.coords["time"].values).replace(" ", "T")
        if actual[:19] != expected:
            raise ValueError(f"GRIB run mismatch: {name}: {actual} != {expected}")


def publish_bulletins(client, bucket, run, images, *, now=None):
    """Commit one run only after every expected image is uploaded and verified.

    ``images`` maps ISO valid dates to encoded WebP bytes. A failure leaves the
    previous run manifest intact. GitHub workflow concurrency serializes writers.
    Root aliases remain available to older app versions, but are not authoritative.
    """
    identity = run_id(run)
    run = run.astimezone(timezone.utc)
    expected = expected_dates(run)
    if set(images) != set(expected) or any(not isinstance(b, bytes) or not b for b in images.values()):
        raise ValueError("Publication is missing expected days or contains empty images")
    manifest_key = f"{PREFIX}/runs/{identity}.json"
    previous = None
    try:
        response = client.get_object(Bucket=bucket, Key=manifest_key)
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code not in ("NoSuchKey", "404", "NotFound"):
            raise
    else:
        with response["Body"] as body:
            previous = json.loads(body.read())
        if previous.get("run") != identity or previous.get("schemaVersion") != 1:
            raise ValueError("Existing publication metadata is inconsistent")

    assets = []
    for valid_date in sorted(expected):
        content = images[valid_date]
        digest = hashlib.sha256(content).hexdigest()
        name = f"{datetime.fromisoformat(valid_date):%d-%m-%Y}_{run:%H}.webp"
        key = f"{PREFIX}/assets/{identity}/{digest}/{name}"
        client.put_object(Bucket=bucket, Key=key, Body=content, ContentType="image/webp",
                          CacheControl="public, max-age=31536000, immutable",
                          Metadata={"sha256": digest, "run": identity, "valid-date": valid_date})
        # Verify bytes, not only object existence or a multipart ETag.
        response = client.get_object(Bucket=bucket, Key=key)
        with response["Body"] as body:
            uploaded = body.read()
        if hashlib.sha256(uploaded).hexdigest() != digest:
            raise IOError(f"Uploaded bulletin checksum mismatch: {key}")
        assets.append({"date": valid_date, "run": identity, "key": key, "sha256": digest})

    # Identical reruns are idempotent and cannot replace a manifest with different metadata.
    if previous and previous.get("assets") == assets and previous.get("complete") is True:
        manifest = previous
    else:
        published_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        manifest = {"schemaVersion": 1, "run": identity,
                    "revision": int(previous["revision"]) + 1 if previous else 1,
                    "publishedAt": published_at.isoformat().replace("+00:00", "Z"),
                    "complete": True, "assets": assets}
        client.put_object(Bucket=bucket, Key=manifest_key,
                          Body=json.dumps(manifest, sort_keys=True).encode("utf-8"),
                          ContentType="application/json", CacheControl="no-cache, max-age=0, must-revalidate")

    # Keep existing clients working; any failure is visible in the workflow and can be retried.
    for asset in assets:
        name = asset["key"].rsplit("/", 1)[1]
        client.put_object(Bucket=bucket, Key=name, Body=images[asset["date"]],
                          ContentType="image/webp", CacheControl="no-cache, max-age=0, must-revalidate")
    return manifest


def write_output_marker(output_dir, run, valid_dates):
    if set(valid_dates) != set(expected_dates(run)):
        raise ValueError("Rendered days do not match the run coverage")
    folder = run.astimezone(timezone.utc).strftime("%Y_%m_%d")
    marker = {"run": run_id(run), "runFolder": folder}
    path = Path(output_dir) / "publication.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker), encoding="utf-8")
    temporary.replace(path)
