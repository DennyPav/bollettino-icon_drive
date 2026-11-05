# bollettino-icon_drive

The scheduled workflow runs `first_guess_icon_v4git.py`. The older v3 script is
retained but does not implement the versioned publication protocol.

## Versioned Android bulletin publication

Root WebP names remain compatible with existing clients. They are mutable aliases
and must never be used to determine a forecast run. New clients use:

```text
bulletins/v1/runs/2026110500.json
bulletins/v1/assets/2026110500/<sha256>/05-11-2026_00.webp
```

The manifest contains `schemaVersion: 1`, the full UTC `run`, a positive numeric
`revision`, `publishedAt`, `complete: true`, and `assets`. Each asset declares its
ISO valid `date`, actual `run`, object `key`, and `sha256`. A 00 run contains today,
tomorrow and the following day; a 12 run contains only tomorrow and the following
day. Today's image remains from the 00 run and must be labelled accordingly.

Images are uploaded to content-addressed keys and read back for checksum
verification before the run manifest is committed. A partial or failed upload
never advertises a new edition. Repeating identical content preserves the revision;
changed content increments it while preserving old object keys. Independent
per-run manifests prevent a delayed old run from replacing the index of a newer
run. The workflow concurrency group serializes publishers, including manual jobs;
all writers to these keys must use the same serialization policy. The three-day
client view can combine the newest available actual run for each valid date.

Run manifests and legacy aliases require HTTP revalidation. Immutable asset keys
have a one-year immutable cache policy. No new credentials or bucket are required:
the existing `R2_ACCESS_KEY`, `R2_SECRET_KEY` and `R2_ENDPOINT` secrets and the
`bollettini` bucket are used. The public R2 base URL stays unchanged. This repository
does not delete historical objects; define a bucket lifecycle separately after
choosing the supported history period.

The generator tries four UTC run candidates in descending order. Downloads live
in separate run directories and fail on missing variables or inconsistent GRIB
initialization times. No old pickle is relabelled as a new run. The Drive upload
uses `output/publication.json` to select the actual output date, including a
previous-day fallback, rather than the workflow machine's current date.

Before production rollout, run the normal scientific workflow on a test bucket
with representative upstream GRIB data and verify the emitted run/coverage. The
strict completeness check requires every variable in `VARIABLES`; upstream missing
fields now reject that candidate rather than silently substituting stale data.
The unit tests below validate the protocol and failure paths, not the scientific
forecast calculations or credentials of a live bucket.

```bash
python -m unittest discover -s tests -v
python -m py_compile first_guess_icon_v4git.py bulletin_publication.py
```

Android clients can ship before this producer change: until manifests exist, a
migration reader can use the dated Drive folder and its file checksum, validating
matching R2 PNG bytes or downloading the Drive file. A converted R2 WebP and a
Drive PNG have different encoded bytes and must not be compared as the same
checksum. After the first successful versioned publication, clients can use the
small run manifests and cached immutable WebP files directly.
