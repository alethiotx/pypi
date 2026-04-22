"""
Enrich CellProfiler CSVs with TIF metadata from Image_enriched.csv.

Downloads files from S3, streams the merge row-by-row (~12 MB memory),
and writes the enriched CSV locally.
"""

import csv
import subprocess
import tempfile
import time
from pathlib import Path


def _s3_download(s3_uri, local_path):
    """Download a file from S3 using the AWS CLI.

    :param s3_uri: S3 URI to download (e.g. ``s3://bucket/key``)
    :param local_path: Local path to write the file to
    :raises RuntimeError: If the download fails
    """
    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download {s3_uri}: {result.stderr.strip()}")


def list_metadata_columns(s3_path):
    """List available metadata columns in Image_enriched.csv.

    Downloads only ``Image_enriched.csv`` from the S3 output directory
    and returns the column names (excluding ``ImageNumber``).

    :param s3_path: S3 URI of the CellProfiler output directory
    :type s3_path: str
    :return: List of available column names
    :rtype: list[str]

    **Example**

    >>> from alethiotx.cellprofiler import list_metadata_columns
    >>> cols = list_metadata_columns("s3://example-bucket/my-experiment/")
    >>> print(cols[:5])
    ['PlateID', 'Well', 'Site', 'Z_Step', 'z_position']
    """
    s3_base = s3_path.rstrip("/")

    with tempfile.TemporaryDirectory() as tmpdir:
        ie_local = Path(tmpdir) / "Image_enriched.csv"
        _s3_download(f"{s3_base}/Image_enriched.csv", ie_local)

        with open(ie_local) as f:
            header = csv.DictReader(f).fieldnames

    return [c for c in header if c != "ImageNumber"]


def add_metadata(s3_path, csv_name, columns=None, output_dir=None):
    """Add TIF metadata to a CellProfiler CSV.

    Downloads ``csv_name`` and ``Image_enriched.csv`` from the S3 output
    directory, joins them on ``ImageNumber``, and writes
    ``<stem>_enriched.csv`` locally.

    The merge streams ``csv_name`` row-by-row so memory usage stays at
    ~12 MB regardless of file size. No pandas required.

    :param s3_path: S3 URI of the CellProfiler output directory
        (e.g. ``s3://example-bucket/my-experiment/``)
    :type s3_path: str
    :param csv_name: Name of the CSV to enrich
        (e.g. ``Cell.csv``, ``Nucleus.csv``, ``Cytoplasm.csv``)
    :type csv_name: str
    :param columns: Specific columns to add from Image_enriched.csv.
        If ``None``, all columns except ``ImageNumber`` are added.
    :type columns: list[str] or None
    :param output_dir: Directory to write the enriched CSV.
        Defaults to the current working directory.
    :type output_dir: str or Path or None
    :return: Path to the enriched CSV file
    :rtype: Path

    **Examples**

    Add all metadata columns to ``Cell.csv``::

        >>> from alethiotx.cellprofiler import add_metadata
        >>> out = add_metadata("s3://example-bucket/my-experiment/", "Cell.csv")
        >>> print(out)
        /Users/you/Cell_enriched.csv

    Add specific columns only::

        >>> out = add_metadata(
        ...     "s3://example-bucket/my-experiment/",
        ...     "Nucleus.csv",
        ...     columns=["PlateID", "Well", "Site", "Z_Step"],
        ... )

    Write to a specific directory::

        >>> out = add_metadata(
        ...     "s3://example-bucket/my-experiment/",
        ...     "Cell.csv",
        ...     output_dir="~/Downloads",
        ... )
    """
    s3_base = s3_path.rstrip("/")
    stem = Path(csv_name).stem
    output_dir = Path(output_dir or Path.cwd()).expanduser()
    output_file = output_dir / f"{stem}_enriched.csv"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ie_local = tmpdir / "Image_enriched.csv"
        csv_local = tmpdir / csv_name

        # Download Image_enriched.csv
        print(f"Downloading {s3_base}/Image_enriched.csv ...")
        _s3_download(f"{s3_base}/Image_enriched.csv", ie_local)

        # Read header and resolve columns
        with open(ie_local) as f:
            ie_header = csv.DictReader(f).fieldnames

        available_cols = [c for c in ie_header if c != "ImageNumber"]

        if columns is not None:
            missing = [c for c in columns if c not in ie_header]
            if missing:
                raise ValueError(
                    f"Columns not found in Image_enriched.csv: {missing}\n"
                    f"Use list_metadata_columns() to see available columns."
                )
            merge_cols = list(columns)
        else:
            merge_cols = available_cols

        # Download the target CSV
        print(f"Downloading {s3_base}/{csv_name} ...")
        _s3_download(f"{s3_base}/{csv_name}", csv_local)

        # Build lookup from Image_enriched (small — one row per image set)
        lookup = {}
        with open(ie_local) as f:
            for row in csv.DictReader(f):
                lookup[row["ImageNumber"]] = {k: row[k] for k in merge_cols}

        empty_meta = {k: "" for k in merge_cols}

        # Stream CSV row-by-row, append metadata, write output
        t0 = time.time()
        n = 0
        with open(csv_local) as fin, open(output_file, "w", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames + merge_cols)
            writer.writeheader()
            for row in reader:
                row.update(lookup.get(row["ImageNumber"], empty_meta))
                writer.writerow(row)
                n += 1

    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"Merged {n:,} rows x {len(merge_cols)} metadata columns in {elapsed:.1f}s")
    print(f"Saved to: {output_file} ({size_mb:.1f} MB)")

    return output_file
