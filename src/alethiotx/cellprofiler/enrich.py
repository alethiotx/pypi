"""
Enrich CellProfiler CSVs with TIF metadata from Image_enriched.csv.

Downloads files from S3, streams the merge row-by-row (~12 MB memory),
and writes the enriched CSV locally.
"""

import csv
import subprocess
import tempfile
import io
import pandas as pd
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
    :param output_dir: Directory to write the enriched CSV. If ``None`` (the
        default), the merged data is NOT written to disk and only a
        `pandas.DataFrame` is returned. To write the merged CSV, pass a
        directory path.
    :type output_dir: str or Path or None
    :return: A pandas `DataFrame` containing the enriched CSV data.
    :rtype: pandas.DataFrame

    **Examples**

    Add all metadata columns to ``Cell.csv`` and receive a pandas ``DataFrame``::

        >>> from alethiotx.cellprofiler import add_metadata
        >>> df = add_metadata("s3://example-bucket/my-experiment/", "Cell.csv")
        >>> print(df.shape)
        (1000, 250)

    Add specific columns only::

        >>> df = add_metadata(
        ...     "s3://example-bucket/my-experiment/",
        ...     "Nucleus.csv",
        ...     columns=["PlateID", "Well", "Site", "Z_Step"],
        ... )

    Write to a specific directory (also writes the merged CSV to disk)::

        >>> df = add_metadata(
        ...     "s3://example-bucket/my-experiment/",
        ...     "Cell.csv",
        ...     output_dir="~/Downloads",
        ... )
    """
    s3_base = s3_path.rstrip("/")
    stem = Path(csv_name).stem

    # Only prepare an output path if the user provided `output_dir`.
    if output_dir is None:
        output_path = None
    else:
        output_dir = Path(output_dir).expanduser()
        output_path = output_dir / f"{stem}_enriched.csv"

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
        with open(csv_local) as fin:
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames + merge_cols

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row.update(lookup.get(row["ImageNumber"], empty_meta))
                writer.writerow(row)
                n += 1

            csv_text = buf.getvalue()
            buf.close()

            # Optionally write the merged CSV to disk if output_dir was provided
            if output_path is not None:
                with open(output_path, "w", newline="") as fout:
                    fout.write(csv_text)

    elapsed = time.time() - t0
    print(f"Merged {n:,} rows x {len(merge_cols)} metadata columns in {elapsed:.1f}s")

    # Convert merged CSV text into a pandas DataFrame and return
    df = pd.read_csv(io.StringIO(csv_text))

    if output_path is not None:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved to: {output_path} ({size_mb:.1f} MB)")

    return df
