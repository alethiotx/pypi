"""
Enrich CellProfiler CSVs with TIF metadata from Image_enriched.csv.

Downloads files from S3 (or reads locally), joins by FileName columns,
and returns a pandas DataFrame.  Optionally writes to disk.
"""

import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd


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


def list_metadata_columns(s3_path='.'):
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
    base = s3_path.rstrip("/")

    # Support local directories as well as S3 URIs
    if base.startswith("s3://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            ie_local = Path(tmpdir) / "Image_enriched.csv"
            _s3_download(f"{base}/Image_enriched.csv", ie_local)
            with open(ie_local) as f:
                header = csv.DictReader(f).fieldnames
    else:
        ie_local = Path(base) / "Image_enriched.csv"
        with open(ie_local) as f:
            header = csv.DictReader(f).fieldnames

    return [c for c in header if c != "ImageNumber"]


def add_metadata(s3_path='.', csv_name='Cells.csv', columns=None, output_dir=None):
    """Add TIF metadata to a CellProfiler CSV.

    Downloads ``csv_name`` and ``Image_enriched.csv`` from the S3 output
    directory (or reads from a local directory), joins them by
    ``FileName_*`` columns, and returns a pandas DataFrame.

    ``ImageNumber`` cannot be used as a join key because CellProfiler
    parallel tasks each start numbering from 1, producing duplicates
    after ``merge_csv.py`` concatenates them.

    :param s3_path: S3 URI or local path of the CellProfiler output
        directory (e.g. ``s3://example-bucket/my-experiment/`` or ``.``).
    :type s3_path: str
    :param csv_name: Name of the CSV to enrich
        (e.g. ``Cells.csv``, ``Nucleus.csv``, ``Cytoplasm.csv``)
    :type csv_name: str
    :param columns: Specific metadata columns to add from
        Image_enriched.csv.  If ``None``, all metadata columns are added
        (excluding ``ImageNumber``, ``FileName_*``, and ``PathName_*``
        which already exist in the target CSV).
    :type columns: list[str] or None
    :param output_dir: Directory to write ``<stem>_enriched.csv``.
        If ``None`` (the default), nothing is written to disk.
    :type output_dir: str or Path or None
    :return: The enriched data.
    :rtype: pandas.DataFrame

    **Examples**

    Return a DataFrame without writing to disk::

        >>> from alethiotx.cellprofiler import add_metadata
        >>> df = add_metadata("s3://example-bucket/my-experiment/", "Cells.csv")
        >>> print(df.shape)
        (1000, 250)

    Add specific columns only::

        >>> df = add_metadata(
        ...     "s3://example-bucket/my-experiment/",
        ...     "Nucleus.csv",
        ...     columns=["PlateID", "Well", "Site", "Z_Step"],
        ... )

    Also write the merged CSV to disk::

        >>> df = add_metadata(
        ...     "s3://example-bucket/my-experiment/",
        ...     "Cells.csv",
        ...     output_dir="~/Downloads",
        ... )
    """
    s3_base = s3_path.rstrip("/")
    stem = Path(csv_name).stem

    if output_dir is not None:
        output_dir = Path(output_dir).expanduser()
        output_path = output_dir / f"{stem}_enriched.csv"
    else:
        output_path = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ie_local = tmpdir / "Image_enriched.csv"
        csv_local = tmpdir / csv_name

        # Download from S3 or resolve local paths
        if s3_base.startswith("s3://"):
            print(f"Downloading {s3_base}/Image_enriched.csv ...")
            _s3_download(f"{s3_base}/Image_enriched.csv", ie_local)
            print(f"Downloading {s3_base}/{csv_name} ...")
            _s3_download(f"{s3_base}/{csv_name}", csv_local)
        else:
            local_base = Path(s3_base)
            ie_local = local_base / "Image_enriched.csv"
            csv_local = local_base / csv_name

        # Read Image_enriched header
        with open(ie_local) as f:
            ie_header = csv.DictReader(f).fieldnames

        # FileName columns are the join key (ImageNumber is unreliable
        # because parallel CellProfiler tasks each start from 1)
        fname_cols = [c for c in ie_header if c.startswith("FileName_")]
        if not fname_cols:
            raise ValueError(
                "Image_enriched.csv must contain FileName_* columns for joining.\n"
                f"Available columns: {ie_header}"
            )

        # Determine which metadata columns to add
        if columns is not None:
            missing = [c for c in columns if c not in ie_header]
            if missing:
                raise ValueError(
                    f"Columns not found in Image_enriched.csv: {missing}\n"
                    f"Use list_metadata_columns() to see available columns."
                )
            merge_cols = list(columns)
        else:
            # Exclude columns that already exist in the target CSV
            exclude = {"ImageNumber"} | {
                c for c in ie_header
                if c.startswith(("FileName_", "PathName_"))
            }
            merge_cols = [c for c in ie_header if c not in exclude]

        # Build lookup: filename tuple → metadata dict
        lookup = {}
        with open(ie_local) as f:
            for row in csv.DictReader(f):
                key = tuple(row.get(k, "") for k in fname_cols)
                lookup[key] = {k: row[k] for k in merge_cols}

        empty_meta = {k: "" for k in merge_cols}

        # Stream CSV row-by-row: write to output (or temp) file
        t0 = time.time()
        n = 0

        # Write to final destination or a temp file
        write_path = output_path or Path(tmpdir) / f"{stem}_enriched.csv"

        with open(csv_local) as fin, \
             open(write_path, "w", newline="") as fout:
            reader = csv.DictReader(fin)

            # Verify target CSV has the same FileName columns
            target_fname_cols = {c for c in reader.fieldnames
                                 if c.startswith("FileName_")}
            missing_fnames = set(fname_cols) - target_fname_cols
            if missing_fnames:
                raise ValueError(
                    f"{csv_name} is missing FileName columns needed for "
                    f"join: {sorted(missing_fnames)}"
                )

            writer = csv.DictWriter(
                fout, fieldnames=reader.fieldnames + merge_cols,
            )
            writer.writeheader()

            for row in reader:
                key = tuple(row.get(k, "") for k in fname_cols)
                row.update(lookup.get(key, empty_meta))
                writer.writerow(row)
                n += 1

        elapsed = time.time() - t0
        print(f"Merged {n:,} rows x {len(merge_cols)} metadata columns "
              f"in {elapsed:.1f}s")

        # Read back as DataFrame
        df = pd.read_csv(write_path)

    if output_path is not None:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved to: {output_path} ({size_mb:.1f} MB)")

    return df
