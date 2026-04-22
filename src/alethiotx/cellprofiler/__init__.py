"""
**CellProfiler**: tools for post-processing CellProfiler pipeline outputs.

Provides utilities for enriching CellProfiler measurement CSVs with
TIF metadata from ImageXpress microscope images.

Usage
-----

::

    from alethiotx.cellprofiler import add_metadata
    add_metadata("s3://altmx-cellprofiler-results/my-experiment/", "Cell.csv")
"""

from .enrich import add_metadata, list_metadata_columns

__all__ = ["add_metadata", "list_metadata_columns"]
