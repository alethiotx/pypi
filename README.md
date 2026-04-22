# alethiotx

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Alethio Therapeutics Python Toolkit** - A growing collection of open-source computational tools used by Alethio Therapeutics.

## Overview

`alethiotx` is a modular Python package providing specialized tools for therapeutic research and drug discovery. Currently, the package features the **Artemis** module for drug target prioritization using public knowledge graphs. Additional modules and capabilities will be added in future releases.

### Current Modules

#### Artemis Module (`alethiotx.artemis`)

The Artemis module enables accessible and scalable drug target prioritization by integrating drug molecule and target data from ChEMBL (including clinical trial phases and approvals), MeSH disease hierarchies, HGNC gene families, pathway information from GeneShot, and machine learning pipelines. It leverages public knowledge graphs to prioritize therapeutic targets across multiple disease areas.

##### Artemis Module Features

- **ChEMBL Integration**: Query and process ChEMBL bioactive molecule database with clinical trial information and automatic parent molecule normalization
- **MeSH Hierarchy**: Retrieve MeSH disease trees and descendants for comprehensive disease coverage
- **HGNC Gene Families**: Download and analyze gene family data to identify and filter over-represented families
- **Clinical Scoring**: Calculate clinical validation scores for drug targets based on trial phases, approvals, and family representation
- **Pathway Genes**: Retrieve and analyze disease-associated genes using Ma'ayan Lab's GeneShot API
- **Machine Learning Pipeline**: Built-in cross-validation with configurable classifiers for target prediction
- **UpSet Plots**: Visualize gene set intersections across multiple diseases
- **Multi-Disease Support**: Pre-configured for breast, lung, prostate, melanoma, bowel cancer, diabetes, and cardiovascular disease

### CellProfiler Module (`alethiotx.cellprofiler`)

The CellProfiler module provides small, memory-efficient utilities for post-processing CellProfiler pipeline outputs. It offers helpers to add TIF metadata (from an Image_enriched.csv produced by the pipeline) to large CellProfiler CSVs without loading everything into memory.

## Installation

```bash
pip install alethiotx
```

## Quick Start

### 1. Query ChEMBL and Compute Clinical Scores

```python
from alethiotx.artemis.chembl import molecules
from alethiotx.artemis.clinical import compute

# Query ChEMBL for parent molecules with clinical trial data
chembl_data = molecules(version='36', top_n_activities=1)

# Compute clinical validation scores for specific diseases
results = compute(
    mesh_headings=['Breast Neoplasms', 'Lung Neoplasms'],
    chembl_version='36',
    trials_only_last_n_years=6,
    filter_families=True
)

# Access results for each disease
breast_targets = results['Breast Neoplasms']
print(breast_targets.head())
```

### 2. Load Pre-computed Clinical Scores

```python
from alethiotx.artemis.clinical import load

# Load pre-computed clinical scores for multiple diseases from S3
breast, lung, prostate, melanoma, bowel, diabetes, cardio = load(date='2025-12-08')
```

### 3. Pathway Gene Analysis

```python
from alethiotx.artemis.pathway import get, load

# Query GeneShot API for disease-associated genes
aml_genes = get("acute myeloid leukemia", rif='generif')
print(aml_genes.loc["FLT3", ["gene_count", "rank"]])

# Load pre-computed pathway genes for multiple diseases
breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardio_pg = load(date='2025-11-11', n=100)
```

### 4. Machine Learning Pipeline

```python
from alethiotx.artemis.cv import prepare, run
import pandas as pd

# Prepare your knowledge graph features (X) and clinical scores (y)
result = prepare(
    X, 
    y, 
    pathway_genes=pathway_genes, 
    known_targets=known_targets,
    bins=3,
    rand_seed=12345
)

# Run cross-validation pipeline
scores = run(
    result['X'], 
    result['y_binary'], 
    n_splits=5, 
    n_iterations=10, 
    classifier='rf',
    scoring='roc_auc'
)
print(f"Mean AUC: {sum(scores)/len(scores):.3f}")
```

### 5. Visualize Gene Overlaps with UpSet Plots

```python
from alethiotx.artemis.upset import prepare, create
from alethiotx.artemis.clinical import load
from alethiotx.artemis.pathway import load as load_pathway

# Load clinical scores for multiple diseases
breast, lung, prostate, melanoma, bowel, diabetes, cardio = load(date='2025-12-08')

# Prepare data for UpSet plot (mode='ct' for clinical targets)
upset_data = prepare(breast, lung, prostate, melanoma, bowel, diabetes, cardio, mode='ct')

# Create and display the UpSet plot
plot = create(upset_data, min_subset_size=5)
plot.plot()

# For pathway genes, use mode='pg'
breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardio_pg = load_pathway(date='2025-11-11', n=100)
upset_data_pg = prepare(breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardio_pg, mode='pg')
plot_pg = create(upset_data_pg, min_subset_size=10)
plot_pg.plot()
```

### 6. Merge CellProfiler output with TIFF metadata (CellProfiler Module)

The CellProfiler module provides small, memory-efficient utilities for post-processing CellProfiler pipeline outputs. It offers helpers to add TIF metadata (from an Image_enriched.csv produced by the pipeline) to large CellProfiler CSVs without loading everything into memory.

- `add_metadata(s3_path, csv_name, columns=None, output_dir=None)` — Add metadata columns from `Image_enriched.csv` to a target CSV and write `<stem>_enriched.csv` locally.
- `list_metadata_columns(s3_path)` — List available metadata columns present in `Image_enriched.csv`.

Quick example:

```python
from alethiotx.cellprofiler import add_metadata, list_metadata_columns

# List available metadata columns
cols = list_metadata_columns("s3://example-bucket/my-experiment/")

# Add all metadata columns to Cell.csv — returns a pandas DataFrame
df = add_metadata("s3://example-bucket/my-experiment/", "Cell.csv")

# To also write the merged CSV to disk, pass `output_dir`
# df = add_metadata("s3://example-bucket/my-experiment/", "Cell.csv", output_dir="~/Downloads")
```

## Supported Disease Indications (Artemis Module)

The Artemis module includes built-in pre-computed data for:

- **Breast Cancer** (Breast Neoplasms)
- **Lung Cancer** (Lung Neoplasms)
- **Prostate Cancer** (Prostatic Neoplasms)
- **Melanoma** (Skin Neoplasms)
- **Bowel Cancer** (Intestinal Neoplasms)
- **Diabetes Mellitus Type 2**
- **Cardiovascular Disease**

The module supports querying any disease with MeSH headings via the `compute()` function.

## Data Storage (Artemis Module)

The Artemis module uses AWS S3 for storing pre-computed data:

```
s3://alethiotx-artemis/data/
├── clinical_scores/{date}/{disease}.csv
├── pathway_genes/{date}/{disease}.csv
├── chembl/{version}/molecules.csv
└── mesh/d{year}.pkl
```

## Requirements

- Python >= 3.9
- requests
- scikit-learn
- pandas
- numpy
- setuptools
- fsspec
- s3fs
- upsetplot
- chembl-downloader

## Citation

If you use the Artemis module in your research, please cite:

```
Artemis: public knowledge graphs enable accessible and scalable drug target discovery
Vladimir Kiselev, Alethio Therapeutics
```

For other modules, citation information will be provided as they are released.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Vladimir Kiselev**  
Email: vlad.kiselev@alethiotx.com

## Links

- **Homepage**: https://github.com/alethiotx/pypi
- **Issues**: https://github.com/alethiotx/pypi/issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Public knowledge graph providers (Hetionet, BioKG, OpenBioLink, PrimeKG)
- PyKEEN, scikit-learn, and Nextflow communities
- ChEMBL and MeSH data sources
- Portions of this codebase were generated, refactored, and/or cleaned using GitHub Copilot (Claude Sonnet 4.5). The authors reviewed, modified, and validated all AI-assisted code. Responsibility for the correctness, performance, and reproducibility of the code rests entirely with the authors. 

---

**Current Focus:** Artemis - Enabling accessible and scalable drug target discovery through public knowledge graphs.  
**Coming Soon:** Additional modules for expanded drug discovery capabilities. 
