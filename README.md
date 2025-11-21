# alethiotx

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Alethio Therapeutics Python Toolkit** - A growing collection of open-source computational tools used by Alethio Therapeutics.

## Overview

`alethiotx` is a modular Python package providing specialized tools for therapeutic research and drug discovery. Currently, the package features the **Artemis** module for drug target prioritization using public knowledge graphs. Additional modules and capabilities will be added in future releases.

### Current Modules

#### Artemis Module (`alethiotx.artemis`)

The Artemis module enables accessible and scalable drug prioritization by integrating clinical trial data, drug databases (DrugBank), pathway information, and machine learning models. It leverages public knowledge graphs to prioritize therapeutic targets across multiple disease areas.

### Artemis Module Features

- **Clinical Trials**: Query and analyze clinical trials data from ClinicalTrials.gov
- **DrugBank**: Match clinical interventions with DrugBank drug information and targets
- **Pathway Genes**: Retrieve and analyze pathway genes using GeneShot API
- **Target Scoring**: Calculate clinical target scores for drug targets based on trial phases and approvals
- **Machine Learning Pipeline**: Built-in cross-validation and for target prediction
- **Multi-Disease Support**: Pre-configured for breast, lung, prostate, melanoma, bowel cancer, diabetes, and cardiovascular disease

### Future Modules

Additional modules for various aspects of drug discovery and therapeutic research are planned for future releases. Stay tuned!

## Installation

```bash
pip install alethiotx
```

## Quick Start

> **Note:** The examples below demonstrate the **Artemis** module functionality. As new modules are added to the package, they will have their own usage examples.

### 1. Retrieve Clinical Trials Data

```python
from alethiotx.artemis import trials, drugbank, drugscores

# Query clinical trials for a specific indication
breast_trials = get_clinical_trials(search='Breast Cancer', last_6_years=True)

# Match trials with DrugBank to get target information
db_data = drugbank(breast_trials)

# Calculate clinical development scores
scores = get_clinical_scores(db_data, include_approved=True)
print(scores.head())
```

### 2. Load Pre-computed Clinical Scores

```python
from alethiotx.artemis import load_clinical_scores

# Load clinical scores for multiple diseases
breast, lung, prostate, melanoma, bowel, diabetes, cardio = load_clinical_scores(date='2025-11-11')
```

### 3. Pathway Gene Analysis

```python
from alethiotx.artemis import get_pathway_genes load_pathway_genes

# Query GeneShot for disease-associated genes
aml_genes = get_pathway_genes("acute myeloid leukemia")
print(aml_genes.loc["FLT3", ["gene_count", "rank"]])

# Get top pathway genes for diseases
breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardio_pg = load_pathway_genes(n=100)
```

### 4. Machine Learning Pipeline

```python
from alethiotx.artemis import pre_model, cv_pipeline, roc_curve
import pandas as pd

# Prepare your knowledge graph features (X) and clinical scores (y)
result = pre_model(X, y, pathway_genes=pathway_genes, bins=3)

# Run cross-validation pipeline
scores = cv_pipeline(X, y, n_iterations=10, scoring='roc_auc')
print(f"Mean AUC: {sum(scores)/len(scores):.3f}")

# Generate ROC curves
mean_auc = roc_curve(result['X'], result['y_binary'], n_splits=5, classifier='rf')
```

### 5. Visualize Gene Overlaps with UpSet Plots

```python
from alethiotx.artemis import prepare_upset, create_upset_plot

# Load clinical scores or pathway genes for multiple diseases
breast, lung, prostate, melanoma, bowel, diabetes, cardio = load_clinical_scores()

# Prepare data for UpSet plot (mode='ct' for clinical targets)
upset_data = prepare_upset(breast, lung, prostate, melanoma, bowel, diabetes, cardio, mode='ct')

# Create and display the UpSet plot
plot = create_upset_plot(upset_data, min_subset_size=5)
plot.plot()

# For pathway genes, use mode='pg'
breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardio_pg = load_pathway_genes(n=100)
upset_data_pg = prepare_upset(breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardio_pg, mode='pg')
plot_pg = create_upset_plot(upset_data_pg, min_subset_size=10)
plot_pg.plot()
```

## Supported Disease Indications (Artemis Module)

The Artemis module includes built-in support for:

- **Myeloproliferative Neoplasm (MPN)**
- **Breast Cancer**
- **Lung Cancer**
- **Prostate Cancer**
- **Bowel Cancer (Colorectal)**
- **Melanoma**
- **Diabetes Mellitus Type 2**
- **Cardiovascular Disease**

## Artemis Module API Reference

### Data Loading & Processing

- `get_clinical_trials()` - Retrieve clinical trials from ClinicalTrials.gov
- `drugbank()` - Match trials with DrugBank drug/target data
- `get_clinical_scores()` - Calculate per-target clinical development scores
- `load_clinical_scores()` - Load pre-computed clinical scores from S3
- `get_pathway_genes()` - Query Ma'ayan Lab's GeneShot API for gene associations
- `load_pathway_genes()` - Retrieve pathway gene data

### Data Preparation

- `get_all_targets()` - Extract unique target genes from score lists
- `cut_clinical_scores()` - Filter scores by threshold
- `find_overlapping_genes()` - Identify genes present in multiple datasets
- `uniquify_clinical_scores()` - Remove overlapping genes from clinical scores
- `uniquify_pathway_genes()` - Remove overlapping genes from pathway lists

### Machine Learning

- `pre_model()` - Prepare datasets for ML model training
- `cv_pipeline()` - Cross-validation pipeline with customizable classifiers

### Visualization

- `prepare_upset()` - Prepare disease-related data for UpSet plot visualization
- `create_upset_plot()` - Create UpSet plots for visualizing gene set intersections across diseases

## Data Storage (Artemis Module)

The Artemis module uses AWS S3 for storing pre-computed data:

```
s3://alethiotx-artemis/data/
├── clinical_targets/{date}/{disease}.csv
├── pathway_genes/{date}/{disease}.csv
└── drugbank/{date}/webdata.csv
```

## Requirements

- Python >= 3.9
- requests
- scikit-learn
- pandas
- numpy
- matplotlib
- setuptools
- fsspec
- s3fs
- upsetplot

## Documentation

For detailed API documentation and examples, see the module docstrings:

```python
from alethiotx import artemis
help(artemis.trials)
help(artemis.pre_model)
```

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
Email: vlad.kiselev@alethiomics.com

## Links

- **Homepage**: https://github.com/alethiotx/pypi
- **Issues**: https://github.com/alethiotx/pypi/issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Current Focus:** Artemis - Enabling accessible and scalable drug target discovery through public knowledge graphs.  
**Coming Soon:** Additional modules for expanded drug discovery capabilities. 
