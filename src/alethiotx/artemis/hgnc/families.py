"""
Export complete HGNC gene family data.

This script downloads HGNC gene group data and exports the complete mapping
of all genes to their gene families.
"""

import pandas as pd
from pathlib import Path

def download(
    gene_has_family_url="https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/gene_has_family.csv",
    family_url="https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/family.csv",
    hgnc_complete_url="https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
):
    """
    Download HGNC (HUGO Gene Nomenclature Committee) data tables.
    This function downloads three datasets from the HGNC public repository:
    gene-family associations, family definitions, and the complete gene set.
    :param gene_has_family_url: URL to the gene_has_family CSV file containing
        gene-to-family associations, defaults to HGNC public storage URL
    :type gene_has_family_url: str, optional
    :param family_url: URL to the family CSV file containing family definitions,
        defaults to HGNC public storage URL
    :type family_url: str, optional
    :param hgnc_complete_url: URL to the complete HGNC dataset in TSV format,
        defaults to HGNC public storage URL
    :type hgnc_complete_url: str, optional
    :return: A tuple containing three pandas DataFrames:
    - gene_has_family: DataFrame with gene-to-family associations
    - family: DataFrame with family information
    - hgnc_data: DataFrame with complete HGNC gene set
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    :raises requests.exceptions.RequestException: If download fails
    :raises pandas.errors.ParserError: If CSV/TSV parsing fails
    Example
    -------
    >>> gene_has_family, family, hgnc_data = download()
    Downloading HGNC gene_has_family table...
    Downloading HGNC family table...
    Downloading HGNC complete set (this may take a moment)...
    """
    print("Downloading HGNC gene_has_family table...")
    gene_has_family = pd.read_csv(gene_has_family_url)
    
    print("Downloading HGNC family table...")
    family = pd.read_csv(family_url)
    
    print("Downloading HGNC complete set (this may take a moment)...")
    hgnc_data = pd.read_csv(hgnc_complete_url, sep='\t', low_memory=False)
    
    return gene_has_family, family, hgnc_data

def process(gene_has_family, family, hgnc_data):
    """
    Process HGNC gene family data and create detailed mappings.
    This function processes HGNC (HUGO Gene Nomenclature Committee) data to analyze
    gene family relationships. It extracts numeric HGNC IDs, counts genes per family,
    and creates comprehensive gene-to-family mappings with associated metadata.
    :param gene_has_family: DataFrame containing gene-to-family relationships with columns
                            'hgnc_id' and 'family_id'
    :type gene_has_family: pandas.DataFrame
    :param family: DataFrame containing family information with columns 'id', 'abbreviation',
                   and 'name'
    :type family: pandas.DataFrame
    :param hgnc_data: DataFrame containing HGNC gene information with columns 'hgnc_id',
                      'symbol', 'name', 'locus_group', and 'locus_type'
    :type hgnc_data: pandas.DataFrame
    :return: A tuple containing:
    - ``family_summary``: DataFrame with family information and gene counts, sorted by
    total_genes_in_family in descending order
    - ``gene_family_mapping``: DataFrame with detailed gene-to-family mappings including
    gene symbols, names, locus information, and family details
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    :raises ValueError: If HGNC IDs cannot be converted to numeric format
    .. note::
       The function expects HGNC IDs in the format "HGNC:12345" in the hgnc_data DataFrame.
       These are converted to numeric values for processing.
    .. note::
       Progress information is printed to stdout during processing.
    """
    print("\nProcessing HGNC data...")
    
    # Extract numeric HGNC ID from "HGNC:12345" format
    hgnc_data['hgnc_id_numeric'] = hgnc_data['hgnc_id'].str.replace('HGNC:', '').astype(int)
    
    # Count genes per family
    gene_counts = gene_has_family.groupby('family_id').size().reset_index(name='total_genes_in_family')
    
    # Add gene counts to family table
    family_summary = family.merge(gene_counts, left_on='id', right_on='family_id', how='left')
    family_summary['total_genes_in_family'] = family_summary['total_genes_in_family'].fillna(0).astype(int)
    
    # Sort by number of genes
    family_summary = family_summary.sort_values('total_genes_in_family', ascending=False)
    
    print(f"  Found {len(family_summary)} gene families")
    print(f"  Total genes with family assignments: {len(gene_has_family)}")
    
    # Create detailed gene-to-family mapping
    gene_family_mapping = gene_has_family.merge(
        hgnc_data[['hgnc_id_numeric', 'symbol', 'name', 'locus_group', 'locus_type']], 
        left_on='hgnc_id',
        right_on='hgnc_id_numeric',
        how='left'
    ).merge(
        family[['id', 'abbreviation', 'name']], 
        left_on='family_id', 
        right_on='id',
        how='left',
        suffixes=('_gene', '_family')
    )
    
    print(f"  Created mapping for {len(gene_family_mapping)} gene-family pairs")
    
    return family_summary, gene_family_mapping

if __name__ == '__main__':
    """Main execution function."""
    print("=" * 80)
    print("HGNC Gene Family Data Export")
    print("=" * 80)
    
    # Download HGNC data
    gene_has_family, family, hgnc_data = download()
    
    # Process gene family data
    family_summary, gene_family_mapping = process(
        gene_has_family, family, hgnc_data
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total gene families: {len(family_summary)}")
    print(f"  Total gene-family mappings: {len(gene_family_mapping)}")
    print(f"  Unique genes with family assignments: {gene_family_mapping['symbol'].nunique()}")
    print(f"\n  Top 10 largest gene families:")
    print(family_summary[['abbreviation', 'name', 'total_genes_in_family']].head(10).to_string(index=False))
    print("=" * 80)
