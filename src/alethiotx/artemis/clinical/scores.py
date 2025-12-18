from pandas import DataFrame, merge, read_csv, concat as pd_concat, notna
from datetime import date
from ..mesh.get import descendants as mesh_descendants
from ..hgnc.families import download as hgnc_download, process as hgnc_process
from ..utils import find_overlapping_genes

def lookup_drug_family_representation(chembl: DataFrame) -> DataFrame:
   """
   Create a comprehensive lookup table mapping drug-disease-family combinations to gene family representation.
   
   This function analyzes drug-target relationships from ChEMBL clinical trial data to identify potential
   bias from drugs that target many genes within the same family. For example, multi-kinase inhibitors
   targeting 10 out of 15 kinases in a family would represent 67% of that family, potentially inflating
   the importance of kinase targets in clinical scoring.
   
   The lookup table enables downstream filtering of over-represented families to produce unbiased
   target prioritization scores.
   
   **Processing Workflow:**
   
   1. **Download HGNC Data**: Retrieves complete gene family definitions from HGNC
   2. **Extract Relationships**: Identifies all unique drug-target-disease combinations from ChEMBL
   3. **Family Mapping**: Maps each drug target to its HGNC gene family (if assigned)
   4. **Calculate Representation**: For each drug-disease-family combination, computes:
      
      - Number of family genes targeted by the drug
      - Total genes in the family
      - Percentage representation (targets/total * 100)
   
   5. **Return Lookup**: Creates sorted lookup table for efficient querying
   
   **Why This Matters:**
   
   When a drug targets multiple members of the same gene family (e.g., multi-kinase inhibitors,
   pan-HDAC inhibitors), each family member contributes to clinical scores independently. This
   artificially inflates the importance of that family in target prioritization. By quantifying
   family representation, we can identify and filter these cases to prevent bias.
   
   :param chembl: ChEMBL molecule data containing drug-target-indication relationships.
                  Must include columns: ``chembl_id``, ``target_gene_name``, ``mesh_heading``
   :type chembl: DataFrame
   :return: Lookup table with one row per drug-disease-family combination. 
   
   **Columns:**
      
      - ``drug_chembl_id`` (str): ChEMBL identifier for the drug (e.g., 'CHEMBL123')
      - ``mesh_heading`` (str): MeSH disease term (e.g., 'Lung Neoplasms')
      - ``family_id`` (int): HGNC gene family identifier
      - ``family_abbreviation`` (str): Short family name (e.g., 'RTK' for receptor tyrosine kinases)
      - ``family_name`` (str): Full descriptive family name (e.g., 'Receptor tyrosine kinases')
      - ``total_genes_in_family`` (int): Total number of genes in this HGNC family
      - ``targets_in_family`` (int): Number of family genes targeted by this drug for this disease
      - ``target_genes_in_family`` (str): Comma-separated list of targeted gene symbols
      - ``pct_represented`` (float): Percentage of family genes targeted (0.0-100.0)
      
      Sorted by: drug_chembl_id, mesh_heading, then pct_represented (descending)
      
   :rtype: DataFrame
   
   .. note::
      Only includes targets with HGNC family assignments. Targets without family data are
      excluded from the lookup table (they pass through unfiltered in downstream processing).
   
   .. note::
      This function downloads fresh HGNC data on each call (~10-30 seconds). For production
      workflows processing multiple diseases, call this function once and reuse the lookup table.
   
   .. warning::
      Requires internet connection to download HGNC gene family data from public repositories.
      Ensure network access to HGNC/Google Cloud Storage endpoints.
   
   **Examples**
   
   Basic usage::
   
      >>> from alethiotx.artemis.chembl import molecules
      >>> chembl_data = molecules(version='36')
      >>> lookup = lookup_drug_family_representation(chembl_data)
      >>> print(f"Created lookup with {len(lookup)} drug-disease-family combinations")
   
   Find drugs over-representing kinase families::
   
      >>> kinase_overrep = lookup[
      ...     (lookup['family_name'].str.contains('kinase', case=False)) &
      ...     (lookup['pct_represented'] > 50)
      ... ]
      >>> print(kinase_overrep[['drug_chembl_id', 'family_name', 'pct_represented']].head())
   
   Identify drugs with many targets in small families::
   
      >>> small_families = lookup[
      ...     (lookup['total_genes_in_family'] < 10) &
      ...     (lookup['targets_in_family'] >= 4)
      ... ]
      >>> for _, row in small_families.iterrows():
      ...     print(f"{row['drug_chembl_id']}: {row['targets_in_family']}/{row['total_genes_in_family']} "
      ...           f"of {row['family_name']}")
   
   .. seealso::
      - :func:`filter_overrepresented_families`: Uses this lookup table to filter biased targets
      - :func:`compute`: Integrates this lookup for unbiased clinical score calculation
   """
   # Download HGNC gene family data
   gene_has_family, family, hgnc_data = hgnc_download()
   family_summary, gene_family_mapping = hgnc_process(gene_has_family, family, hgnc_data)
   
   # Extract unique drug-target-disease combinations from ChEMBL
   # Use vectorized operations for better performance
   valid_rows = chembl[['chembl_id', 'target_gene_name', 'mesh_heading']].copy()
   valid_rows = valid_rows[
      notna(valid_rows['target_gene_name']) & notna(valid_rows['mesh_heading'])
   ]
   valid_rows.rename(columns={'chembl_id': 'drug_chembl_id'}, inplace=True)
   targets_df = valid_rows.drop_duplicates()
   
   print(f"\nTotal drug-target-disease combinations: {len(targets_df)}")
   print(f"Unique targets: {targets_df['target_gene_name'].nunique()}")
   print(f"Unique drugs: {targets_df['drug_chembl_id'].nunique()}")
   
   # Map targets to their HGNC gene families
   targets_with_families = targets_df.merge(
      gene_family_mapping[['symbol', 'family_id', 'abbreviation', 'name_family']],
      left_on='target_gene_name',
      right_on='symbol',
      how='left'
   )
   
   # Report mapping success rate
   with_family = targets_with_families['family_id'].notna().sum()
   without_family = targets_with_families['family_id'].isna().sum()
   print(f"  Targets with family: {with_family}, without family: {without_family}")
   
   # Keep only targets with family assignments for representation calculation
   targets_with_families = targets_with_families[
      targets_with_families['family_id'].notna()
   ].copy()
   
   # Calculate family representation metrics
   # Group by drug-disease-family to count targets per family
   family_counts = targets_with_families.groupby(
      ['drug_chembl_id', 'mesh_heading', 'family_id'],
      as_index=False
   ).agg(
      targets_in_family=('target_gene_name', 'nunique'),
      target_genes_in_family=('symbol', lambda x: ', '.join(sorted(x.unique())))
   )
   
   # Add family metadata (name, abbreviation, total gene count)
   family_counts = family_counts.merge(
      family_summary[['id', 'abbreviation', 'name', 'total_genes_in_family']],
      left_on='family_id',
      right_on='id',
      how='left'
   )
   
   # Calculate percentage representation: (targets in family / total genes in family) * 100
   family_counts['pct_represented'] = (
      100 * family_counts['targets_in_family'] / family_counts['total_genes_in_family']
   ).round(1)
   
   # Select and organize final columns
   result = family_counts[[
      'drug_chembl_id',
      'mesh_heading',
      'family_id',
      'abbreviation',
      'name',
      'total_genes_in_family',
      'targets_in_family',
      'target_genes_in_family',
      'pct_represented'
   ]].copy()
   
   # Rename for clarity
   result.rename(columns={
      'abbreviation': 'family_abbreviation',
      'name': 'family_name'
   }, inplace=True)
   
   # Sort by drug, disease, then highest representation first
   result.sort_values(
      ['drug_chembl_id', 'mesh_heading', 'pct_represented'],
      ascending=[True, True, False],
      inplace=True
   )
   
   return result

def filter_overrepresented_families(
   targets_df: DataFrame,
   drug_chembl_id: str,
   mesh_heading: str,
   lookup_table: DataFrame,
   pct_threshold: float = 40.0,
   small_family_threshold: int = 5,
   min_targets_threshold: int = 3
) -> DataFrame:
   """
   Filter over-represented gene families from drug targets to eliminate family-based bias in clinical scoring.
   
   Multi-target drugs (e.g., multi-kinase inhibitors like sunitinib) often target multiple members of the
   same gene family. When computing clinical validation scores, each family member contributes independently,
   artificially inflating the perceived importance of that family. This function identifies over-represented
   families and keeps only one representative target per family to produce unbiased scores.
   
   **Why Filtering is Needed:**
   
   Consider sunitinib, which targets 8 receptor tyrosine kinases (RTKs). Without filtering, all 8 RTKs
   contribute to clinical scores, making RTKs appear highly validated. With filtering, we keep only 1
   representative RTK (e.g., FLT3 alphabetically), preventing the family from dominating results.
   
   **Filtering Logic:**
   
   A gene family is considered "over-represented" and filtered if **ALL** of these conditions are met:
   
   1. **Sufficient targets**: Drug targets > ``min_targets_threshold`` genes in the family (default: >3)
   2. **Size-adjusted criteria** (either condition triggers filtering):
      
      a. **Large family with high %**: Family size ≥ ``small_family_threshold`` (default: ≥5)
         AND representation > ``pct_threshold``% (default: >40%)
      b. **Absolute threshold**: Drug targets ≥ 4 genes in family (regardless of family size)
   
   **Exemptions from Filtering:**
   
   - **Small families** (<5 genes by default): Never filtered to preserve biological specificity
     (e.g., the 4-gene HDAC class IIA family)
   - **Low target count** (≤3 targets by default): Insufficient redundancy to justify filtering
   - **Targets without family assignments**: Pass through unfiltered
   
   **Selection of Representative:**
   
   For each over-represented family, the first target alphabetically is kept as the representative.
   All other family members are excluded from the targets DataFrame.
   
   :param targets_df: DataFrame of drug targets for a specific drug-disease combination.
                      Must contain a ``target_gene_name`` column with gene symbols.
   :type targets_df: DataFrame
   :param drug_chembl_id: ChEMBL identifier for the drug being analyzed (e.g., 'CHEMBL1200')
   :type drug_chembl_id: str
   :param mesh_heading: MeSH disease heading for this drug-disease combination (e.g., 'Lung Neoplasms')
   :type mesh_heading: str
   :param lookup_table: Pre-computed family representation lookup from :func:`lookup_drug_family_representation`.
                        Used to identify which families are over-represented for this drug-disease pair.
   :type lookup_table: DataFrame
   :param pct_threshold: Percentage threshold for filtering large families (0-100). Families with
                         representation above this threshold are filtered (if other criteria met).
                         Lower values = more aggressive filtering. Defaults to 40.0.
   :type pct_threshold: float, optional
   :param small_family_threshold: Minimum family size for percentage-based filtering. Families smaller
                                  than this are exempt from filtering to preserve biological specificity.
                                  Defaults to 5.
   :type small_family_threshold: int, optional
   :param min_targets_threshold: Minimum number of targets in a family to trigger filtering. Families with
                                 fewer targets lack sufficient redundancy to justify filtering. Defaults to 3.
   :type min_targets_threshold: int, optional
   :return: Filtered DataFrame with same columns as input. Over-represented family members (except
            the first alphabetical representative) are removed. Returns input unchanged if no filtering
            is needed or if drug/disease not found in lookup table.
   :rtype: DataFrame
   
   .. note::
      **Alphabetical Selection**: Using the first alphabetical target as representative ensures
      deterministic, reproducible filtering across runs.
   
   .. note::
      **No Family Assignment**: Targets that don't map to HGNC gene families in the lookup table
      pass through unfiltered. These are typically novel targets or non-gene targets.
   
   .. warning::
      If ``drug_chembl_id`` and ``mesh_heading`` combination is not found in the lookup table,
      the function returns the input unfiltered and prints a warning. This can occur if the
      drug has no targets with family assignments.
   
   .. warning::
      **Aggressive Filtering**: Lowering ``pct_threshold`` or ``min_targets_threshold`` makes
      filtering more aggressive but may remove biologically important targets. Default values
      balance bias reduction with information retention.
   
   **Examples**
   
   Basic usage with default thresholds::
   
      >>> lookup = lookup_drug_family_representation(chembl_data)
      >>> filtered = filter_overrepresented_families(
      ...     targets_df=drug_targets,
      ...     drug_chembl_id='CHEMBL1200',  # Sunitinib
      ...     mesh_heading='Renal Cell Carcinoma',
      ...     lookup_table=lookup
      ... )
      >>> print(f"Filtered {len(drug_targets) - len(filtered)} targets")
   
   More aggressive filtering (lower thresholds)::
   
      >>> filtered = filter_overrepresented_families(
      ...     targets_df,
      ...     drug_chembl_id='CHEMBL123',
      ...     mesh_heading='Lung Neoplasms',
      ...     lookup_table=lookup,
      ...     pct_threshold=25.0,  # Filter families >25% represented
      ...     min_targets_threshold=2  # Filter families with >2 targets
      ... )
   
   Preserve small families (e.g., HDACs) from filtering::
   
      >>> # Don't filter families with <8 genes
      >>> filtered = filter_overrepresented_families(
      ...     targets_df,
      ...     drug_chembl_id='CHEMBL123',
      ...     mesh_heading='Lymphoma',
      ...     lookup_table=lookup,
      ...     small_family_threshold=8
      ... )
   
   Inspect which families were filtered::
   
      >>> original_targets = set(drug_targets['target_gene_name'])
      >>> filtered_targets = set(filtered['target_gene_name'])
      >>> removed = original_targets - filtered_targets
      >>> print(f"Removed targets: {removed}")
   
   .. seealso::
      - :func:`lookup_drug_family_representation`: Creates the required lookup table
      - :func:`compute`: Integrates this filtering into the clinical score calculation pipeline
   """
   # Query lookup table for this drug's family representation
   drug_families = lookup_table[
      (lookup_table['drug_chembl_id'] == drug_chembl_id) &
      (lookup_table['mesh_heading'] == mesh_heading)
   ].copy()
   
   # Handle case where drug has no family data
   if len(drug_families) == 0:
      print(f"Warning: No family data found for drug {drug_chembl_id} and mesh {mesh_heading}, skipping filtering")
      return targets_df
   
   # Identify over-represented families based on thresholds
   # Family must be: 
   # (1) large enough (≥ small_family_threshold) AND 
   # (2) has enough targets from this drug (> min_targets_threshold) AND
   # (3) EITHER highly represented (> pct_threshold%) OR has many absolute targets
   #
   # The last condition handles two cases:
   # - High percentage: drug targets most of a small/medium family (e.g., 80% of 10 genes)
   # - Many targets: drug targets a subfamily within a large family (e.g., 4 NDUFAF genes 
   #   within a 72-gene family, which is only 5.6% but still represents redundancy)
   #
   # NOTE: When absolute threshold is met (targets >= 4), bypass small family check
   # to handle cases like HDAC class IIA (4 genes total, all 4 targeted = 100%)
   overrep_families = drug_families[
      (drug_families['targets_in_family'] > min_targets_threshold) &
      (
         # Case 1: Large family with high percentage representation
         (
            (drug_families['total_genes_in_family'] >= small_family_threshold) &
            (drug_families['pct_represented'] > pct_threshold)
         ) |
         # Case 2: Absolute threshold met (bypass small family check)
         (drug_families['targets_in_family'] >= 4)
      )
   ]
   
   if len(overrep_families) > 0:
      # Report filtering context
      print(f"Found {len(overrep_families)} over-represented families for drug {drug_chembl_id} in {mesh_heading} (>{pct_threshold}%)")
      
      # Extract and deduplicate target lists from over-represented families
      # Use list comprehension for better performance than iterrows()
      all_family_targets = [
         row['target_genes_in_family'].split(', ')
         for _, row in overrep_families.iterrows()
      ]
      
      # Keep first alphabetical target from each family as representative
      targets_to_keep = [
         sorted(targets)[0] for targets in all_family_targets
      ]
      
      # Build exclusion list (all family targets except the kept representatives)
      targets_to_exclude = [
         target 
         for targets in all_family_targets 
         for target in targets 
         if target not in targets_to_keep
      ]
      
      print(f"  Excluding {len(targets_to_exclude)} targets, keeping {len(targets_to_keep)} (first alphabetical from each family)")
      
      # Apply filter and return
      return targets_df[
         ~targets_df['target_gene_name'].isin(targets_to_exclude)
      ].reset_index(drop=True)
   
   # No filtering needed - return input unchanged
   return targets_df

def compute(mesh_headings: list, chembl_version: str = '36', trials_only_last_n_years: int = None, filter_families: bool = True) -> dict:
   """
   Compute clinical validation scores for drug targets across multiple disease indications.
   
   This is the core function of the Artemis clinical scoring module. It processes ChEMBL drug-target-disease
   relationships combined with clinical trial data to calculate comprehensive validation scores for each
   target gene. These scores quantify the strength of clinical evidence supporting each target, enabling
   data-driven target prioritization for drug discovery.
   
   **What This Function Does:**
   
   - Loads drug molecule, target, and clinical trial data from ChEMBL
   - Expands diseases using MeSH hierarchies (e.g., "Lung Neoplasms" → all lung cancer subtypes)
   - Counts clinical trials at each phase (0-4) for each drug-target pair
   - Optionally filters out over-represented gene families to reduce bias
   - Aggregates evidence across all drugs targeting the same gene
   - Returns comprehensive target-level scores for each disease
   
   **Processing Workflow:**
   
   1. **Data Loading** (once for all diseases):
      
      - Downloads pre-computed ChEMBL molecules.csv from S3
      - Contains drug-target-indication-trial relationships
      - Optional temporal filtering to recent trials
   
   2. **Gene Family Analysis** (optional, once for all diseases):
      
      - Downloads HGNC gene family data
      - Creates lookup table of drug-family representation percentages
      - Identifies multi-target drugs (e.g., multi-kinase inhibitors)
   
   3. **Per-Disease Processing** (for each disease in mesh_headings):
      
   - **MeSH Expansion**: Retrieves all descendant disease terms
      (e.g., "Lung Neoplasms" includes "Adenocarcinoma of Lung", "Small Cell Lung Carcinoma", etc.)
   
   - **Trial De-duplication**: Removes duplicate trials from:

      - Same trial appearing in multiple mesh descendants
      - Same trial with different molecular forms (salts) of the same parent drug
   
   - **Family Filtering** (optional): For each drug-disease combination:

      - Identifies over-represented gene families (>40% of family targeted)
      - Keeps one representative target per family (first alphabetically)
      - Eliminates bias from multi-target drugs
   
   - **Trial Counting**: Counts trials per phase for each target:

      - Phase 0-3: Experimental/developmental trials
      - Phase 4: Approved drugs (post-marketing surveillance)
   
   - **Score Aggregation**: For each target gene:

      - Sum all trial phases (0-3) across all drugs → phase_scores
      - Check if any drug is approved (phase 4) → approved (boolean)
      - Calculate final score: clinical_scores = phase_scores + (20 if approved else 0)
   
   - **Output**: DataFrame with one row per target, sorted by clinical_scores (descending)
   
   **Scoring System Details:**
   
   Each target receives a **clinical_scores** value based on:
   
   1. **Phase Scores** (0-3 points per trial):
      
   - Phase 0: 0 points (exploratory studies)
   - Phase 1: 1 point (safety/dosing in small groups)
   - Phase 2: 2 points (efficacy in larger groups)
   - Phase 3: 3 points (large-scale confirmation)
   - Phase 4: 0 points (counted separately as "approved")
   
   2. **Approval Bonus**: +20 points if target has any approved drug
   
   3. **Multi-Drug Aggregation**: If multiple drugs target the same gene, all their phase scores sum
   
   4. **Multi-Trial Aggregation**: Each distinct clinical trial contributes its phase score
   
   **Example Scoring:**
   
   - Target A: 2 Phase 2 trials + 1 Phase 3 trial = 2+2+3 = 7 points
   - Target B: 1 Phase 3 trial + 1 approved drug = 3+20 = 23 points
   - Target C: 3 drugs with Phase 2 trials + approved = (2+2+2)+20 = 26 points
   
   **Why Family Filtering Matters:**
   
   Multi-target drugs (e.g., sunitinib targeting 8 RTKs) can artificially inflate scores for
   specific gene families. Without filtering, all 8 RTKs receive high scores from the same drug,
   biasing results toward that family. With filtering (default), only 1 RTK representative is kept,
   producing unbiased scores that better reflect biological diversity.
   
   :param mesh_headings: List of MeSH disease headings to analyze. Standard MeSH terms like
   - 'Breast Neoplasms' (breast cancer)
   - 'Lung Neoplasms' (lung cancer)
   - 'Diabetes Mellitus, Type 2' (type 2 diabetes)
   - 'Cardiovascular Diseases' (heart disease)
   
   Each heading is automatically expanded to include all MeSH descendant terms.
   Use exact MeSH terminology for best results.
   :type mesh_headings: list[str]

   :param chembl_version: ChEMBL database version to use. Versions correspond to data release dates
   - '36' (default): Data through 2024
   - '35': Data through 2023
   
   See https://www.ebi.ac.uk/chembl/ for version details.
   
   :type chembl_version: str, optional

   :param trials_only_last_n_years: Temporal filter for recent trials only. If provided
   - Filters to trials registered in last N years
   - Year inferred from ClinicalTrials.gov NCT IDs
   - Useful for identifying emerging targets
   
   Examples:
   - ``6``: Last 6 years (captures recent development)
   - ``10``: Last decade
   - ``None`` (default): All historical trials
   :type trials_only_last_n_years: int or None, optional

   :param filter_families: Whether to apply gene family filtering to reduce bias
                           
   - ``True`` (default): Filter over-represented families
      (recommended for unbiased target prioritization)
   - ``False``: Include all targets without filtering
      (use when family representation is biologically relevant)
   :type filter_families: bool, optional
   :return: Dictionary mapping each MeSH heading to its results DataFrame.
            
   **Dictionary Structure:**
   
   - Keys: MeSH disease headings (str) - same as input ``mesh_headings``
   - Values: DataFrames with target-level clinical scores
   
   **DataFrame Columns:**
   
   - ``target_gene_name`` (str): HGNC gene symbol (e.g., 'EGFR', 'TP53')
   - ``phase_0`` (int): Count of phase 0 trials for this target
   - ``phase_1`` (int): Count of phase 1 trials for this target
   - ``phase_2`` (int): Count of phase 2 trials for this target
   - ``phase_3`` (int): Count of phase 3 trials for this target
   - ``phase_4`` (int): Count of phase 4 (approved) drugs for this target
   - ``phase_scores`` (int): Sum of phase_1 + 2×phase_2 + 3×phase_3
   - ``approved`` (bool): True if phase_4 > 0 (any approved drug exists)
   - ``clinical_scores`` (int): phase_scores + (20 if approved else 0)
   
   DataFrames are sorted by ``clinical_scores`` descending (highest validation first).
      
   :rtype: dict[str, DataFrame]
   
   .. note::
      **Efficiency Optimization**: ChEMBL data and gene family lookup tables are loaded once and
      shared across all diseases. Processing 7 diseases takes ~5-10 minutes total, not per disease.
   
   .. note::
      **MeSH Hierarchy Expansion**: Diseases are automatically expanded to include all subtypes.
      For example, "Breast Neoplasms" includes "Ductal Carcinoma", "Lobular Carcinoma", etc.
      This ensures comprehensive coverage of clinical evidence.
   
   .. note::
      **Trial De-duplication**: The function carefully de-duplicates clinical trials to avoid
      double-counting. Trials are unique by (drug, NCT ID, target, disease) to ensure accurate counts.
   
   .. warning::
      **AWS S3 Access Required**: This function reads ChEMBL and MeSH data from the
      ``alethiotx-artemis`` S3 bucket. Configure AWS credentials via environment variables
      or ``aws configure`` before calling.
   
   .. warning::
      **Memory Usage**: ChEMBL v36 molecules.csv is ~500MB-1GB in memory. Processing multiple
      diseases simultaneously requires sufficient RAM. For memory-constrained environments,
      process diseases in smaller batches.
   
   .. warning::
      **Processing Time**: Full computation takes 5-10 minutes for 7 diseases with family filtering.
      Set ``filter_families=False`` to reduce runtime by ~50% (but scores may be biased).
   
   .. warning::
      **Family Filtering Off**: When ``filter_families=False``, results may show artificial
      enrichment for gene families with multi-target drugs. Use this option only when
      family-level representation is biologically meaningful for your analysis.
   
   **Examples**
   
   Basic usage - single disease::
   
   >>> from alethiotx.artemis.clinical import compute
   >>> results = compute(['Lung Neoplasms'])
   >>> lung_targets = results['Lung Neoplasms']
   >>> print(f"Found {len(lung_targets)} lung cancer targets")
   >>> print(lung_targets.head())
      target_gene_name  phase_0  phase_1  phase_2  phase_3  phase_4  phase_scores  approved  clinical_scores
   0  EGFR             0        5        15       8        3         45           True      65
   1  ALK              0        2        8        6        2         32           True      52
      
   Multiple diseases simultaneously::
   
   >>> diseases = ['Breast Neoplasms', 'Lung Neoplasms', 'Prostatic Neoplasms']
   >>> results = compute(diseases, chembl_version='36', filter_families=True)
   >>> for disease, df in results.items():
   ...     approved_count = df['approved'].sum()
   ...     print(f"{disease}: {len(df)} total targets, {approved_count} approved")
   
   Recent trials only (emerging targets)::
   
   >>> # Focus on targets with recent clinical activity (last 6 years)
   >>> results = compute(
   ...     ['Lung Neoplasms'],
   ...     trials_only_last_n_years=6,
   ...     filter_families=True
   ... )
   >>> recent_lung = results['Lung Neoplasms']
   >>> print("Emerging lung cancer targets:", recent_lung['target_gene_name'].tolist()[:10])
   
   Disable family filtering for comprehensive view::
   
   >>> # Include all targets without filtering (may show family bias)
   >>> results = compute(['Lung Neoplasms'], filter_families=False)
   >>> lung_all = results['Lung Neoplasms']
   >>> print(f"All targets (no filtering): {len(lung_all)}")
   
   Filter to approved targets only::
   
   >>> results = compute(['Diabetes Mellitus, Type 2'])
   >>> diabetes = results['Diabetes Mellitus, Type 2']
   >>> approved_diabetes = diabetes[diabetes['approved'] == True]
   >>> print(f"Approved diabetes targets: {len(approved_diabetes)}")
   >>> print(approved_diabetes[['target_gene_name', 'clinical_scores']].head())
   
   Compare targets across diseases::
   
   >>> diseases = ['Breast Neoplasms', 'Lung Neoplasms', 'Prostatic Neoplasms']
   >>> results = compute(diseases)
   >>> target = 'EGFR'
   >>> for disease in diseases:
   ...     df = results[disease]
   ...     if target in df['target_gene_name'].values:
   ...         score = df[df['target_gene_name'] == target]['clinical_scores'].iloc[0]
   ...         print(f"{disease}: {target} = {score} points")
   
   Save results to CSV files::
   
   >>> results = compute(['Breast Neoplasms', 'Lung Neoplasms'])
   >>> for disease, df in results.items():
   ...     filename = disease.replace(' ', '_').replace(',', '') + '.csv'
   ...     df.to_csv(filename, index=False)
   ...     print(f"Saved {filename}")
   
   Batch processing for all major diseases::
   
   >>> all_diseases = [
   ...     'Breast Neoplasms',
   ...     'Lung Neoplasms', 
   ...     'Prostatic Neoplasms',
   ...     'Skin Neoplasms',
   ...     'Intestinal Neoplasms',
   ...     'Diabetes Mellitus, Type 2',
   ...     'Cardiovascular Diseases'
   ... ]
   >>> results = compute(all_diseases, trials_only_last_n_years=6)
   >>> print(f"Processed {len(results)} diseases in one batch")
   
   .. seealso::
      - :func:`lookup_drug_family_representation`: Creates family lookup table used when ``filter_families=True``
      - :func:`filter_overrepresented_families`: Filters individual drug-disease combinations
      - :func:`load`: Load pre-computed scores from S3 (faster than computing from scratch)
      - :func:`~alethiotx.artemis.mesh.descendants`: Expands MeSH terms to include all subtypes
      - :func:`~alethiotx.artemis.chembl.molecules`: Retrieves raw ChEMBL molecule data
   """
   print("="*80)
   print("Loading ChEMBL data...")
   print("="*80)
   
   # Load ChEMBL clinical trials data once for all diseases (efficiency optimization)
   chembl = read_csv(f's3://alethiotx-artemis/data/chembl/{chembl_version}/molecules.csv')
   
   # Apply temporal filter if requested (e.g., only trials from last 6 years)
   if trials_only_last_n_years is not None:
      current_year = int(date.today().strftime("%Y"))
      cutoff_year = current_year - trials_only_last_n_years
      chembl = chembl[chembl['trial_year'] >= cutoff_year]
      print(f"Filtered to trials from last {trials_only_last_n_years} years (>= {cutoff_year})")
   
   # Create gene family representation lookup table once (shared across all diseases)
   # This identifies which drugs over-represent specific gene families
   lookup_table = None
   if filter_families:
      print("\nCreating drug-family representation lookup table...")
      lookup_table = lookup_drug_family_representation(chembl)
      print(f"Lookup table created with {len(lookup_table)} drug-mesh-family combinations")
   else:
      print("\nFamily filtering disabled - all targets will be included")
   
   # Process each disease indication independently
   results = {}
   
   for idx, mesh_heading in enumerate(mesh_headings, 1):
      print("\n" + "="*80)
      print(f"Processing disease {idx}/{len(mesh_headings)}: {mesh_heading}")
      print("="*80)
      
      # Step 1: Get all MeSH descendants first
      # Example: "Lung Neoplasms" includes "Adenocarcinoma of Lung", "Small Cell Lung Carcinoma", etc.
      mesh_descendants_list = mesh_descendants(
         mesh_heading,
         s3_base='s3://alethiotx-artemis/data/mesh/',
         file_base='d2025'
      )
      print(f"Including {len(mesh_descendants_list)} MeSH headings (including descendants)")
      
      # Filter ChEMBL data to this disease and its descendants
      disease_data = chembl[chembl['mesh_heading'].isin(mesh_descendants_list)].copy()
      
      # Step 2: De-duplicate clinical trials
      # Each row in ChEMBL represents one clinical trial. We need to avoid counting:
      # - Same trial with different child molecules (salt forms) of the same parent
      # - Same trial appearing in multiple mesh descendants
      # De-duplicate by (parent_molregno, clinical_trial_id, mesh_heading) first
      if 'clinical_trial_id' in disease_data.columns:
         disease_data = disease_data.drop_duplicates(
            subset=['parent_molregno', 'clinical_trial_id', 'mesh_heading', 'target_gene_name']
         )
      
      # Step 3: Group by mesh_heading + parent_molregno + target to aggregate trials per mesh
      # This intermediate step is needed for family filtering (which operates per mesh heading)
      parent_per_mesh = disease_data.groupby(
         ['mesh_heading', 'parent_molregno', 'target_gene_name'], 
         as_index=False
      ).agg(
         max_phase=('phase', 'max'),  # Keep max phase for this parent in this mesh
         chembl_id=('chembl_id', 'first')
      )
      
      # Step 4: Identify approved (per mesh)
      parent_per_mesh['approved'] = (parent_per_mesh['max_phase'] == 4)
      
      # Step 5: Calculate phase scores (per mesh)
      parent_per_mesh['phase_scores'] = parent_per_mesh['max_phase'].apply(
         lambda p: p if p != 4 else 0
      )
      
      res = parent_per_mesh
      
      # Handle empty result case (no trials for this disease)
      if len(res) == 0:
         print(f"Warning: No data found for {mesh_heading}")
         results[mesh_heading] = DataFrame(
            columns=['target_gene_name', 'phase_scores', 'approved', 'clinical_scores']
         )
         continue

      # Step 5: Filter over-represented gene families per drug across ALL mesh descendants (optional)
      # This prevents multi-target drugs (e.g., multi-kinase inhibitors) from biasing results
      # IMPORTANT: Filter at the parent mesh_heading level (not per descendant) to avoid
      # keeping multiple representatives from different sub-indications
      if filter_families:
         unique_drugs = res['chembl_id'].drop_duplicates()
         print(f"Filtering over-represented families for {len(unique_drugs)} drugs across all descendants...")
         
         filtered_results = []
         for drug_id in unique_drugs:
            # Extract all targets for this drug across ALL mesh descendants
            drug_data = res[res['chembl_id'] == drug_id].copy()
            
            # Get unique mesh headings for this drug
            unique_mesh = drug_data['mesh_heading'].unique()
            
            # Apply family filtering FOR EACH mesh heading separately
            # This is critical because we need to filter each mesh heading's records independently
            drug_filtered = []
            for mesh in unique_mesh:
               mesh_data = drug_data[drug_data['mesh_heading'] == mesh].copy()
               
               # Try filtering with the specific mesh first, fall back to parent if no data
               filtered_mesh = filter_overrepresented_families(
                  mesh_data, 
                  drug_chembl_id=drug_id,
                  mesh_heading=mesh,  # Use specific mesh heading
                  lookup_table=lookup_table
               )
               
               # If no filtering happened (no family data for specific mesh), try parent heading
               if len(filtered_mesh) == len(mesh_data) and mesh != mesh_heading:
                  filtered_mesh = filter_overrepresented_families(
                     mesh_data, 
                     drug_chembl_id=drug_id,
                     mesh_heading=mesh_heading,  # Fall back to parent heading
                     lookup_table=lookup_table
                  )
               
               drug_filtered.append(filtered_mesh)
            
            # Combine all mesh headings for this drug
            if len(drug_filtered) > 0:
               filtered_results.append(pd_concat(drug_filtered, ignore_index=True))
         
         # Combine all filtered drug data
         res = pd_concat(filtered_results, ignore_index=True)
         print(f"After filtering: {len(res)} records, {res['target_gene_name'].nunique()} unique targets")
      else:
         print("Skipping family filtering (filter_families=False)")
      
      # Step 6: Use FILTERED data and go back to trial level for counting
      # Important: Use 'res' (after family filtering) to preserve filtering results
      # Get the filtered parent-target combinations
      filtered_combinations = res[['parent_molregno', 'target_gene_name', 'chembl_id']].drop_duplicates()
      
      # Now go back to original trial data but only keep filtered combinations
      # This preserves family filtering while counting ALL trials
      if 'clinical_trial_id' in disease_data.columns:
         # Merge to keep only filtered parent-target combinations
         trials_filtered = disease_data.merge(
            filtered_combinations,
            on=['parent_molregno', 'target_gene_name'],
            how='inner'
         )
         
         # De-duplicate by clinical trial ID globally across mesh descendants
         trials_global = trials_filtered.drop_duplicates(
            subset=['parent_molregno', 'clinical_trial_id', 'target_gene_name']
         ).copy()  # Create explicit copy to avoid SettingWithCopyWarning
      else:
         # Fallback: if no trial IDs, use the aggregated filtered data
         trials_global = res.drop_duplicates(
            subset=['parent_molregno', 'target_gene_name', 'max_phase']
         ).copy()
      
      # Now aggregate all trials per target
      trials_global['approved'] = (trials_global['phase'] == 4)
      trials_global['phase_scores'] = trials_global['phase'].apply(
         lambda p: p if p != 4 else 0
      )
      
      print(f"After de-duplicating trials globally: {len(trials_global)} trial records, {trials_global['target_gene_name'].nunique()} unique targets")
      
      # Step 7: Count TRIALS per phase for each target
      # Each clinical trial in a given phase counts towards that phase's total
      phase_counts = trials_global.groupby('target_gene_name', as_index=False).agg(
         phase_0=('phase', lambda x: (x == 0).sum()),
         phase_1=('phase', lambda x: (x == 1).sum()),
         phase_2=('phase', lambda x: (x == 2).sum()),
         phase_3=('phase', lambda x: (x == 3).sum()),
         phase_4=('phase', lambda x: (x == 4).sum())
      )
      
      # Step 8: Aggregate ALL trial scores per target
      # Sum all clinical trial phases (0-3 for non-approved, 0 for approved)
      # Each trial contributes its phase score to the cumulative total
      phase_scores_final = trials_global.groupby('target_gene_name', as_index=False).agg(
         phase_scores=('phase_scores', 'sum')  # Sum ALL trial phase scores
      )
      
      # Check if target has any approved drug (any trial in phase 4)
      approved_final = trials_global.groupby('target_gene_name', as_index=False).agg(
         approved=('approved', lambda x: x.any())  # True if ANY trial is phase 4
      )
      
      # Merge all components
      res_final = merge(phase_scores_final, approved_final, on='target_gene_name')
      res_final = merge(res_final, phase_counts, on='target_gene_name')
      
      # Step 9: Calculate final clinical scores with approval bonus
      # Formula: clinical_score = phase_scores + (20 if approved else 0)
      res_final['clinical_scores'] = res_final['phase_scores'] + 20 * res_final['approved'].astype(int)
      
      # Reorder columns: target_gene_name, phase_0-4, phase_scores, approved, clinical_scores
      column_order = ['target_gene_name', 'phase_0', 'phase_1', 'phase_2', 'phase_3', 'phase_4',
                      'phase_scores', 'approved', 'clinical_scores']
      res_final = res_final[column_order]
      
      # Sort by clinical score descending (highest validation first)
      res_final.sort_values('clinical_scores', ascending=False, inplace=True)
      res_final.reset_index(drop=True, inplace=True)
      
      results[mesh_heading] = res_final
      print(f"Completed {mesh_heading}: {len(res_final)} targets")
   
   return results

def load(date = '2025-12-08'):
   """
   Load pre-computed clinical validation scores for seven major disease types from S3 storage.

   This convenience function retrieves pre-calculated clinical target scores that have been
   computed using :func:`compute` and stored in the Alethio Therapeutics public S3 bucket.
   Each disease has its own CSV file containing target-level clinical validation scores.
   
   **Disease Coverage:**
   
   Returns scores for these seven diseases (in order):
   
   1. Breast cancer (Breast Neoplasms)
   2. Lung cancer (Lung Neoplasms)
   3. Prostate cancer (Prostatic Neoplasms)
   4. Melanoma (Skin Neoplasms)
   5. Bowel/colorectal cancer (Intestinal Neoplasms)
   6. Type 2 diabetes (Diabetes Mellitus, Type 2)
   7. Cardiovascular disease (Cardiovascular Diseases)

   :param date: Date stamp identifying which pre-computed dataset to load. Format: ``YYYY-MM-DD``.
                Different dates represent different ChEMBL versions, temporal filters, or
                computation parameters. Defaults to ``2025-12-08``.
   :type date: str, optional
   :return: Tuple of 7 DataFrames containing clinical scores for each disease. Each DataFrame has:
            
   **Columns:**
   
   - ``Target Gene`` (str): HGNC gene symbol
   - ``phase_0`` through ``phase_4`` (int): Count of trials in each phase
   - ``Phase Score`` (int): Sum of trial phases (0-3)
   - ``approved`` (bool): Whether target has any approved (phase 4) drug
   - ``Clinical Score`` (int): Total validation score (Phase Score + 20 if approved)
   
   **Order:** 
   
   (breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular)

   :rtype: tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]
   :raises FileNotFoundError: If CSV files don't exist at the specified S3 paths for the given date
   :raises ValueError: If CSV files are malformed or missing required columns
   :raises botocore.exceptions.NoCredentialsError: If AWS credentials are not configured

   .. note::
      **S3 Storage Structure**: Files are located at:
      ``s3://alethiotx-artemis/data/clinical_scores/{date}/{disease}.csv``
      
      Where ``{disease}`` is: breast, lung, prostate, melanoma, bowel, diabetes, or cardiovascular

   .. note::
      **AWS Configuration**: Requires AWS credentials with read access to the ``alethiotx-artemis``
      S3 bucket. Configure via environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``)
      or AWS CLI (``aws configure``).

   .. warning::
      Verify the date exists in S3 before calling. Available dates depend on when scores were
      computed and uploaded. Contact Alethio Therapeutics for a list of available dates.

   .. warning::
      Each DataFrame can be large (thousands of rows). Consider filtering to specific targets
      or approved-only drugs after loading to reduce memory usage.

   **Examples**
   
   Load latest pre-computed scores::
   
      >>> from alethiotx.artemis.clinical import load
      >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load(date='2025-12-08')
      >>> print(f"Breast cancer: {len(breast)} targets")
      >>> print(breast.head())
   
   Load and filter to approved targets only::
   
      >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load()
      >>> breast_approved = breast[breast['approved'] == True]
      >>> print(f"Approved breast cancer targets: {len(breast_approved)}")
   
   Compare scores across diseases for a specific gene::
   
      >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load()
      >>> diseases = [breast, lung, prostate, melanoma, bowel, diabetes, cardio]
      >>> names = ['Breast', 'Lung', 'Prostate', 'Melanoma', 'Bowel', 'Diabetes', 'Cardio']
      >>> target = 'EGFR'
      >>> for name, df in zip(names, diseases):
      ...     if target in df['Target Gene'].values:
      ...         score = df[df['Target Gene'] == target]['Clinical Score'].iloc[0]
      ...         print(f"{name}: {score}")
   
   Load older dataset version::
   
      >>> # Load scores computed on an earlier date
      >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load(date='2025-11-11')
   
   .. seealso::
      - :func:`compute`: Function used to generate these pre-computed scores
      - :func:`approved`: Filter loaded scores to approved targets only
   """
   breast = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/breast.csv")
   lung = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/lung.csv")
   prostate = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/prostate.csv")
   melanoma = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/melanoma.csv")
   bowel = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/bowel.csv")
   diabetes = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/diabetes.csv")
   cardiovascular = read_csv("s3://alethiotx-artemis/data/clinical_scores/" + date + "/cardiovascular.csv")

   return(breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular)

def approved(scores: list):
   """
   Filter clinical score DataFrames to include only targets with approved drugs (Phase 4).

   This utility function processes a list of clinical score DataFrames and retains only
   targets that have at least one approved drug (phase 4 trials). This is useful for
   focusing analysis on clinically validated targets with FDA/EMA-approved therapeutics.
   
   **Use Cases:**
   
   - Benchmark new targets against established approved targets
   - Study successful drug development patterns
   - Identify approved drugs for repurposing opportunities
   - Filter out experimental/developmental targets

   :param scores: List of DataFrames containing clinical scores. Each DataFrame must have
                  an ``approved`` column (boolean) indicating whether the target has any
                  approved (phase 4) drugs. Typically obtained from :func:`load` or :func:`compute`.
   :type scores: list[DataFrame]
   :return: List of DataFrames (same length and order as input) with only approved targets.
            Original DataFrames are **not** modified (returns filtered copies).
   :rtype: list[DataFrame]

   .. note::
      This function creates copies of the input DataFrames. Original DataFrames remain unchanged.

   .. note::
      Empty DataFrames are returned for diseases with no approved targets (e.g., some rare cancers).

   **Examples**
   
   Filter pre-computed scores to approved only::
   
      >>> from alethiotx.artemis.clinical import load, approved
      >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load()
      >>> breast_app, lung_app, prostate_app, melanoma_app, bowel_app, diabetes_app, cardio_app = approved(
      ...     [breast, lung, prostate, melanoma, bowel, diabetes, cardio]
      ... )
      >>> print(f"Approved breast cancer targets: {len(breast_app)}")
   
   Compare approved vs all targets::
   
      >>> all_scores = load()
      >>> approved_scores = approved(list(all_scores))
      >>> for i, disease in enumerate(['Breast', 'Lung', 'Prostate', 'Melanoma', 'Bowel', 'Diabetes', 'Cardio']):
      ...     all_count = len(all_scores[i])
      ...     approved_count = len(approved_scores[i])
      ...     pct = 100 * approved_count / all_count if all_count > 0 else 0
      ...     print(f"{disease}: {approved_count}/{all_count} ({pct:.1f}%) approved")
   
   Identify top approved targets::
   
      >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load()
      >>> breast_approved = approved([breast])[0]
      >>> top_5 = breast_approved.nlargest(5, 'Clinical Score')
      >>> print(top_5[['Target Gene', 'Clinical Score']])
   
   .. seealso::
      - :func:`load`: Load pre-computed scores (returns all targets)
      - :func:`compute`: Compute scores with ``approved`` column
   """
   res = scores.copy()

   for n, d in enumerate(res):
      res[n] = d[d['approved'] == True]

   return(res)

def unique(scores: list, overlap = 1, common_genes = []):
    """
    Remove cross-disease overlapping genes from clinical scores to create disease-specific target lists.
    
    When analyzing multiple diseases, some genes appear as targets across many indications (e.g., EGFR
    in lung, breast, and colorectal cancers). For disease-specific analyses or comparative studies,
    removing these "pan-disease" targets reveals indication-specific biology.
    
    This function identifies genes present in multiple disease DataFrames and removes them from all
    DataFrames, leaving only targets unique to fewer diseases.
    
    **Overlap Logic:**
    
    A gene is considered "overlapping" and removed if:
    
    - It appears in **more than** ``overlap`` diseases (not in all), OR
    - It's in the ``common_genes`` list (manual specification)
    
    **Important**: Genes appearing in **all** diseases are NOT removed by default (see Notes).
    
    **Use Cases:**
    
    - Identify disease-specific therapeutic targets
    - Remove broadly-acting targets (e.g., TP53, EGFR) for specialized analysis
    - Create non-overlapping target sets for comparative drug discovery
    - Study indication-specific versus pan-cancer biology
    
    :param scores: List of DataFrames containing clinical scores. Each must have a ``Target Gene``
                   column with HGNC gene symbols. Typically one DataFrame per disease.
    :type scores: list[DataFrame]
    :param overlap: Minimum number of diseases a gene must appear in to be removed (exclusive of all-disease genes).
                    - ``overlap=1``: Remove genes in 2+ diseases (but not all)
                    - ``overlap=2``: Remove genes in 3+ diseases (but not all)
                    - ``overlap=N``: Remove genes in (N+1)+ diseases (but not all)
                    Defaults to 1.
    :type overlap: int, optional
    :param common_genes: List of gene symbols to always remove regardless of overlap count.
                         Useful for manually excluding known pan-disease targets. Defaults to [].
    :type common_genes: list[str], optional
    :return: List of DataFrames (same length/order as input) with overlapping genes removed.
             Original DataFrames are **not** modified (returns filtered copies).
    :rtype: list[DataFrame]

    .. note::
       **All-Disease Genes Preserved**: By default, genes appearing in ALL input DataFrames are
       NOT removed. This preserves truly universal targets. To change this behavior, modify the
       comparison in :func:`find_overlapping_genes` from ``<`` to ``<=``.

    .. note::
       This function uses :func:`find_overlapping_genes` internally to identify overlap.

    .. warning::
       This function creates copies of input DataFrames. Originals remain unchanged.

    .. warning::
       Setting ``overlap`` too low (e.g., 0) removes genes present in only 1+ diseases,
       potentially eliminating most of your data. Use with caution.

    **Examples**
    
    Remove genes appearing in 2+ diseases (but not all 7)::
    
       >>> from alethiotx.artemis.clinical import load, unique
       >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load()
       >>> scores_list = [breast, lung, prostate, melanoma, bowel, diabetes, cardio]
       >>> unique_scores = unique(scores_list, overlap=1)
       >>> breast_unique, lung_unique, *rest = unique_scores
       >>> print(f"Breast: {len(breast)} -> {len(breast_unique)} unique targets")
    
    More aggressive filtering (remove genes in 3+ diseases)::
    
       >>> unique_scores = unique(scores_list, overlap=2)
    
    Manually exclude known pan-cancer targets::
    
       >>> pan_cancer_genes = ['TP53', 'EGFR', 'KRAS', 'PIK3CA']
       >>> unique_scores = unique(scores_list, overlap=1, common_genes=pan_cancer_genes)
    
    Compare disease-specific vs all targets::
    
       >>> all_scores = load()
       >>> unique_scores = unique(list(all_scores), overlap=1)
       >>> for i, disease in enumerate(['Breast', 'Lung', 'Prostate']):
       ...     original = len(all_scores[i])
       ...     unique_count = len(unique_scores[i])
       ...     removed = original - unique_count
       ...     print(f"{disease}: removed {removed} overlapping genes ({100*removed/original:.1f}%)")
    
    Identify what genes were removed::
    
       >>> original_breast_genes = set(breast['Target Gene'])
       >>> unique_breast_genes = set(unique_scores[0]['Target Gene'])
       >>> removed_genes = original_breast_genes - unique_breast_genes
       >>> print(f"Removed from breast: {sorted(removed_genes)}")
    
    .. seealso::
       - :func:`find_overlapping_genes`: Identifies which genes overlap across disease lists
       - :func:`load`: Load clinical scores for multiple diseases
    """
    genes = []
    for n, d in enumerate(scores):
        genes.append(d['Target Gene'].tolist())

    overlapping_genes = find_overlapping_genes(genes, overlap = overlap, common_genes = common_genes)
    
    res = scores.copy()

    for n, d in enumerate(res):
        res[n] = d[~d['Target Gene'].isin(overlapping_genes)]
    
    return(res)

def all_targets(scores: list):
    """
    Extract all unique target genes from multiple clinical score DataFrames.
    
    This utility function aggregates target genes across multiple diseases to create a comprehensive
    list of all genes with clinical validation evidence. Useful for building negative sample sets,
    identifying the universe of clinically-validated targets, or creating background gene sets for
    statistical analysis.

    :param scores: List of DataFrames containing clinical scores. Each must have a ``Target Gene``
                   column with gene symbols. Typically one DataFrame per disease from :func:`load`
                   or :func:`compute`.
    :type scores: list[DataFrame]
    :return: List of unique gene symbols (HGNC) found across all input DataFrames. Order is not guaranteed.
    :rtype: list[str]

    .. note::
       Duplicates are automatically removed. If GENE1 appears in multiple diseases, it's included
       only once in the output.

    .. note::
       The output is a plain Python list, not a pandas Series. Convert if needed:
       ``pd.Series(all_targets(scores))``

    **Examples**
    
    Get all clinically-validated targets across diseases::
    
       >>> from alethiotx.artemis.clinical import load, all_targets
       >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load()
       >>> all_genes = all_targets([breast, lung, prostate, melanoma, bowel, diabetes, cardio])
       >>> print(f"Total unique clinical targets: {len(all_genes)}")
    
    Build known target list for machine learning::
    
       >>> # Use as known_targets in cv.prepare() to exclude from negative sampling
       >>> from alethiotx.artemis.cv import prepare
       >>> all_clinical_targets = all_targets(disease_scores)
       >>> result = prepare(X, y, pathway_genes=pg, known_targets=all_clinical_targets)
    
    Compare target overlap between diseases::
    
       >>> breast_genes = set(breast['Target Gene'])
       >>> lung_genes = set(lung['Target Gene'])
       >>> all_genes = set(all_targets([breast, lung]))
       >>> print(f"Breast: {len(breast_genes)}, Lung: {len(lung_genes)}, Total unique: {len(all_genes)}")
       >>> print(f"Overlap: {len(breast_genes & lung_genes)} genes")
    
    Create gene set for enrichment analysis::
    
       >>> import gseapy as gp
       >>> all_clinical = all_targets(disease_scores)
       >>> enrich = gp.enrichr(gene_list=all_clinical, gene_sets='KEGG_2021_Human')
    
    .. seealso::
       - :func:`unique`: Remove overlapping genes across diseases (opposite operation)
       - :func:`load`: Load clinical scores containing target genes
    """
    all_targets = set()

    for d in scores:
        all_targets.update(d['Target Gene'].tolist())

    return(list(all_targets))

# run when file is directly executed
if __name__ == '__main__':

   # breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular = load(date = '2025-12-08')
   # breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular = approved([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular])
   # print(breast)

   # Define all diseases to process
   diseases = [
      'Breast Neoplasms',
      'Lung Neoplasms',
      'Prostatic Neoplasms',
      'Skin Neoplasms',
      'Intestinal Neoplasms',
      'Diabetes Mellitus, Type 2',
      'Cardiovascular Diseases'
   ]
   
   # Compute scores for all diseases in one call
   print("\nComputing clinical scores for all diseases...")
   results = compute(diseases, chembl_version='36', trials_only_last_n_years=6, filter_families=True)
   
   # Save results to individual CSV files
   print("\n" + "="*80)
   print("Saving results...")
   print("="*80)
   
   # Map disease names to file names
   disease_to_filename = {
      'Breast Neoplasms': 'breast.csv',
      'Lung Neoplasms': 'lung.csv',
      'Prostatic Neoplasms': 'prostate.csv',
      'Skin Neoplasms': 'melanoma.csv',
      'Intestinal Neoplasms': 'bowel.csv',
      'Diabetes Mellitus, Type 2': 'diabetes.csv',
      'Cardiovascular Diseases': 'cardiovascular.csv'
   }
   
   for disease, filename in disease_to_filename.items():
      if disease in results:
         d = results[disease]
         d.rename(columns={'target_gene_name': 'Target Gene', 'phase_scores': 'Phase Score', 'clinical_scores': 'Clinical Score'}, inplace=True)
         d.to_csv(filename, index=False)
         print(f"  Saved {filename}: {len(d)} targets")
   
   print("\n" + "="*80)
   print("All processing complete!")
   print("="*80)