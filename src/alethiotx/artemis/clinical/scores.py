from pandas import DataFrame, merge, read_csv, concat as pd_concat, notna
from datetime import date
from ..mesh.get import descendants as mesh_descendants
from ..hgnc.families import download as hgnc_download, process as hgnc_process
from ..utils import find_overlapping_genes

def lookup_drug_family_representation(chembl: DataFrame) -> DataFrame:
   """
   Create a lookup table mapping drug-disease-family combinations to their gene family representation.
   
   This function analyzes clinical trial data to determine what percentage of each gene family's
   genes are targeted by each drug for each disease indication. This lookup table is used to
   identify drugs that over-represent specific gene families, which can introduce bias in
   target prioritization.
   
   The function performs the following steps:
   
   1. Downloads HGNC gene family data
   2. Extracts unique drug-target-disease combinations from ChEMBL
   3. Maps each target to its HGNC gene family
   4. Calculates the percentage of each family represented by each drug's targets
   5. Returns a comprehensive lookup table
   
   :param chembl: ChEMBL clinical trials data with columns ``chembl_id``, ``target_gene_name``, ``mesh_heading``
   :type chembl: DataFrame
   :return: Lookup table with one row per drug-disease-family combination containing:
      
      - ``drug_chembl_id``: ChEMBL drug identifier
      - ``mesh_heading``: Disease indication (MeSH term)
      - ``family_id``: HGNC gene family identifier
      - ``family_abbreviation``: Short name for the gene family
      - ``family_name``: Full descriptive name of the gene family
      - ``total_genes_in_family``: Total number of genes in this family
      - ``targets_in_family``: Number of family genes targeted by this drug
      - ``target_genes_in_family``: Comma-separated list of targeted gene symbols
      - ``pct_represented``: Percentage of family genes targeted (0-100)
      
   :rtype: DataFrame
   
   .. note::
      Only drug-disease combinations where targets have HGNC family assignments are included.
      Targets without family assignments are excluded from the final table.
   
   .. warning::
      This function downloads HGNC data which may take time on first execution.
      Consider caching the result for repeated analyses.
   
   **Example**
   
   >>> chembl_data = read_csv('s3://bucket/chembl/36/molecules.csv')
   >>> lookup = lookup_drug_family_representation(chembl_data)
   >>> # Find drugs over-representing receptor tyrosine kinases
   >>> rtk_overrep = lookup[(lookup['family_name'].str.contains('tyrosine kinase')) & 
   ...                      (lookup['pct_represented'] > 50)]
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
   Filter out targets from over-represented gene families to reduce bias in target prioritization.
   
   When a drug targets many genes from the same family (e.g., multiple kinases), this creates
   bias in clinical scoring by artificially inflating the importance of that family. This function
   identifies over-represented families and keeps only one representative target per family
   (the first alphabetically) to eliminate this bias.
   
   **Filtering Strategy:**
   
   - Families with ≥ ``small_family_threshold`` genes AND > ``pct_threshold``% representation 
     AND > ``min_targets_threshold`` targets from the drug are filtered
   - Small families (< ``small_family_threshold`` genes) are never filtered (preserve biological specificity)
   - Families with few targets (≤ ``min_targets_threshold`` targets) are never filtered (not enough redundancy)
   - For each over-represented family, keep first alphabetical target, exclude all others
   
   :param targets_df: DataFrame containing drug targets with 'target_gene_name' column
   :type targets_df: DataFrame
   :param drug_chembl_id: ChEMBL drug identifier
   :type drug_chembl_id: str
   :param mesh_heading: MeSH disease heading for the drug-disease combination
   :type mesh_heading: str
   :param lookup_table: Pre-computed drug-disease-family representation table from
      :func:`lookup_drug_family_representation`
   :type lookup_table: DataFrame
   :param pct_threshold: Percentage threshold for over-representation (0-100).
      Families with > this percentage of genes targeted are filtered, defaults to 40.0
   :type pct_threshold: float, optional
   :param small_family_threshold: Minimum family size for filtering. Families with fewer genes
      are exempt from filtering regardless of representation, defaults to 5
   :type small_family_threshold: int, optional
   :param min_targets_threshold: Minimum number of targets from the drug in a family for filtering.
      Families with ≤ this many targets are exempt from filtering, defaults to 3
   :type min_targets_threshold: int, optional
   :return: Filtered DataFrame with same structure as input, over-represented family targets removed
   :rtype: DataFrame
   
   .. note::
      Targets not in the lookup table (no family assignment) pass through unfiltered.
   
   .. warning::
      If the drug has no data in the lookup table, returns input unfiltered with a warning message.
   
   **Example**
   
   >>> # Filter targets for a specific drug-disease combination
   >>> filtered = filter_overrepresented_families(
   ...     targets_df,
   ...     drug_chembl_id='CHEMBL123',
   ...     mesh_heading='Lung Neoplasms',
   ...     lookup_table=lookup_df,
   ...     pct_threshold=45.0,
   ...     min_targets_threshold=5
   ... )
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
   
   This function processes clinical trial data to calculate validation scores for drug targets,
   helping prioritize targets based on their clinical evidence. The workflow is optimized to
   process multiple diseases efficiently by sharing common resources (ChEMBL data and gene
   family lookup tables) across all disease computations.
   
   **Workflow:**
   
   1. **Data Loading**: Downloads ChEMBL clinical trials data once for all diseases
   2. **Temporal Filtering**: Optionally filters to recent trials (last N years)
   3. **Family Analysis**: Optionally creates gene family representation lookup table to identify biased drugs
   4. **Per-Disease Processing**: For each disease indication:

   - Calculates phase scores (sum of trial phases 0-3)
   - Identifies approved targets (phase 4 trials present)
   - Expands to include MeSH descendants (e.g., "Lung Neoplasms" includes "Adenocarcinoma of Lung")
   - Optionally filters over-represented gene families per drug-disease combination
   - Aggregates scores across all drugs and sub-diseases per target
   - Computes final clinical score = phase_scores + (20 if approved else 0)
   
   **Scoring System:**
   
   - Phase contributions: ``Phase 0``=0, ``Phase 1``=1, ``Phase 2``=2, ``Phase 3``=3 (cumulative per drug)
   - Approval bonus: +20 points if target has any phase 4 (approved) trial
   - Multiple drugs: Scores sum across all drugs targeting the same gene
   - Multiple trials: Each trial phase contributes to the cumulative score
   
   **Family Filtering:**
   
   When enabled (``filter_families``=True), this reduces bias from drugs targeting many genes in the 
   same family (e.g., multi-kinase inhibitors) by filtering over-represented families per drug-disease 
   combination, keeping only one representative target (first alphabetically) from each over-represented family.
   Set to False to include all targets without family-based filtering.
   
   :param mesh_headings: List of MeSH disease headings to analyze (e.g., ['Breast Neoplasms', 'Lung Neoplasms']).
      Each heading is automatically expanded to include all descendant terms in the MeSH hierarchy
   :type mesh_headings: list of str
   :param chembl_version: ChEMBL database version to use. Version 36 contains trials up to 2024, defaults to ``36``
   :type chembl_version: str, optional
   :param trials_only_last_n_years: Number of recent years to include. If provided, filters trials
      where ``trial_year`` >= (``current_year`` - N). If None, includes all historical trials, defaults to None
   :type trials_only_last_n_years: int, optional
   :param filter_families: Whether to filter over-represented gene families. When True, removes targets
      from over-represented families to reduce bias. When False, includes all targets, defaults to True
   :type filter_families: bool, optional
   :return: Dictionary mapping each disease heading to its results DataFrame. Each DataFrame contains: 
      - **target_gene_name** (str): HGNC gene symbol of the drug target
      - **phase_scores** (int): Sum of all clinical trial phases (0-3) across all drugs
      - **approved** (bool): Whether target has at least one approved (phase 4) drug
      - **clinical_scores** (int): Total clinical validation score (phase_scores + approval bonus)
      
   .. note::
      When ``filter_families``=True, the function creates a shared gene family lookup table once 
      for efficiency. This table is reused across all diseases, significantly reducing computation 
      time compared to processing diseases independently.
   
   .. warning::
      - Requires AWS credentials with S3 read access to alethiotx-artemis bucket
      - Processing time scales with number of diseases and trial data volume
      - Memory usage can be significant for large ChEMBL datasets (>1GB for version 36)
      - When ``filter_families``=False, results may include bias from multi-target drugs
   
   **Examples**
   
   Basic usage with single disease::
   
      >>> results = compute(['Lung Neoplasms'])
      >>> lung_targets = results['Lung Neoplasms']
      >>> print(lung_targets.head())
         target_gene_name  phase_scores  approved  clinical_scores
      0  EGFR             45            True      65
      1  ALK              32            True      52
      
   Multiple diseases with recent trials only::
   
      >>> results = compute(
      ...     ['Breast Neoplasms', 'Lung Neoplasms', 'Prostatic Neoplasms'],
      ...     trials_only_last_n_years=5
      ... )
      >>> for disease, df in results.items():
      ...     print(f"{disease}: {len(df)} targets")
   
   Filter to approved targets only::
   
      >>> results = compute(['Diabetes Mellitus, Type 2'])
      >>> approved_targets = results['Diabetes Mellitus, Type 2']
      >>> approved_targets = approved_targets[approved_targets['approved'] == True]
      >>> print(f"Found {len(approved_targets)} approved diabetes targets")
   
   Disable family filtering::
   
      >>> results = compute(['Lung Neoplasms'], filter_families=False)
      >>> # All targets included without family-based filtering
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
   Retrieve clinical scores for multiple disease types from S3 storage.

   This function reads CSV files containing clinical target data for various diseases
   from an S3 bucket, organized by date.

   :param date: The date string used to construct the S3 file paths, defaults to ``2025-12-08``
   :type date: str, optional
   :return: A tuple containing DataFrames for breast, lung, prostate, melanoma, bowel,
            diabetes, and cardiovascular clinical scores (in that order)
   :rtype: tuple of pandas.DataFrame
   :raises FileNotFoundError: If any of the CSV files do not exist at the specified S3 paths
   :raises ValueError: If the CSV files cannot be parsed correctly

   .. note::

      All CSV files are expected to be located in the S3 bucket at:
      ``s3://alethiotx-artemis/data/clinical_scores/{date}/{disease}.csv``

   .. warning::

      Ensure data exists
      for the specified date before calling this function.

   **Example**
   >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load_clinical_scores('2025-12-08')
   >>> print(breast.shape)
   (100, 5)
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
   Filter scores to include only approved entries.

   This function creates a copy of the input list of DataFrames and filters each
   DataFrame to retain only rows where the 'approved' column is True.

   :param scores: A list of pandas DataFrames, each containing an 'approved' column
   :type scores: list
   :return: A list of DataFrames with only approved entries
   :rtype: list

   :Example:

   >>> import pandas as pd
   >>> df1 = pd.DataFrame({'approved': [True, False, True], 'value': [1, 2, 3]})
   >>> df2 = pd.DataFrame({'approved': [False, True], 'value': [4, 5]})
   >>> result = approved([df1, df2])
   >>> len(result[0])
   2
   >>> len(result[1])
   1
   """
   res = scores.copy()

   for n, d in enumerate(res):
      res[n] = d[d['approved'] == True]

   return(res)

def unique(scores: list, overlap = 1, common_genes = []):
    """
    Remove overlapping genes from clinical score dataframes to ensure uniqueness.
    This function processes a list of dataframes containing clinical scores and removes
    genes that appear in multiple dataframes above a specified overlap threshold.
    :param scores: List of dataframes, each containing a ``Target Gene`` column with gene identifiers
    :type scores: list
    :param overlap: Minimum number of dataframes a gene must appear in to be considered overlapping, defaults to 1
    :type overlap: int, optional
    :param common_genes: Additional list of genes to always consider as overlapping regardless of frequency, defaults to []
    :type common_genes: list, optional
    :return: List of dataframes with overlapping genes removed from each dataframe
    :rtype: list

    **Example**
    >>> df1 = pd.DataFrame({'Target Gene': ['BRCA1', 'TP53', 'EGFR']})
    >>> df2 = pd.DataFrame({'Target Gene': ['TP53', 'KRAS', 'MYC']})
    >>> result = uniquify_clinical_scores([df1, df2], overlap=2)
    >>> # TP53 will be removed from both dataframes as it appears in 2 or more

    .. note::

        The function uses :func:`find_overlapping_genes` to identify genes that should be removed.
    .. warning::

        This function modifies copies of the input dataframes. The original dataframes remain unchanged.
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
    Extract all unique target genes from a list of score dictionaries.

    :param scores: A list of dictionaries, where each dictionary contains a ``Target Gene`` key
                   with a value that can be converted to a list using tolist() method
                   (typically a pandas Series or similar object)
    :type scores: list
    :return: A list of all unique target genes found across all dictionaries in the input list
    :rtype: list

    **Example**
    >>> scores = [
    ...     {'Target Gene': pd.Series(['GENE1', 'GENE2'])},
    ...     {'Target Gene': pd.Series(['GENE2', 'GENE3'])}
    ... ]
    >>> get_all_targets(scores)
    ['GENE1', 'GENE2', 'GENE3']
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