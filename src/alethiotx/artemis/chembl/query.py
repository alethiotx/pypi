import chembl_downloader
import pandas as pd
import re

def infer_nct_year(nct_id):
    """
    Infer the approximate registration year from a ClinicalTrials.gov NCT identifier.
    
    NCT IDs follow the format ``NCT########``, where the 8-digit numeric portion is assigned
    sequentially and increases over time. This function uses empirically-observed NCT ID
    allocation patterns to estimate when a trial was registered, which is useful for temporal
    filtering and analysis when exact registration dates are not available.

    **NCT ID Allocation Ranges:**
    
    - ``NCT00000000`` - ``NCT00999999``: ~1999-2004
    - ``NCT01000000`` - ``NCT01999999``: ~2005-2011
    - ``NCT02000000`` - ``NCT02999999``: ~2012-2015
    - ``NCT03000000`` - ``NCT03999999``: ~2016-2018
    - ``NCT04000000`` - ``NCT04999999``: ~2019-2021
    - ``NCT05000000`` - ``NCT05999999``: ~2022-2023
    - ``NCT06000000``+: ~2024+

    :param nct_id: A ClinicalTrials.gov identifier (e.g., ``NCT00500000``)
    :type nct_id: str
    :return: Estimated year of trial registration as an integer, or ``None`` if the NCT ID is invalid/malformed
    :rtype: int or None

    **Examples**
    
    >>> infer_nct_year("NCT00500000")
    2002
    >>> infer_nct_year("NCT03000000")
    2016
    >>> infer_nct_year("NCT06123456")
    2024
    >>> infer_nct_year("invalid")
    None
    >>> infer_nct_year("NCT123")  # Too short
    None
    
    **Use Cases:**
    
    - Filtering clinical trials data by approximate time period
    - Temporal analysis of drug development trends
    - Quick year estimation when full trial metadata is unavailable
    
    .. note::
        This function provides an **approximation** based on historical NCT ID allocation
        patterns. Individual trials may vary by ±1-2 years from the estimated value.
        For precise temporal analysis, obtain the actual registration date from the
        ClinicalTrials.gov API or database.
    
    .. warning::
        Returns ``None`` for invalid inputs including non-string types, IDs without the
        "NCT" prefix, or IDs that don't contain exactly 8 digits after "NCT".
    """
    if not isinstance(nct_id, str) or not nct_id.startswith('NCT'):
        return None
    
    # Extract the numeric part
    match = re.search(r'NCT(\d{8})', nct_id)
    if not match:
        return None
    
    num = int(match.group(1))
    
    # Approximate year ranges based on NCT ID ranges
    if num < 1000000:
        return 1999 + (num // 200000)  # ~1999-2004
    elif num < 2000000:
        return 2005 + ((num - 1000000) // 140000)  # ~2005-2011
    elif num < 3000000:
        return 2012 + ((num - 2000000) // 250000)  # ~2012-2015
    elif num < 4000000:
        return 2016 + ((num - 3000000) // 330000)  # ~2016-2018
    elif num < 5000000:
        return 2019 + ((num - 4000000) // 350000)  # ~2019-2021
    elif num < 6000000:
        return 2022 + ((num - 5000000) // 500000)  # ~2022-2023
    else:
        return 2024 + ((num - 6000000) // 500000)  # ~2024+

def molecules(version: str = '36', top_n_activities: int = 1):
    """
    Query ChEMBL database for bioactive drug molecules with clinical trial data and therapeutic indications.
    
    This function retrieves comprehensive drug-target-indication relationships from ChEMBL, automatically
    normalizing all molecular forms (salts, formulations, etc.) to their parent compounds. It integrates
    clinical trial phases, MeSH disease classifications, drug mechanisms, and target information to create
    a unified dataset for drug target prioritization and discovery.
    
    **Data Processing Workflow:**
    
    1. **Parent Normalization**: All child molecules (salts, prodrugs, formulations) are mapped to their
       parent compound using ChEMBL's molecule hierarchy
    2. **Indication Aggregation**: Drug indications from both parent and all child molecules are combined
    3. **Target Assignment**: Molecular targets are identified using a three-tier hierarchy:
       
    - Primary: Parent molecule's ``DRUG_MECHANISM`` table entries (known mechanisms)
    - Secondary: Child molecule mechanisms (inherited when parent lacks mechanisms)
    - Tertiary: Top N most-studied targets from ``ACTIVITIES`` table (bioassay data)
    
    4. **Clinical Trial Mapping**: Links molecules to ClinicalTrials.gov identifiers with phase information
    5. **Year Inference**: Estimates trial registration year from NCT identifiers
    
    **Key Features:**
    
    - Only includes molecules with clinical trial references (ClinicalTrials.gov)
    - Filters to human targets only (``Homo sapiens``)
    - One molecule-indication-target per row (exploded format for multi-trial drugs)
    - Mechanisms apply to all indications of a molecule (not indication-specific)
    
    :param version: ChEMBL database version to query. Version 36 covers data through 2024.
                    See https://www.ebi.ac.uk/chembl/ for available versions.
    :type version: str, optional
    :param top_n_activities: For molecules without documented mechanisms (no ``DRUG_MECHANISM`` entries),
                             include the top N most-studied targets from bioassay data. Set to 0 to
                             exclude activity-based targets entirely. Defaults to 1 (most-studied target only).
    :type top_n_activities: int, optional
    
    :return: DataFrame with one row per parent-molecule-indication-target combination. 
    
    **Columns:**
    
    **Molecule Information:**
    
    - ``chembl_id`` (str): ChEMBL identifier for the parent molecule (e.g., 'CHEMBL25')
    - ``pref_name`` (str): Preferred drug name (e.g., 'ASPIRIN')
    - ``parent_molregno`` (int): Internal ChEMBL registry number for parent molecule
    
    **Indication Information:**
    
    - ``mesh_heading`` (str): MeSH disease term (e.g., 'Lung Neoplasms')
    - ``mesh_id`` (str): MeSH unique identifier (e.g., 'D008175')
    - ``phase`` (int): Maximum clinical trial phase for this indication (0-4, where 4=approved)
    - ``reference_type`` (str): Always 'ClinicalTrials' (pre-filtered)
    - ``clinical_trial_id`` (str): ClinicalTrials.gov NCT identifier (e.g., 'NCT00123456')
    - ``trial_year`` (int, nullable): Inferred trial registration year via :func:`infer_nct_year`
    
    **Target Information:**
    
    - ``target_chembl_id`` (str): ChEMBL target identifier (e.g., 'CHEMBL240')
    - ``target_organism`` (str): Always 'Homo sapiens' (pre-filtered)
    - ``target_type`` (str): Target classification (e.g., 'SINGLE PROTEIN', 'PROTEIN COMPLEX')
    - ``target_uniprot_id`` (str, nullable): UniProt accession (e.g., 'P35354')
    - ``target_gene_name`` (str, nullable): HGNC gene symbol (e.g., 'EGFR')
    - ``mechanism_of_action`` (str, nullable): Mechanism description (NULL for activity-derived targets)
    - ``action_type`` (str, nullable): Drug action type (e.g., 'INHIBITOR', 'AGONIST'; NULL for activities)
    - ``target_source`` (str): Data provenance - one of:
        
        - ``DRUG_MECHANISM``: From parent's mechanism table (highest confidence)
        - ``DRUG_MECHANISM_CHILD``: Inherited from child molecule's mechanism
        - ``ACTIVITIES``: Derived from bioassay activity data (lower confidence)
    
    :rtype: pandas.DataFrame
    
    **Examples**
    
    Basic usage - retrieve all molecules from ChEMBL v36::
    
    >>> from alethiotx.artemis.chembl import molecules
    >>> df = molecules(version='36', top_n_activities=1)
    >>> print(f"{len(df)} records, {df['chembl_id'].nunique()} unique molecules")
    >>> print(df[['chembl_id', 'pref_name', 'mesh_heading', 'target_gene_name']].head())
    
    Filter to specific disease and approved drugs only::
    
    >>> df = molecules(version='36')
    >>> lung_cancer = df[df['mesh_heading'] == 'Lung Neoplasms']
    >>> approved = lung_cancer[lung_cancer['phase'] == 4]
    >>> print(approved[['pref_name', 'target_gene_name']].drop_duplicates())
    
    Exclude activity-based targets (mechanism data only)::
    
    >>> df = molecules(version='36', top_n_activities=0)
    >>> print(f"Mechanisms only: {df['target_source'].value_counts()}")
    
    Analyze recent trials (last 6 years)::
    
    >>> from datetime import datetime
    >>> df = molecules(version='36')
    >>> current_year = datetime.now().year
    >>> recent = df[df['trial_year'] >= current_year - 6]
    >>> print(f"Recent trials: {len(recent)} records")
    
    .. note::
        **Parent-Child Normalization**: All molecular forms (salts like 'aspirin sodium',
        formulations like 'aspirin tablet') are normalized to their parent compound ('aspirin').
        This ensures consistent target mapping and prevents double-counting.
    
    .. note::
        **Mechanism-Indication Independence**: A molecule's targets are the same across all its
        indications. For example, if aspirin targets COX1/COX2, these targets apply whether the
        indication is 'Pain' or 'Cardiovascular Disease'. This reflects biological reality - a
        drug's mechanism doesn't change based on what it's prescribed for.
    
    .. note::
        **Clinical Trial ID Explosion**: When a molecule has multiple comma-separated trial IDs
        (e.g., 'NCT001,NCT002'), they are exploded into separate rows. This enables per-trial
        analysis and proper trial counting.
    
    .. warning::
        **Data Volume**: ChEMBL v36 contains hundreds of thousands of molecule-target relationships.
        Full queries may take several minutes and return large DataFrames (>100K rows). Consider
        filtering by version, phase, or disease after loading to reduce memory usage.
    
    .. warning::
        **Activity-Based Targets**: Targets from the ``ACTIVITIES`` table (``target_source='ACTIVITIES'``)
        have lower confidence than mechanism-based targets. They represent bioassay activity but may
        not reflect clinical mechanisms. Set ``top_n_activities=0`` to exclude these if you need
        high-confidence mechanisms only.
    
    .. warning::
        **Requires chembl-downloader**: This function requires the ``chembl-downloader`` package
        to be installed. Install via: ``pip install chembl-downloader``
    
    .. seealso::
        - :func:`infer_nct_year`: Used internally to estimate trial registration years
        - ChEMBL Documentation: https://chembl.gitbook.io/chembl-interface-documentation/
        - ClinicalTrials.gov: https://clinicaltrials.gov/

    """
    
    print("Step 1: Getting all parent molecules with their children's indications...")
    # Get all molecules (parent and children) with clinical trial indications
    # Map everything to parent_molregno
    sql_all_indications = """
    SELECT DISTINCT
        COALESCE(MH.parent_molregno, MD.molregno) AS parent_molregno,
        MD_parent.chembl_id AS parent_chembl_id,
        MD_parent.pref_name AS parent_pref_name,
        DI.mesh_heading,
        DI.mesh_id,
        DI.max_phase_for_ind AS phase,
        IREF.ref_type AS reference_type,
        IREF.ref_id AS clinical_trial_id
    FROM MOLECULE_DICTIONARY MD
    INNER JOIN DRUG_INDICATION DI ON MD.molregno = DI.molregno
    INNER JOIN INDICATION_REFS IREF ON DI.drugind_id = IREF.drugind_id
    LEFT JOIN MOLECULE_HIERARCHY MH ON MD.molregno = MH.molregno
    INNER JOIN MOLECULE_DICTIONARY MD_parent ON COALESCE(MH.parent_molregno, MD.molregno) = MD_parent.molregno
    WHERE IREF.ref_id IS NOT NULL 
        AND IREF.ref_type = 'ClinicalTrials'
    """
    df_indications = chembl_downloader.query(sql_all_indications, version=version)
    print(f"  Found {len(df_indications)} indication records for {df_indications['parent_molregno'].nunique()} parent molecules")
    
    print("\nStep 2: Getting mechanisms for parent molecules...")
    # Get mechanisms from parent molecules directly
    sql_parent_mechanisms = """
    SELECT DISTINCT
        MD.molregno AS parent_molregno,
        TD.chembl_id AS target_chembl_id,
        TD.organism AS target_organism,
        TD.target_type,
        CS_TARGET.accession AS target_uniprot_id,
        COMP_SYN.component_synonym AS target_gene_name,
        DM.mechanism_of_action,
        DM.action_type,
        'DRUG_MECHANISM' AS target_source
    FROM MOLECULE_DICTIONARY MD
    INNER JOIN DRUG_MECHANISM DM ON MD.molregno = DM.molregno
    INNER JOIN TARGET_DICTIONARY TD ON DM.tid = TD.tid
    LEFT JOIN TARGET_COMPONENTS TC ON TD.tid = TC.tid
    LEFT JOIN COMPONENT_SEQUENCES CS_TARGET ON TC.component_id = CS_TARGET.component_id
    LEFT JOIN COMPONENT_SYNONYMS COMP_SYN ON CS_TARGET.component_id = COMP_SYN.component_id AND COMP_SYN.syn_type = 'GENE_SYMBOL'
    WHERE TD.organism = 'Homo sapiens'
    """
    df_parent_mech = chembl_downloader.query(sql_parent_mechanisms, version=version)
    print(f"  Found {len(df_parent_mech)} mechanism records for {df_parent_mech['parent_molregno'].nunique()} parent molecules")
    
    print("\nStep 3: Getting mechanisms from children molecules for parents without mechanisms...")
    # Get mechanisms from children for parents that lack mechanisms
    sql_child_mechanisms = """
    SELECT DISTINCT
        MH.parent_molregno,
        TD.chembl_id AS target_chembl_id,
        TD.organism AS target_organism,
        TD.target_type,
        CS_TARGET.accession AS target_uniprot_id,
        COMP_SYN.component_synonym AS target_gene_name,
        DM.mechanism_of_action,
        DM.action_type,
        'DRUG_MECHANISM_CHILD' AS target_source
    FROM MOLECULE_HIERARCHY MH
    INNER JOIN DRUG_MECHANISM DM ON MH.molregno = DM.molregno
    INNER JOIN TARGET_DICTIONARY TD ON DM.tid = TD.tid
    LEFT JOIN TARGET_COMPONENTS TC ON TD.tid = TC.tid
    LEFT JOIN COMPONENT_SEQUENCES CS_TARGET ON TC.component_id = CS_TARGET.component_id
    LEFT JOIN COMPONENT_SYNONYMS COMP_SYN ON CS_TARGET.component_id = COMP_SYN.component_id AND COMP_SYN.syn_type = 'GENE_SYMBOL'
    WHERE TD.organism = 'Homo sapiens'
        AND NOT EXISTS (
            SELECT 1 FROM DRUG_MECHANISM DM2 WHERE DM2.molregno = MH.parent_molregno
        )
    """
    df_child_mech = chembl_downloader.query(sql_child_mechanisms, version=version)
    print(f"  Found {len(df_child_mech)} child mechanism records for {df_child_mech['parent_molregno'].nunique()} parent molecules")
    
    # Combine mechanisms
    df_mechanisms = pd.concat([df_parent_mech, df_child_mech], ignore_index=True)
    
    print("\nStep 4: Getting activities for parents without any mechanisms...")
    # Find parents without mechanisms
    parents_with_mech = set(df_mechanisms['parent_molregno'].unique())
    all_parents = set(df_indications['parent_molregno'].unique())
    parents_without_mech = all_parents - parents_with_mech
    
    print(f"  Parents needing activities: {len(parents_without_mech)}")
    
    if len(parents_without_mech) > 0:
        molregno_list = ','.join(map(str, parents_without_mech))
        
        sql_activities = f"""
        WITH RankedActivities AS (
            SELECT 
                COALESCE(MH.parent_molregno, ACT.molregno) AS parent_molregno,
                TD.chembl_id AS target_chembl_id,
                TD.organism AS target_organism,
                TD.target_type,
                CS_TARGET.accession AS target_uniprot_id,
                COMP_SYN.component_synonym AS target_gene_name,
                COUNT(ACT.activity_id) as activity_count,
                ROW_NUMBER() OVER (PARTITION BY COALESCE(MH.parent_molregno, ACT.molregno) ORDER BY COUNT(ACT.activity_id) DESC) as rn
            FROM ACTIVITIES ACT
            LEFT JOIN MOLECULE_HIERARCHY MH ON ACT.molregno = MH.molregno
            INNER JOIN ASSAYS A ON ACT.assay_id = A.assay_id
            INNER JOIN TARGET_DICTIONARY TD ON A.tid = TD.tid
            LEFT JOIN TARGET_COMPONENTS TC ON TD.tid = TC.tid
            LEFT JOIN COMPONENT_SEQUENCES CS_TARGET ON TC.component_id = CS_TARGET.component_id
            LEFT JOIN COMPONENT_SYNONYMS COMP_SYN ON CS_TARGET.component_id = COMP_SYN.component_id AND COMP_SYN.syn_type = 'GENE_SYMBOL'
            WHERE COALESCE(MH.parent_molregno, ACT.molregno) IN ({molregno_list})
                AND TD.organism = 'Homo sapiens'
            GROUP BY COALESCE(MH.parent_molregno, ACT.molregno), TD.chembl_id, TD.organism, TD.target_type, CS_TARGET.accession, COMP_SYN.component_synonym
        )
        SELECT 
            parent_molregno,
            target_chembl_id,
            target_organism,
            target_type,
            target_uniprot_id,
            target_gene_name,
            NULL AS mechanism_of_action,
            NULL AS action_type,
            'ACTIVITIES' AS target_source
        FROM RankedActivities
        WHERE rn <= {top_n_activities}
        """
        
        df_activities = chembl_downloader.query(sql_activities, version=version)
        print(f"  Found {len(df_activities)} activity-based targets for {df_activities['parent_molregno'].nunique()} parent molecules")
        
        # Add to mechanisms
        df_mechanisms = pd.concat([df_mechanisms, df_activities], ignore_index=True)
    
    print("\nStep 5: Combining indications with mechanisms...")
    # Cross join indications with mechanisms at parent level (each indication gets all targets)
    df_combined = df_indications.merge(
        df_mechanisms,
        on='parent_molregno',
        how='left'
    )
    
    # Rename columns for final output
    df_combined = df_combined.rename(columns={
        'parent_chembl_id': 'chembl_id',
        'parent_pref_name': 'pref_name'
    })
    
    print(f"  Total combined records: {len(df_combined)}")
    print(f"  Total unique parent molecules: {df_combined['chembl_id'].nunique()}")
    
    # Explode clinical_trial_id column if it contains multiple comma-separated IDs
    if 'clinical_trial_id' in df_combined.columns and df_combined['clinical_trial_id'].notna().any():
        df_combined['clinical_trial_id'] = df_combined['clinical_trial_id'].apply(
            lambda x: x.split(',') if isinstance(x, str) and ',' in x else [x] if x else []
        )
        df_combined = df_combined.explode('clinical_trial_id')
        df_combined['clinical_trial_id'] = df_combined['clinical_trial_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Add inferred year column as integer
        df_combined['trial_year'] = df_combined['clinical_trial_id'].apply(infer_nct_year)
        df_combined['trial_year'] = df_combined['trial_year'].astype('Int64')  # Use nullable integer type

    return df_combined

if __name__ == '__main__':    
    # Query for all drugs with top 1 activity for molecules without mechanism
    print("\nQuerying all molecules with clinical trial data from ChEMBL...")
    print("Including molecules without DRUG_MECHANISM (using top 1 activity)...\n")
    df = molecules(top_n_activities=1)    
    
    output_file = "molecules.csv"
    df.drop_duplicates().to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique molecules: {df['chembl_id'].nunique()}")
    print(f"  From DRUG_MECHANISM: {len(df[df['target_source'] == 'DRUG_MECHANISM'])}")
    print(f"  From DRUG_MECHANISM_CHILD: {len(df[df['target_source'] == 'DRUG_MECHANISM_CHILD'])}")
    print(f"  From ACTIVITIES: {len(df[df['target_source'] == 'ACTIVITIES'])}")
