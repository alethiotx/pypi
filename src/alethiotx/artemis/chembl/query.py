import chembl_downloader
import pandas as pd
import re

def infer_nct_year(nct_id):
    """
    Infer the approximate registration year from a ClinicalTrials.gov NCT identifier.
    NCT IDs follow the format ``NCT########``, where the numeric portion generally increases
    over time. This function uses approximate year ranges based on observed NCT ID
    allocation patterns to estimate when a trial was registered.

    NCT IDs are sequential and follow approximate ranges:
    - ``NCT00000000``-``NCT00999999``: ~1999-2004
    - ``NCT01000000``-``NCT01999999``: ~2005-2011
    - ``NCT02000000``-``NCT02999999``: ~2012-2015
    - ``NCT03000000``-``NCT03999999``: ~2016-2018
    - ``NCT04000000``-``NCT04999999``: ~2019-2021
    - ``NCT05000000``-``NCT05999999``: ~2022-2023
    - ``NCT06000000``+: ~2024+

    :param nct_id: A ClinicalTrials.gov identifier (e.g., ``NCT00000001``)
    :type nct_id: str
    :return: Estimated year of trial registration, or None if the NCT ID is invalid
    :rtype: int or None

    **Example**

    >>> infer_nct_year("NCT00500000")
    2002
    >>> infer_nct_year("NCT03000000")
    2016
    >>> infer_nct_year("invalid")
    None
    .. note::
        This function provides an approximation based on historical NCT ID allocation
        patterns and may not be accurate for all trials. The actual registration date
        should be obtained from the official ClinicalTrials.gov database when precision
        is required.
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
    Query ChEMBL database for parent molecules with clinical trial data and drug indications.
    
    This function normalizes all molecules to their parent forms and aggregates indications from
    both parent and child molecules (e.g., salt forms). Mechanism assignment follows this hierarchy:
    1. Use parent's ``DRUG_MECHANISM`` if available
    2. Inherit from any child's ``DRUG_MECHANISM`` if parent lacks mechanisms
    3. Use top N activities from ``ACTIVITIES`` table if no mechanisms exist
    
    All mechanisms are independent of indication - a molecule has one set of targets that apply
    to all its indications.
    
    :param version: ChEMBL database version to query, defaults to ``36``
    :type version: str, optional
    :param top_n_activities: For molecules without ``DRUG_MECHANISM``, use top N targets from ``ACTIVITIES`` table, defaults to 1
    :type top_n_activities: int, optional
    :return: DataFrame containing parent molecule information with the following key columns:
    - ``chembl_id``: ChEMBL identifier for the parent molecule
    - ``pref_name``: Preferred name of the parent molecule
    - ``mesh_heading``: MeSH term for the indication (aggregated from parent and children)
    - ``mesh_id``: MeSH identifier
    - ``phase``: Clinical trial phase for this indication
    - ``reference_type``: Type of reference (filtered to 'ClinicalTrials')
    - ``clinical_trial_id``: ClinicalTrials.gov identifier(s), exploded if multiple
    - ``target_chembl_id``: ChEMBL identifier for the target
    - ``target_organism``: Target organism (filtered to 'Homo sapiens')
    - ``target_type``: Type of target
    - ``target_uniprot_id``: UniProt accession for the target
    - ``target_gene_name``: Gene symbol for the target
    - ``mechanism_of_action``: Description of the mechanism of action (NULL for activity-derived targets)
    - ``action_type``: Type of action on the target (NULL for activity-derived targets)
    - ``parent_molregno``: Internal molecule registry number of parent
    - ``trial_year``: Inferred year from clinical trial ID (nullable integer)
    - ``target_source``: ``DRUG_MECHANISM``, ``DRUG_MECHANISM_CHILD``, or ``ACTIVITIES``
    :rtype: pandas.DataFrame
    
    .. note::
       All child molecules (salts, formulations) are converted to their parent compound.
       Indications are aggregated from both parent and all children.
    
    .. note::
       Mechanisms are assigned at the parent level and apply to all indications.
       If a parent has no mechanism but children do, the child's mechanism is inherited.
    
    .. note::
       Clinical trial IDs containing multiple comma-separated values are exploded into separate rows.
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
