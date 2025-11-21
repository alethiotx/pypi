"""
alethiotx_artemis.kg
====================

This module provides utilities for loading clinical and pathway gene score data from S3,
preparing modeling datasets, filtering overlapping genes, and running cross-validation pipelines
for drug target prediction.

Functions
---------

Data Loading
~~~~~~~~~~~~
.. autosummary::
    load_clinical_scores
    get_pathway_genes

Target Processing
~~~~~~~~~~~~~~~~~
.. autosummary::
    get_all_targets
    cut_clinical_scores
    find_overlapping_genes
    uniquify_clinical_scores
    uniquify_pathway_genes

Model Preparation and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    pre_model
    cv_pipeline
    roc_curve

Notes
-----
This module is designed to work with clinical target data and pathway gene data
stored in S3 buckets with date-based directory structures.

Examples
--------
>>> # Load clinical scores for different disease areas
>>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load_clinical_scores()
>>> 
>>> # Get all unique target genes
>>> all_targets = get_all_targets([breast, lung, prostate])
>>> 
>>> # Load pathway genes
>>> pathway_genes = get_pathway_genes(n=100)
>>> 
>>> # Prepare data for modeling
>>> result = pre_model(X, y, pathway_genes=pathway_genes[0])
>>> 
>>> # Run cross-validation pipeline
>>> scores = cv_pipeline(X, y, n_iterations=10)
"""

from requests import get
from datetime import date
from typing import List
from re import escape, match
import json
import requests
from numpy import mean, log2, linspace, interp, std, minimum, maximum, random
from typing import List
from pandas import DataFrame, Series, concat, options, qcut, read_csv, isna, json_normalize
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm

def indication_pattern(indication: str = 'mpn') -> str:
    """Select an indication regular expression pattern for a given indication.

    It can be then used to filter indication specific clinical trials or approved drugs.

    :param indication: indication. Choose from: ``mpn``, ``bowel``, ``breast``, ``lung``, ``melanoma`` or ``prostate``.

    :return: regular expression pattern corresponding to an input indication.
    """

    pattern = '.*'

    lookup = DataFrame({
        'indication': [
            'Myeloproliferative Neoplasm',  
            'Breast Cancer', 
            'Lung Cancer',
            'Prostate Cancer',
            'Bowel Cancer',
            'Melanoma',
            'Diabetes Mellitus Type 2',
            'Cardiovascular Disease'
        ],
        'regexp': [
            r'.*MPN.*|.*MDS.*|.*[Mm]yeloproliferative.*|.*[Mm]yelofibrosis.*|.*[Pp]olycythemia.*|.*[Tt]hrombocytha?emia.*|.*[Mm]yelodysplastic.*|.*[Nn]eoplasm.*',
            r'.*[Bb]reast.+[Cc]ancer.*|.*[Bb]reast.+[Cc]arcinoma.*|.*[Cc]arcinoma.+[Bb]reast.*',
            r'.*[Cc]arcinoma.+[Bb]ronch.*|.*MDS.*|.*[Ll]ung.+[Cc]ancer.*|.*[Ll]ung.+[Cc]arcinoma.*|.*[Ll]ung.+[Tt]umor.*',
            r'.*[Cc]arcinoma.+[Pp]rostate.*|.*[Pp]rostat.+[Cc]ancer.*|.*[Pp]rostat.+[Cc]arcinoma.*',
            r'.*[Cc]olon.+[Cc]ancer.*|.*[Rr]ectal.+[Cc]ancer.*|.*[Rr]ectal.+[Cc]arcinoma.*',
            r'.*[Mm]elanoma.*|.*[Ss]kin.+[Ll]esion.*',
            r'.*[Tt]ype.+2.*|.*[Tt]ype.+ii.*|.*[Tt]ype-2.*|.*[Tt]ype-ii.*|.*[Tt]2[Dd][Mm].*',
            r'.*cvd.*|.*[Aa]rteriosclero.*|.*[Aa]thero.*|.*[Cc]ardiovascular [Dd]isease.*|.*[Mm]yocardial.*|[Aa]cute [Cc]oronary [Ss]yndrome.+|[Cc]ardi.+|[Cc]oronary.+|[Hh]eart [Dd]isease.+|[Pp]eripheral [Aa]rtery [Dd]isease.+|.*[Vv]ascular [Dd]isease.+'
        ]
    })

    if indication in lookup['indication'].tolist():
        pattern = list(lookup.loc[lookup['indication'] == indication, 'regexp'])[0]
    else: print('Selected indication: ' + indication + ' is not known. Proceeding with the full list of indications.')

    return pattern

def trials(search: str = 'Myeloproliferative Neoplasm', filter_by_regexp: bool = True, fields: List[str] = ['NCTId', 'OverallStatus', 'StudyType', 'Condition', 'InterventionType', 'InterventionName', 'Phase', 'StartDate'], interventional_only: bool = True, intervention_types: List[str] = ['DRUG', 'BIOLOGICAL'], last_6_years: bool = True) -> DataFrame:
    """
    Retrieve and process clinical trials data from ClinicalTrials.gov.
    This function searches for clinical trials based on a medical condition,
    retrieves the data through the ClinicalTrials.gov API, and processes it
    into a clean DataFrame format.
    Parameters
    ----------
    search : str, default='Myeloproliferative Neoplasm'
        The medical condition to search for. Spaces are replaced with '+' characters.
    fields : List[str], default=['NCTId', 'OverallStatus', 'StudyType', 'Condition', 'InterventionType', 'InterventionName', 'Phase', 'StartDate']
        List of fields to retrieve from the API.
    interventional_only : bool, default=True
        If True, only return interventional trials.
    intervention_types : List[str], default=['DRUG', 'BIOLOGICAL']
        Types of interventions to include.
    indication : str, default='mpn'
        Disease indication to filter for, used to create a regex pattern.
    last_6_years : bool, default=True
        If True, only include trials that started within the last 6 years.
    Returns
    -------
    DataFrame
        A pandas DataFrame containing the filtered clinical trials data with columns
        corresponding to the requested fields. Each row represents a unique trial-intervention
        combination after filtering.
    Notes
    -----
    The function performs several data cleaning operations:
    - Combines early phase 1 and phase 0 with phase 1
    - Combines phase 4 and 5 with phase 3
    - Filters trials to include only those matching the disease indication
    - Converts start dates to years
    - Removes duplicate entries
    """
    search_url = search.replace(" ", "+")
    fields = '%2C'.join(fields)

    resp = get('https://www.clinicaltrials.gov/api/v2/studies?query.cond=' + search_url + '&pageSize=1000' + '&markupFormat=legacy' + '&fields=' + fields)
    txt = resp.json()

    # if there are no results
    if txt['studies'] == []:
        print('Your request did not return any data. Please check request spelling/words and try again.')
        return

    res = json_normalize(txt['studies'])

    while 'nextPageToken' in txt:
        resp = get('https://www.clinicaltrials.gov/api/v2/studies?query.cond=' + search_url + '&pageSize=1000'  + '&markupFormat=legacy' + '&pageToken=' + txt['nextPageToken'] + '&fields=' + fields)
        txt = resp.json()
        res = concat([res, json_normalize(txt['studies'])])

    res.rename(columns={
        'protocolSection.identificationModule.nctId': 'NCTId', 
        'protocolSection.statusModule.overallStatus': 'OverallStatus',
        'protocolSection.statusModule.startDateStruct.date': 'StartDate',
        'protocolSection.conditionsModule.conditions': 'Condition',
        'protocolSection.designModule.studyType': 'StudyType',
        'protocolSection.designModule.phases': 'Phase', 
        'protocolSection.armsInterventionsModule.interventions': 'Interventions'
    }, inplace=True)

    # remove trials with no intervention name and no phase
    res = res[~res["Interventions"].isna()]
    res = res[~res["Phase"].isna()]

    # filter empty phases and select maximum phase (last value in the list, if there are multiple phases)
    res = res[res["Phase"].str.len() != 0]
    res['Phase'] = res['Phase'].apply(lambda x: x if len(x) == 1 else x.pop())

    # unlist phase column
    res = res.explode('Phase')
    res = res[res['Phase'] != 'NA']

    # filter and rename phases
    res.loc[res['Phase'] == 'PHASE0', 'Phase'] = 'PHASE1'
    res.loc[res['Phase'] == 'EARLY_PHASE1', 'Phase'] = 'PHASE1'
    res.loc[res['Phase'] == 'PHASE4', 'Phase'] = 'PHASE3'
    res.loc[res['Phase'] == 'PHASE5', 'Phase'] = 'PHASE3'

    # unlist interventions
    res = res.explode('Interventions')
    res['InterventionType'] = res['Interventions'].apply(lambda x: x['type'])
    res['InterventionName'] = res['Interventions'].apply(lambda x: x['name'])
    res = res.drop(['Interventions'], axis=1)

    # unlist items in columns
    res = res.explode('NCTId')
    res = res.explode('OverallStatus')
    res = res.explode('StudyType')
    res = res.explode('Condition')
    res = res.explode('StartDate')

    # select only interventional trials
    if interventional_only:
        res = res[res['StudyType'] == 'INTERVENTIONAL']
        res = res[(res['InterventionType'].isin(intervention_types))]

    # filter by disease related indications
    if filter_by_regexp:
        regexp_pattern = indication_pattern(search)
        res = res[res['Condition'].str.match(regexp_pattern)]

    res = res.drop(['Condition'], axis=1)
    res = res.drop_duplicates()
    
    # convert date to year
    res['StartDate'] = res['StartDate'].apply(lambda x: int(str(x)[0:4]) if not isna(x) else x)

    if last_6_years:
        current_year = int(date.today().strftime("%Y"))
        res = res[res['StartDate'] >= current_year - 6]

    # remove duplicates
    res = res.drop_duplicates()

    return(res)

def drugbank(trials: DataFrame, date: str = '2025-03-26', pharm_action = True, approved_by = 'US') -> DataFrame:
    """
    Match clinical trials data with DrugBank drug information and filter based on specified criteria.

    This function takes clinical trial data and matches interventions with drugs from
    the DrugBank database using drug synonyms. It filters the drugs based on targets,
    synonym length, pharmacological action, specified indication, and approval status.

    Parameters
    ----------
    trials : DataFrame
        DataFrame containing clinical trial information with an 'InterventionName' column.
    date : str, optional
        Date when DrugBank data was obtained, default is '2025-03-26'.
    pharm_action : bool, optional
        If True, only includes drugs with confirmed pharmacological action, default is True. This filter makes sure that non-pharmacological targets (chemo drugs and PD1/PD1L immuno therapy drugs) are not included in the final result.
    approved_by : str, optional
        Approval status to filter for, one of 'US', 'Other', or 'Both', default is 'US'.
        Determines which approval status is included in the 'Approved' column.

    Returns
    -------
    DataFrame
        A DataFrame containing matched clinical trial and DrugBank information with
        the following key columns:
        - Original clinical trial data
        - DrugBank drug information including target genes
        - 'Approved' column (1 for approved, 0 for not approved based on specified criteria)

    Notes
    -----
    The function performs matching by converting drug synonyms to regular expressions
    and searching for matches in the intervention names from clinical trials.
    """
    # load the drugbank data
    db = read_csv('s3://alethiotx-artemis/data/drugbank/' + date + '/webdata.csv')

    # filter out drugs with no targets
    db = db[db['Target Name'] != 'Not Available']
    db = db[~db['Target Name'].isna()]

    # filter short and long synonyms
    db['Synonyms_length'] = db['Synonyms'].apply(lambda x: len(str(x)))
    db = db[(db['Synonyms_length'] > 5) & (db['Synonyms_length'] <= 30)]
    db = db.drop(columns=['Synonyms_length'])

    # ASSIGN DRUGBANK IDS
    # prepare for matching clinical trials interventions with drugbank synonyms
    # Some drug synonyms likely contain characters like parentheses, brackets, or other regex special characters that need to be escaped.
    db['regex'] = db['Synonyms'].apply(lambda x: '.*' + escape(str(x).lower()) + '.*')
    trials['intervention_lower'] = trials['InterventionName'].apply(str.lower)

    # match clinical trials interventions with drugbank synonyms
    idx = [(i,j) for i,r in enumerate(db['regex']) for j,v in enumerate(trials['intervention_lower']) if match(r,v)]
    # define matched indexes
    df1_idx, df2_idx = zip(*idx)
    # select and reorder original data frames by matced indexes
    df1 = db.iloc[list(df1_idx),:].reset_index(drop=True)
    df2 = trials.iloc[list(df2_idx),:].reset_index(drop=True)
    # concatenate data frames
    res = concat([df2[df2.columns[:-1]], df1[df1.columns[:-1]]],axis=1)

    if pharm_action:
        res = res[res['Pharmacological Action'] == 'Yes']

    res.rename(columns={
        'Target Name': 'Target Gene',
    }, inplace=True)

    if approved_by == 'US':
        res['Approved'] = res['US Approved']
        res['Approved'] = res['Approved'].apply(lambda x: 1 if x == 'YES' else 0)
    elif approved_by == 'Other':
        res['Approved'] = res['Other Approved']
        res['Approved'] = res['Approved'].apply(lambda x: 1 if x == 'YES' else 0)
    elif approved_by == 'Both':
        res['Approved'] = res[['US Approved', 'Other Approved']].apply(lambda x: 1 if all(x == 'YES') else 0, axis = 1)
    res = res.drop(columns=['US Approved', 'Other Approved'])

    return res

def drugscores(trials: DataFrame, include_approved: bool = True) -> DataFrame:
    """
    Compute per-target clinical development scores from a trials table.

    For each target gene, this function:
    - Counts the number of unique trial-phase combinations per phase (1–3), using
        unique rows of ['NCTId', 'Phase'] to avoid double-counting the same trial/phase.
    - Computes a Phase Score as the sum of phase numbers across unique trial/phase
        pairs (e.g., Phase 1 -> 1, Phase 2 -> 2, Phase 3 -> 3).
    - Counts the number of distinct approved drugs (unique 'DrugBank ID' where
        'Approved' == 1).
    - Combines the above into a result indexed by target, with columns:
        ['# Phase 1', '# Phase 2', '# Phase 3', 'Phase Score', '# Approved Drugs'].
    - Calculates a Drug Score as:
            Phase Score + 20 * (# Approved Drugs) if include_approved is True,
            otherwise just Phase Score.
    - Sorts results by Drug Score (descending).

    Parameters
    ----------
    trials : pandas.DataFrame
            Clinical trials data containing at least the following columns:
            - 'Target Gene': target identifier used for grouping.
            - 'NCTId': clinical trial identifier.
            - 'Phase': trial phase as a string ending in the phase number (e.g., 'Phase 1').
            - 'Approved': indicator (1/0) whether the drug is approved.
            - 'DrugBank ID': identifier for deduplicating approved drugs per target.
    include_approved : bool, default True
            Whether to include the number of approved drugs in the Drug Score
            with a weight of 20 per approved drug.

    Returns
    -------
    pandas.DataFrame
            DataFrame indexed by target gene with the following columns:
            - '# Phase 1', '# Phase 2', '# Phase 3': counts of unique trials per phase.
            - 'Phase Score': sum of phase numbers over unique trial/phase pairs.
            - '# Approved Drugs': count of distinct approved drugs per target.
            - 'Drug Score': overall score used for ranking.
            Rows are sorted by 'Drug Score' in descending order. Missing phase counts
            are treated as zero.

    Notes
    -----
    - The function prints the overall distribution of 'Phase' values to stdout.
    - Expects 'Phase' strings to end with an integer (e.g., '...1', '...2', '...3').

    Raises
    ------
    KeyError
            If required columns are missing.
    ValueError
            If 'Phase' values cannot be parsed to extract a numeric phase.
    """
    # calculate number of trials in each phase
    phases = DataFrame(trials.groupby(by='Target Gene')[trials.columns].apply(lambda x: x[['NCTId', 'Phase']].drop_duplicates()['Phase'].value_counts())).unstack()
    phases[phases.isna()] = 0
    # calculate phase scores
    print(trials['Phase'].value_counts())
    phase_scores = DataFrame(trials.groupby(by='Target Gene')[trials.columns].apply(lambda x: sum(x[['NCTId', 'Phase']].drop_duplicates()['Phase'].str[-1].astype(int))))
    # calculate number of approved drugs for each target
    n_approved_drugs = DataFrame(trials.groupby(by='Target Gene')[trials.columns].apply(lambda x: len(x.loc[x['Approved'] == 1, 'DrugBank ID'].drop_duplicates())))
    # concatenate all data frames
    res = concat([phases, phase_scores, n_approved_drugs], axis = 1)
    res.columns = ['# Phase 1', '# Phase 2', '# Phase 3', 'Phase Score', '# Approved Drugs']
    # calculate a total drug score
    if include_approved:
        res['Drug Score'] = res['Phase Score'] + res['# Approved Drugs'] * 20
    else:
        res['Drug Score'] = res['Phase Score']
    # sort by drug score
    res = res.sort_values(by = 'Drug Score', ascending=False)

    return(res)

def geneshot(search, rif = 'generif'):
    """
    Query Ma’ayan Lab’s GeneShot API for genes associated with a search term and return a cleaned pandas DataFrame.

    This function sends a POST request to https://maayanlab.cloud/geneshot/api/search with the provided term
    and RIF source, parses the JSON response into a DataFrame, splits the API’s `gene_count` field into two
    columns (`gene_count` and `rank`), and filters out entries whose index contains '-' or '_', retaining
    canonical gene-like identifiers.

    Parameters
    ----------
    search : str
        Free-text query term (e.g., disease, phenotype, pathway) to search in GeneShot.
    rif : str, optional
        Reference information source used by the API to rank associations (as supported by GeneShot).
        Defaults to 'generif'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by gene identifier with at least:
          - gene_count (int): number of co-mentions/hits for the gene returned by the API.
          - rank (int): the rank assigned by the API.
        Additional columns provided by the API are preserved.

    Raises
    ------
    requests.exceptions.RequestException
        If the HTTP request fails or times out.
    json.JSONDecodeError
        If the response body cannot be parsed as JSON.
    KeyError
        If the expected 'gene_count' field is missing from the API response.
    TypeError or ValueError
        If the structure of 'gene_count' is not the expected sequence of length 2.

    Notes
    -----
    - Requires `requests`, `json`, and `pandas` (DataFrame).
    - Rows with indices containing '-' or '_' are removed to reduce non-canonical identifiers.
    - The exact schema of the API response may change; downstream code should be robust to extra columns.

    Examples
    --------
    >>> df = geneshot("acute myeloid leukemia")
    >>> df.loc["FLT3", ["gene_count", "rank"]]
    """
    GENESHOT_URL = 'https://maayanlab.cloud/geneshot/api/search'
    payload = {"rif": rif, "term": search}
    response = requests.post(GENESHOT_URL, json=payload)
    data = json.loads(response.text)
    d = DataFrame(data)
    d['rank'] = d['gene_count'].apply(lambda x: x[1])
    d['gene_count'] = d['gene_count'].apply(lambda x: x[0])
    d = d[(~d.index.str.contains('-')) & (~d.index.str.contains('_'))]
    return d

def load_clinical_scores(date = '2025-11-11'):
    """
    Retrieve clinical scores for multiple disease types from S3 storage.

    This function reads CSV files containing clinical target data for various diseases
    from an S3 bucket, organized by date.

    :param date: The date string used to construct the S3 file paths, defaults to '2025-11-11'
    :type date: str, optional
    :return: A tuple containing DataFrames for breast, lung, prostate, melanoma, bowel,
             diabetes, and cardiovascular clinical scores (in that order)
    :rtype: tuple of pandas.DataFrame
    :raises FileNotFoundError: If any of the CSV files do not exist at the specified S3 paths
    :raises ValueError: If the CSV files cannot be parsed correctly

    .. note::
       All CSV files are expected to be located in the S3 bucket at:
       s3://alethiotx-artemis/data/clinical_targets/{date}/{disease}.csv

    .. warning::
       The default date '2025-11-11' is set to a future date. Ensure data exists
       for the specified date before calling this function.

    Example
    -------
    >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load_clinical_scores('2025-11-11')
    >>> print(breast.shape)
    (100, 5)
    """
    breast = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/breast.csv")
    lung = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/lung.csv")
    prostate = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/prostate.csv")
    melanoma = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/melanoma.csv")
    bowel = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/bowel.csv")
    diabetes = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/diabetes.csv")
    cardiovascular = read_csv("s3://alethiotx-artemis/data/clinical_targets/" + date + "/cardiovascular.csv")

    return(breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular)

def get_all_targets(scores: list):
    """
    Extract all unique target genes from a list of score dictionaries.

    :param scores: A list of dictionaries, where each dictionary contains a 'Target Gene' key
                   with a value that can be converted to a list using tolist() method
                   (typically a pandas Series or similar object)
    :type scores: list
    :return: A list of all unique target genes found across all dictionaries in the input list
    :rtype: list

    :Example:

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

def cut_clinical_scores(scores: list, lowest_score = 0):
    """
    Filter clinical scores by removing entries below a threshold.

    This function creates a copy of the input scores list and filters each DataFrame
    within it to retain only rows where the 'Drug Score' exceeds the specified
    lowest_score threshold.

    :param scores: List of DataFrames containing clinical score data with a 'Drug Score' column
    :type scores: list
    :param lowest_score: Minimum score threshold for filtering (default: 0)
    :type lowest_score: int or float
    :return: List of filtered DataFrames with scores above the threshold
    :rtype: list

    :Example:

    >>> import pandas as pd
    >>> scores = [pd.DataFrame({'Drug Score': [0.5, 1.5, 2.0]})]
    >>> filtered = cut_clinical_scores(scores, lowest_score=1.0)
    >>> filtered[0]
        Drug Score
    1         1.5
    2         2.0
    """
    res = scores.copy()

    for n, d in enumerate(res):
        res[n] = d[d['Drug Score'] > lowest_score]

    return(res)

def get_pathway_genes(date = '2025-11-11', n = 100):
    """
    Retrieve pathway genes for multiple disease types from S3 storage.

    This function reads CSV files containing pathway gene data for various diseases,
    sorts them by gene count and rank, and returns the top N pathways for each disease.

    :param date: Date string in 'YYYY-MM-DD' format representing the data version,
                 defaults to '2025-11-11'
    :type date: str, optional
    :param n: Number of top pathways to retrieve for each disease, defaults to 100
    :type n: int, optional

    :return: A tuple containing lists of pathway indices for each disease type in the
             following order: (breast, lung, prostate, melanoma, bowel, diabetes,
             cardiovascular)
    :rtype: tuple[list, list, list, list, list, list, list]

    :raises FileNotFoundError: If the specified CSV files do not exist in S3
    :raises ValueError: If the CSV files are malformed or missing required columns

    .. note::
       The function expects CSV files to be located at
       's3://alethiotx-artemis/data/pathway_genes/{date}/{disease}.csv'

    .. note::
       Each CSV file must contain 'gene_count' and 'rank' columns for sorting

    :Example:

    >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = get_pathway_genes(
    ...     date='2025-11-11', n=50
    ... )
    >>> len(breast)
    50
    """
    breast = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/breast.csv', index_col = 0)
    breast = breast.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()
    lung = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/lung.csv', index_col = 0)
    lung = lung.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()
    bowel = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/bowel.csv', index_col = 0)
    bowel = bowel.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()
    prostate = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/prostate.csv', index_col = 0)
    prostate = prostate.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()
    melanoma = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/melanoma.csv', index_col = 0)
    melanoma = melanoma.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()
    diabetes = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/diabetes.csv', index_col = 0)
    diabetes = diabetes.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()
    cardiovascular = read_csv('s3://alethiotx-artemis/data/pathway_genes/' + date + '/cardiovascular.csv', index_col = 0)
    cardiovascular = cardiovascular.sort_values(['gene_count', 'rank'], ascending=False).head(n).sort_index().index.tolist()

    return (breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular)

def find_overlapping_genes(genes: list, overlap = 1, common_genes = []):
    """
    Find genes that overlap across multiple gene lists.

    :param genes: A list of gene lists to check for overlapping genes.
    :type genes: list
    :param overlap: Minimum number of lists a gene must appear in to be considered overlapping (default: 1).
    :type overlap: int
    :param common_genes: Initial list of common genes to include in the result (default: []).
    :type common_genes: list
    :return: A list of genes that appear in more than 'overlap' lists but not in all lists.
    :rtype: list

    .. note::
        Genes that appear in all input lists are excluded from the result by default.
        To include them, modify the condition ``d[i] < len(genes)`` to ``d[i] <= len(genes)``.

    :Example:

    >>> genes_list = [['gene1', 'gene2'], ['gene2', 'gene3'], ['gene2', 'gene4']]
    >>> find_overlapping_genes(genes_list, overlap=1)
    ['gene2']
    """
    d = {}
    overlapping_genes = common_genes.copy()

    for i in [x for y in genes for x in y]:
        if i in d.keys():
            d[i]=d[i]+1
        else:
            d[i]=1
    for i in d.keys():
        if d[i]>overlap and d[i] < len(genes): # the second condition excludes genes present in all lists, if you want to include those, change < to <=, or remove the condition entirely
            overlapping_genes.append(i)
    return overlapping_genes

def uniquify_clinical_scores(scores: list, overlap = 1, common_genes = []):
    """
    Remove overlapping genes from clinical score dataframes to ensure uniqueness.
    This function processes a list of dataframes containing clinical scores and removes
    genes that appear in multiple dataframes above a specified overlap threshold.
    :param scores: List of dataframes, each containing a 'Target Gene' column with gene identifiers
    :type scores: list
    :param overlap: Minimum number of dataframes a gene must appear in to be considered overlapping, defaults to 1
    :type overlap: int, optional
    :param common_genes: Additional list of genes to always consider as overlapping regardless of frequency, defaults to []
    :type common_genes: list, optional
    :return: List of dataframes with overlapping genes removed from each dataframe
    :rtype: list
    .. note::
        The function uses :func:`find_overlapping_genes` to identify genes that should be removed.
    .. warning::
        This function modifies copies of the input dataframes. The original dataframes remain unchanged.
    :Example:
    >>> df1 = pd.DataFrame({'Target Gene': ['BRCA1', 'TP53', 'EGFR']})
    >>> df2 = pd.DataFrame({'Target Gene': ['TP53', 'KRAS', 'MYC']})
    >>> result = uniquify_clinical_scores([df1, df2], overlap=2)
    >>> # TP53 will be removed from both dataframes as it appears in 2 or more
    """
    genes = []
    for n, d in enumerate(scores):
        genes.append(d['Target Gene'].tolist())

    overlapping_genes = find_overlapping_genes(genes, overlap = overlap, common_genes = common_genes)
    
    res = scores.copy()

    for n, d in enumerate(res):
        res[n] = d[~d['Target Gene'].isin(overlapping_genes)]
    
    return(res)


def uniquify_pathway_genes(genes: list, overlap = 1, common_genes = []):
    """
    Remove overlapping genes from pathway gene lists.
    This function identifies genes that appear in multiple pathways (based on the
    specified overlap threshold) and removes them from each pathway's gene list,
    returning uniquified pathway gene lists.
    :param genes: A list of gene lists, where each inner list represents genes
                  associated with a particular pathway.
    :type genes: list
    :param overlap: The minimum number of pathways a gene must appear in to be
                    considered overlapping and removed. Defaults to 1.
    :type overlap: int, optional
    :param common_genes: A list of genes that should be considered as commonly
                         overlapping regardless of their occurrence count.
    :type common_genes: list, optional
    :return: A copy of the input gene lists with overlapping genes removed from
             each pathway.
    :rtype: list
    :Example:
    >>> genes = [['GENE1', 'GENE2', 'GENE3'], ['GENE2', 'GENE4'], ['GENE3', 'GENE5']]
    >>> uniquify_pathway_genes(genes, overlap=2)
    [['GENE1'], ['GENE4'], ['GENE5']]
    """
    overlapping_genes = find_overlapping_genes(genes, overlap = overlap, common_genes = common_genes)
    
    res = genes.copy()

    for n, y in enumerate(res):
        res[n] = [x for x in y if x not in overlapping_genes]

    return(res)

def pre_model(X: DataFrame, y: DataFrame, pathway_genes: List = [], known_targets: List = [], term_num = None, bins: int = 3, rand_seed: int = 12345) -> dict:
    """
    Prepare and preprocess data for machine learning model training.

    This function processes knowledge graph (KG) features and clinical scores to create
    training datasets for drug target prediction models. It handles positive targets,
    negative samples, and pathway genes, then returns formatted features and labels.

    :param X: Knowledge graph features with genes as columns
    :type X: DataFrame
    :param y: Clinical data containing target genes and drug scores
    :type y: DataFrame
    :param pathway_genes: List of pathway-associated genes to include as a separate class
    :type pathway_genes: List, optional
    :param known_targets: List of known target genes to exclude from negative sampling
    :type known_targets: List, optional
    :param term_num: Number of random KG features to sample; if None, uses all features
    :type term_num: int, optional
    :param bins: Number of bins for discretizing positive target drug scores
    :type bins: int, optional
    :param rand_seed: Random seed for reproducibility
    :type rand_seed: int, optional
    :return: Dictionary containing:
        - 'X': Feature matrix (KG features) for modeling
        - 'y': Continuous drug scores (log2-transformed)
        - 'y_encoded': Categorical labels with binned targets and pathway genes
        - 'y_binary': Binary labels (1 for targets/pathway genes, 0 for non-targets)
    :rtype: dict

    :raises Exception: May raise exceptions during quantile binning if insufficient unique values exist

    .. note::
        - Drug scores are log2-transformed as log2(score + 1)
        - Positive targets are binned into categories: 'target_0', 'target_1', etc.
        - Negative samples are randomly selected to match the number of positive + pathway genes
        - Pathway genes are assigned a drug score of 1 and labeled as 'pathway_gene'
        - The function sets pandas chained assignment warnings to None
    """
    options.mode.chained_assignment = None  # default='warn'
    random.seed(rand_seed)
    # prepare KG features
    if term_num:
        X = X.sample(n = term_num, axis = 1, random_state=rand_seed)
    # prepare clinical scores
    y = y[['Target Gene', 'Drug Score']]
    y['Drug Score'] = log2(y['Drug Score'] + 1)
    y.index = y['Target Gene']
    y = y.drop(columns=['Target Gene'])
    # merge clinical scores and KG features
    y = y.join(X, how = 'right')
    # add negative targets as the number of positive
    y_pos = y[~y['Drug Score'].isna()]
    try:
        y_pos['Drug Score Binned'] = qcut(y_pos['Drug Score'], q=bins, labels=['target_' + str(x) for x in list(range(bins))])
    except:
        y_pos['Drug Score Binned'] = "target"
        print("Binning of cancer targets didn't work, using only one bin!")
    # prepare pathway genes
    y_pg = DataFrame()
    if pathway_genes:
        pathway_genes = [g for g in pathway_genes if g not in y_pos.index.tolist() and g not in known_targets]
        y_pg = y[y.index.isin(pathway_genes)]
        y_pg['Drug Score'] = 1
        y_pg['Drug Score Binned'] = 'pathway_gene'
    else:
        print("No pathway genes were provided!")
    y_neg = y[(y['Drug Score'].isna()) & (~y.index.isin(pathway_genes)) & (~y.index.isin(known_targets))].sample(y_pos.shape[0] + y_pg.shape[0], random_state=rand_seed)
    y_neg['Drug Score'] = -1
    y_neg['Drug Score Binned'] = 'not_target'
    y = concat([y_pos, y_neg, y_pg])
    # shuffle targets
    y = y.sample(frac=1, random_state=rand_seed)

    # CREATE OBJECTS FOR MODELLING
    # create X for modelling
    X_out = y.iloc[:,1:-1]
    X_out.index = y.index
    # create y for modelling
    y_out = y['Drug Score']
    y_out.index = y.index
    y_encoded = y['Drug Score Binned']
    y_encoded.index = y.index
    # binarise y for binary classification
    y_binary = (y_out > 0).astype(int)

    return({
        'X': X_out,
        'y': y_out,
        'y_encoded': y_encoded,
        'y_binary': y_binary
    })

def cv_pipeline(X: DataFrame, y: DataFrame, y_slot = 'y_binary', bins: int = 3, pathway_genes: List = [], classifier: RandomForestClassifier = RandomForestClassifier(), cv: StratifiedKFold = StratifiedKFold(), scoring = 'roc_auc', n_iterations = 10, shuffle_scores = False) -> List:
    """
    Perform cross-validation pipeline for classification tasks.

    This function executes a cross-validation pipeline over multiple iterations,
    preprocessing data and evaluating a classifier using specified scoring metrics.

    :param X: Input features DataFrame
    :type X: DataFrame
    :param y: Target variable DataFrame
    :type y: DataFrame
    :param y_slot: Column name in the preprocessed result to use as target variable, defaults to 'y_binary'
    :type y_slot: str, optional
    :param bins: Number of bins for data preprocessing, defaults to 3
    :type bins: int, optional
    :param pathway_genes: List of pathway genes to be used in preprocessing, defaults to []
    :type pathway_genes: List, optional
    :param classifier: Classifier instance to use for cross-validation, defaults to RandomForestClassifier()
    :type classifier: RandomForestClassifier, optional
    :param cv: Cross-validation splitting strategy, defaults to StratifiedKFold()
    :type cv: StratifiedKFold, optional
    :param scoring: Scoring metric for evaluation, defaults to 'roc_auc'
    :type scoring: str, optional
    :param n_iterations: Number of iterations to run the pipeline, defaults to 10
    :type n_iterations: int, optional
    :param shuffle_scores: Whether to shuffle target variable for permutation testing, defaults to False
    :type shuffle_scores: bool, optional
    :return: List of cross-validation scores from each iteration
    :rtype: List
    """
    score = []
    for i in range(n_iterations):
        res = pre_model(X, y, bins = bins, pathway_genes = pathway_genes, rand_seed = i)
        if shuffle_scores:
            score.append(mean(cross_val_score(classifier, res['X'], res[y_slot].sample(frac=1), scoring=scoring, cv = cv)))
        else:
            score.append(mean(cross_val_score(classifier, res['X'], res[y_slot], scoring=scoring, cv = cv)))

    return(score)

def roc_curve(X: DataFrame, y: Series, n_splits = 5, classifier: str = 'rf', random_state: int = 1234) -> float:
    """
    Generate and plot ROC curves using k-fold cross-validation.

    This function performs stratified k-fold cross-validation to generate ROC curves
    for a given classifier and dataset. It plots individual fold ROC curves along with
    the mean ROC curve and standard deviation bands.

    :param X: Feature matrix containing the independent variables
    :type X: DataFrame
    :param y: Target vector containing the dependent variable (binary classification)
    :type y: Series
    :param n_splits: Number of folds for cross-validation, defaults to 5
    :type n_splits: int, optional
    :param classifier: Type of classifier to use. Options: 'rf' (Random Forest) or 'svm' (Support Vector Machine), defaults to 'rf'
    :type classifier: str, optional
    :param random_state: Random seed for reproducibility, defaults to 1234
    :type random_state: int, optional
    :return: Mean area under the ROC curve (AUC) across all folds
    :rtype: float
    :raises ValueError: If classifier parameter is not 'rf' or 'svm'

    .. note::
        The function displays a matplotlib plot showing individual fold ROC curves,
        mean ROC curve, and ±1 standard deviation bands.

    .. seealso::
        :class:`sklearn.ensemble.RandomForestClassifier`
        :class:`sklearn.svm.SVC`
        :class:`sklearn.model_selection.StratifiedKFold`

    :Example:

    >>> mean_auc = roc_curve(X_train, y_train, n_splits=10, classifier='rf')
    >>> print(f"Mean AUC: {mean_auc:.3f}")
    """
    # define k-fold
    cv = StratifiedKFold(n_splits = n_splits)
    # define classifier
    if classifier == 'rf':
        classifier = RandomForestClassifier()
    elif classifier == 'svm':
        classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)
    else:
        raise ValueError('Wrong classifier parameter! Only `rf` or `svm` can be accepted.')
    # prepare for fold iterations
    tprs = []
    aucs = []
    mean_fpr = linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    # iterate through the folds
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train,:], y.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X.iloc[test,:],
            y.iloc[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    mean_tpr = mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = std(aucs)
    # plot the results
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    std_tpr = std(tprs, axis=0)
    tprs_upper = minimum(mean_tpr + std_tpr, 1)
    tprs_lower = maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label)",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
    return(mean_auc)

# run when file is directly executed
if __name__ == '__main__':
    # 'Myeloproliferative Neoplasm',  
    # 'Breast Cancer', 
    # 'Lung Cancer',
    # 'Prostate Cancer',
    # 'Bowel Cancer',
    # 'Melanoma',
    # 'Diabetes Mellitus Type 2',
    # 'Cardiovascular Disease',
    trs = trials(search="Breast Cancer")
    print(trs)
    db = drugbank(trs)
    print(db)
    scores = drugscores(db)
    print(scores)
    # df = geneshot("acute myeloid leukemia")
    # print(df.loc["FLT3", ["gene_count", "rank"]])
    breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular = load_clinical_scores(date="2025-09-15")
    print('Clinical scores:\n\n')
    print([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular])
    print('All target genes:\n\n')
    known_targets = get_all_targets([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular])
    print(known_targets)
    print('Cut clinical scores:\n\n')
    print(cut_clinical_scores([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular], lowest_score = 10))
    breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardiovascular_pg = get_pathway_genes(date='2025-09-15', n=50)
    print('Uniquified clinical scores:\n\n')
    print(uniquify_clinical_scores([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular]))
    print('Uniquified pathway genes:\n\n')
    print(uniquify_pathway_genes([breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardiovascular_pg]))

    # X is always bigger than y!!!
    X = DataFrame({
        'term1' : [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2], 
        'term2' : [1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1], 
        'term3' : [2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1, 3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2, 2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1, 3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2], 
        'term4' : [3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2, 9, 2, 1, 3, 4, 5, 6, 7, 5, 7, 2, 3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2, 9, 2, 1, 3, 4, 5, 6, 7, 5, 7, 2], 
        'term5' : [4, 5, 6, 7, 8, 6, 5, 7, 2, 9, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 4, 5, 6, 7, 8, 6, 5, 7, 2, 9, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    })
    y = DataFrame({
        'Target Gene' : ['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6', 'gene7', 'gene8', 'gene9', 'gene10', 'gene11', 'gene12', 'gene13', 'gene14', 'gene15'], 
        'Drug Score' : [1, 1, 1, 4, 5, 10, 5, 3, 2, 5, 1, 100, 200, 300, 400]
    })
    # X is always bigger than y!!!
    X.index = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6', 'gene7', 'gene8', 'gene9', 'gene10', 'gene11', 'gene12', 'gene13', 'gene14', 'gene15', 'gene16', 'gene17', 'gene18', 'gene19', 'gene20', 'gene21', 'gene22', 'gene23', 'gene24', 'gene25', 'gene26', 'gene27', 'gene28', 'gene29', 'gene30', 'gene31', 'gene32', 'gene33', 'gene34', 'gene35', 'gene36', 'gene37', 'gene38', 'gene39', 'gene40', 'gene41', 'gene42', 'gene43', 'gene44']
    print('\nInput term matrix:')
    print(X)
    print('\nInput clinical scores:')
    print(y)
    print('\nCheck binning:')
    res = pre_model(X, y, bins = 5)
    print(res['y_encoded'])

    known_targets = ['gene32', 'gene33']

    res = pre_model(X, y, known_targets = known_targets, bins = 5)
    print(res['y_encoded'])

    # print('\nResults of cross validation pipeline:')
    # print(cv_pipeline(X, y, n_iterations = 3))
    # print(cv_pipeline(X, y, y_slot = 'y_encoded', scoring = 'accuracy', n_iterations = 3))

    # roc_curve(res['X'], res['y_binary'], n_splits=5)