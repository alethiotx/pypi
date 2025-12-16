import json
import requests
from pandas import DataFrame, read_csv
from ..utils import find_overlapping_genes

def get(search, rif = 'generif'):
   """
   Query the Geneshot API to retrieve gene associations for a search term.

   :param search: The search term to query for gene associations
   :type search: str
   :param rif: The type of reference to use for gene associations, defaults to ``generif``
   :type rif: str, optional
   :return: A DataFrame containing gene associations with columns for gene symbols (index),
            gene_count (number of associations), and rank (relevance ranking)
   :rtype: pandas.DataFrame

   :raises requests.exceptions.RequestException: If the API request fails

   **Example**
   >>> df = get_pathway_genes('cancer')
   >>> print(df.head())

   .. note::
      Gene symbols containing hyphens (-) or underscores (_) are filtered out from the results.

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

def load(date = '2025-11-11', n = 100):
   """
   Retrieve pathway genes for multiple disease types from S3 storage.

   This function reads CSV files containing pathway gene data for various diseases,
   sorts them by gene count and rank, and returns the top N pathways for each disease.

   :param date: Date string in ``YYYY-MM-DD`` format representing the data version,
               defaults to ``2025-11-11``
   :type date: str, optional
   :param n: Number of top pathways to retrieve for each disease, defaults to 100
   :type n: int, optional

   :return: A tuple containing lists of pathway indices for each disease type in the
            following order: (breast, lung, prostate, melanoma, bowel, diabetes,
            cardiovascular)
   :rtype: tuple[list, list, list, list, list, list, list]

   :raises FileNotFoundError: If the specified CSV files do not exist in S3
   :raises ValueError: If the CSV files are malformed or missing required columns

   **Example**
   >>> breast, lung, prostate, melanoma, bowel, diabetes, cardio = load_pathway_genes(
   ...     date='2025-11-11', n=50
   ... )
   >>> len(breast)
   50

   .. note::
   
      The function expects CSV files to be located at
      ``s3://alethiotx-artemis/data/pathway_genes/{date}/{disease}.csv``

   .. note::

      Each CSV file must contain ``gene_count`` and ``rank`` columns for sorting

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

def unique(genes: list, overlap = 1, common_genes = []):
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

    **Example**
    >>> genes = [['GENE1', 'GENE2', 'GENE3'], ['GENE2', 'GENE4'], ['GENE3', 'GENE5']]
    >>> uniquify_pathway_genes(genes, overlap=2)
    [['GENE1'], ['GENE4'], ['GENE5']]
    """
    overlapping_genes = find_overlapping_genes(genes, overlap = overlap, common_genes = common_genes)
    
    res = genes.copy()

    for n, y in enumerate(res):
        res[n] = [x for x in y if x not in overlapping_genes]

    return(res)

if __name__ == '__main__':
   d = get('Breast Cancer')
   print(d)
   breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardiovascular_pg = load(date='2025-12-06', n=300)
   print('Breast Pathway Genes:', breast_pg)