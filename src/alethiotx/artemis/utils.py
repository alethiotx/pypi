"""
**Artemis**: public knowledge graphs enable accessible and scalable drug target discovery.

Github Repository   
-----------------

`GitHub - alethiotx/artemis-paper <https://github.com/alethiotx/artemis-paper>`_
"""

def find_overlapping_genes(genes: list, overlap = 1, common_genes = []):
    """
    Find genes that overlap across multiple gene lists.

    :param genes: A list of gene lists to check for overlapping genes.
    :type genes: list
    :param overlap: Minimum number of lists a gene must appear in to be considered overlapping, defaults to 1.
    :type overlap: int
    :param common_genes: Initial list of common genes to include in the result, defaults to [].
    :type common_genes: list
    :return: A list of genes that appear in more than ``overlap`` lists but not in all lists.
    :rtype: list

    **Example**
    >>> genes_list = [['gene1', 'gene2'], ['gene2', 'gene3'], ['gene2', 'gene4']]
    >>> find_overlapping_genes(genes_list, overlap=1)
    ['gene2']

    .. note::

        Genes that appear in all input lists are excluded from the result by default.
        To include them, modify the condition ``d[i] < len(genes)`` to ``d[i] <= len(genes)``.

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

# run when file is directly executed
if __name__ == '__main__': 
    print("Artemis module. Run tests or examples here.")
    # # 'Breast Cancer', 
    # # 'Lung Cancer',
    # # 'Prostate Cancer',
    # # 'Bowel Cancer',
    # # 'Melanoma',
    # # 'Diabetes Mellitus Type 2',
    # # 'Cardiovascular Disease',
    # df = get_pathway_genes("acute myeloid leukemia")
    # print(df.loc["FLT3", ["gene_count", "rank"]])
    # breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular = load_clinical_scores(date="2025-11-11")
    # print('Clinical scores:\n\n')
    # print([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular])
    # print('All target genes:\n\n')
    # known_targets = get_all_targets([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular])
    # print(known_targets)
    # print('Cut clinical scores:\n\n')
    # print(cut_clinical_scores([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular], lowest_score = 10))
    # breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardiovascular_pg = load_pathway_genes(date='2025-09-15', n=50)
    # print('Uniquified clinical scores:\n\n')
    # print(uniquify_clinical_scores([breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular]))
    # print('Uniquified pathway genes:\n\n')
    # print(uniquify_pathway_genes([breast_pg, lung_pg, prostate_pg, melanoma_pg, bowel_pg, diabetes_pg, cardiovascular_pg]))
    # print(prepare_upset(breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular, mode='ct'))

    # # X is always bigger than y!!!
    # X = DataFrame({
    #     'term1' : [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2], 
    #     'term2' : [1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1, 1, 2, 3, 4, 5, 0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1], 
    #     'term3' : [2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1, 3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2, 2, 3, 4, 5, 6, 5, 7, 2, 9, 2, 1, 3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2], 
    #     'term4' : [3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2, 9, 2, 1, 3, 4, 5, 6, 7, 5, 7, 2, 3, 4, 5, 6, 7, 4, 5, 0, 0, 1, 2, 9, 2, 1, 3, 4, 5, 6, 7, 5, 7, 2], 
    #     'term5' : [4, 5, 6, 7, 8, 6, 5, 7, 2, 9, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 4, 5, 6, 7, 8, 6, 5, 7, 2, 9, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    # })
    # y = DataFrame({
    #     'Target Gene' : ['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6', 'gene7', 'gene8', 'gene9', 'gene10', 'gene11', 'gene12', 'gene13', 'gene14', 'gene15'], 
    #     'Clinical Score' : [1, 1, 1, 4, 5, 10, 5, 3, 2, 5, 1, 100, 200, 300, 400]
    # })
    # # X is always bigger than y!!!
    # X.index = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6', 'gene7', 'gene8', 'gene9', 'gene10', 'gene11', 'gene12', 'gene13', 'gene14', 'gene15', 'gene16', 'gene17', 'gene18', 'gene19', 'gene20', 'gene21', 'gene22', 'gene23', 'gene24', 'gene25', 'gene26', 'gene27', 'gene28', 'gene29', 'gene30', 'gene31', 'gene32', 'gene33', 'gene34', 'gene35', 'gene36', 'gene37', 'gene38', 'gene39', 'gene40', 'gene41', 'gene42', 'gene43', 'gene44']
    # print('\nInput term matrix:')
    # print(X)
    # print('\nInput clinical scores:')
    # print(y)
    # print('\nCheck binning:')
    # res = pre_model(X, y, bins = 5)
    # print(res['y_encoded'])

    # known_targets = ['gene32', 'gene33']

    # res = pre_model(X, y, known_targets = known_targets, bins = 5)
    # print(res['y_encoded'])

    # print('\nResults of cross validation pipeline:')
    # print(cv_pipeline(X, y, n_iterations = 3))
    # print(cv_pipeline(X, y, y_slot = 'y_encoded', scoring = 'accuracy', n_iterations = 3))
