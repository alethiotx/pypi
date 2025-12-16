from typing import List
from numpy import mean, log2, random
from pandas import DataFrame, concat, options, qcut
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def prepare(X: DataFrame, y: DataFrame, pathway_genes: List = [], known_targets: List = [], term_num = None, bins: int = 3, rand_seed: int = 12345) -> dict:
    """
    Prepare and preprocess data for machine learning model training.

    This function processes knowledge graph (KG) features and clinical scores to create
    training datasets for drug target prediction models. It handles positive targets,
    negative samples, and pathway genes, then returns formatted features and labels.

    :param X: Knowledge graph features with genes as columns
    :type X: DataFrame
    :param y: Clinical data containing target genes and clinical scores
    :type y: DataFrame
    :param pathway_genes: List of pathway-associated genes to include as a separate class
    :type pathway_genes: List, optional
    :param known_targets: List of known target genes to exclude from negative sampling
    :type known_targets: List, optional
    :param term_num: Number of random KG features to sample; if None, uses all features
    :type term_num: int, optional
    :param bins: Number of bins for discretizing positive target clinical scores
    :type bins: int, optional
    :param rand_seed: Random seed for reproducibility
    :type rand_seed: int, optional
    :return: Dictionary containing:
    * ``X``: Feature matrix (KG features) for modeling
    * ``y``: Continuous clinical scores (log2-transformed)
    * ``y_encoded``: Categorical labels with binned targets and pathway genes
    * ``y_binary``: Binary labels (1 for targets/pathway genes, 0 for non-targets)
    :rtype: dict

    :raises Exception: May raise exceptions during quantile binning if insufficient unique values exist

    .. note::

        - clinical scores are log2-transformed as ``log2(score + 1)``
        - Positive targets are binned into categories: ``target_0``, ``target_1``, etc.
        - Negative samples are randomly selected to match the number of positive + pathway genes
        - Pathway genes are assigned a clinical score of 1 and labeled as ``pathway_gene``
        - The function sets pandas chained assignment warnings to None
    """
    options.mode.chained_assignment = None  # default='warn'
    random.seed(rand_seed)
    # prepare KG features
    if term_num:
        X = X.sample(n = term_num, axis = 1, random_state=rand_seed)
    # prepare clinical scores
    y = y[['Target Gene', 'Clinical Score']]
    y['Clinical Score'] = log2(y['Clinical Score'] + 1)
    y.index = y['Target Gene']
    y = y.drop(columns=['Target Gene'])
    # merge clinical scores and KG features
    y = y.join(X, how = 'right')
    # add negative targets as the number of positive
    y_pos = y[~y['Clinical Score'].isna()]
    try:
        y_pos['Clinical Score Binned'] = qcut(y_pos['Clinical Score'], q=bins, labels=['target_' + str(x) for x in list(range(bins))])
    except:
        y_pos['Clinical Score Binned'] = "target"
        print("Binning of cancer targets didn't work, using only one bin!")
    # prepare pathway genes
    y_pg = DataFrame()
    if pathway_genes:
        pathway_genes = [g for g in pathway_genes if g not in y_pos.index.tolist() and g not in known_targets]
        y_pg = y[y.index.isin(pathway_genes)]
        y_pg['Clinical Score'] = 1
        y_pg['Clinical Score Binned'] = 'pathway_gene'
    else:
        print("No pathway genes were provided!")
    y_neg = y[(y['Clinical Score'].isna()) & (~y.index.isin(pathway_genes)) & (~y.index.isin(known_targets))].sample(y_pos.shape[0] + y_pg.shape[0], random_state=rand_seed)
    y_neg['Clinical Score'] = -1
    y_neg['Clinical Score Binned'] = 'not_target'
    y = concat([y_pos, y_neg, y_pg])
    # shuffle targets
    y = y.sample(frac=1, random_state=rand_seed)

    # CREATE OBJECTS FOR MODELLING
    # create X for modelling
    X_out = y.iloc[:,1:-1]
    X_out.index = y.index
    # create y for modelling
    y_out = y['Clinical Score']
    y_out.index = y.index
    y_encoded = y['Clinical Score Binned']
    y_encoded.index = y.index
    # binarise y for binary classification
    y_binary = (y_out > 0).astype(int)

    return({
        'X': X_out,
        'y': y_out,
        'y_encoded': y_encoded,
        'y_binary': y_binary
    })

def run(X: DataFrame, y: DataFrame, y_slot = 'y_binary', bins: int = 3, pathway_genes: List = [], classifier: RandomForestClassifier = RandomForestClassifier(), cv: StratifiedKFold = StratifiedKFold(), scoring = 'roc_auc', n_iterations = 10, shuffle_scores = False) -> List:
    """
    Perform cross-validation pipeline for classification tasks.

    This function executes a cross-validation pipeline over multiple iterations,
    preprocessing data and evaluating a classifier using specified scoring metrics.

    :param X: Input features DataFrame
    :type X: DataFrame
    :param y: Target variable DataFrame
    :type y: DataFrame
    :param y_slot: Column name in the preprocessed result to use as target variable, defaults to ``y_binary``
    :type y_slot: str, optional
    :param bins: Number of bins for data preprocessing, defaults to 3
    :type bins: int, optional
    :param pathway_genes: List of pathway genes to be used in preprocessing, defaults to []
    :type pathway_genes: List, optional
    :param classifier: Classifier instance to use for cross-validation, defaults to RandomForestClassifier()
    :type classifier: RandomForestClassifier, optional
    :param cv: Cross-validation splitting strategy, defaults to StratifiedKFold()
    :type cv: StratifiedKFold, optional
    :param scoring: Scoring metric for evaluation, defaults to ``roc_auc``
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
        res = prepare(X, y, bins = bins, pathway_genes = pathway_genes, rand_seed = i)
        if shuffle_scores:
            score.append(mean(cross_val_score(classifier, res['X'], res[y_slot].sample(frac=1), scoring=scoring, cv = cv)))
        else:
            score.append(mean(cross_val_score(classifier, res['X'], res[y_slot], scoring=scoring, cv = cv)))

    return(score)