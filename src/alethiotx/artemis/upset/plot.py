from upsetplot import UpSet, from_contents

def prepare(breast, lung, prostate, melanoma, bowel, diabetes, cardiovascular, mode):
    """
    Prepare data for UpSet plot visualization.

    This function processes disease-related data and converts it into a format
    suitable for UpSet plot generation using the `from_contents` function.

    :param breast: Breast cancer data. In ``ct`` mode, should be a DataFrame with
                   ``Target Gene`` column. In ``pg`` mode, should be a list.
    :type breast: pandas.DataFrame or list
    :param lung: Lung cancer data. In ``ct`` mode, should be a DataFrame with
                 ``Target Gene`` column. In ``pg`` mode, should be a list.
    :type lung: pandas.DataFrame or list
    :param prostate: Prostate cancer data. In ``ct`` mode, should be a DataFrame with
                     ``Target Gene`` column. In ``pg`` mode, should be a list.
    :type prostate: pandas.DataFrame or list
    :param melanoma: Melanoma data. In ``ct`` mode, should be a DataFrame with
                     ``Target Gene`` column. In ``pg`` mode, should be a list.
    :type melanoma: pandas.DataFrame or list
    :param bowel: Bowel cancer data. In ``ct`` mode, should be a DataFrame with
                  ``Target Gene`` column. In ``pg`` mode, should be a list.
    :type bowel: pandas.DataFrame or list
    :param diabetes: Diabetes data. In ``ct`` mode, should be a DataFrame with
                     ``Target Gene`` column. In ``pg`` mode, should be a list.
    :type diabetes: pandas.DataFrame or list
    :param cardiovascular: Cardiovascular disease data. In ``ct`` mode, should be a
                           DataFrame with ``Target Gene`` column. In ``pg`` mode,
                           should be a list.
    :type cardiovascular: pandas.DataFrame or list
    :param mode: Processing mode. ``ct`` extracts ``Target Gene`` column from DataFrames,
                 ``pg`` uses the data directly as lists.
    :type mode: str
    :return: Dictionary formatted for UpSet plot with disease names as keys and
             gene lists or data lists as values.
    :rtype: dict
    :raises ValueError: If mode is not ``ct`` or ``pg``.
    """
    if mode == 'ct':
        return from_contents(
            {
                "breast": breast['Target Gene'].tolist(), 
                "lung": lung['Target Gene'].tolist(), 
                "prostate": prostate['Target Gene'].tolist(), 
                "melanoma": melanoma['Target Gene'].tolist(), 
                "bowel": bowel['Target Gene'].tolist(), 
                "diabetes": diabetes['Target Gene'].tolist(), 
                "cardiovascular": cardiovascular['Target Gene'].tolist()
            }
        )
    elif mode == 'pg':
        return from_contents(
            {
                "breast": breast, 
                "lung": lung, 
                "prostate": prostate, 
                "melanoma": melanoma, 
                "bowel": bowel, 
                "diabetes": diabetes, 
                "cardiovascular": cardiovascular
            }
        )
    else:
        raise ValueError("Mode must be either 'ct' or 'pg'")

def create(indications, min_subset_size):
    """
    Create an UpSet plot for visualizing set intersections.

    :param indications: Data structure containing set membership information for creating the UpSet plot
    :type indications: pandas.DataFrame or similar data structure compatible with UpSet
    :param min_subset_size: Minimum size threshold for subsets to be displayed in the plot
    :type min_subset_size: int
    :return: An UpSet plot object configured with the specified parameters
    :rtype: UpSet
    :raises TypeError: If indications is not a compatible data structure
    :raises ValueError: If min_subset_size is negative

    .. note::

        The plot is configured with the following settings:
        
        - subset_size: "count" - shows count of elements in each subset
        - orientation: "vertical" - displays plot in vertical orientation
        - show_counts: "{:d}" - formats counts as integers
        - sort_by: ``cardinality`` - sorts subsets by their size
        - sort_categories_by: ``input`` - maintains input order for categories
        - min_subset_size: filters out subsets smaller than the specified threshold

    .. seealso::

        `UpSet documentation <https://upsetplot.readthedocs.io/>`_
    """
    return UpSet(
        indications, 
        subset_size="count", 
        orientation="vertical", 
        show_counts="{:d}", 
        sort_by='cardinality', 
        sort_categories_by = 'input',
        min_subset_size = min_subset_size,
    )
