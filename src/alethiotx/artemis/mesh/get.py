import requests
from pandas import read_pickle

def tree(s3_base = None, url_base = None, file_base = None) -> dict:
    """
    Retrieve MeSH tree structure from S3 cache or download from URL.
    This function attempts to load the MeSH (Medical Subject Headings) tree structure,
    first trying an S3 cache if provided, then falling back to downloading from a URL.
    :param s3_base: Base path for S3 storage location (optional). If provided, ``file_base`` is required.
    :type s3_base: str or None
    :param url_base: Base URL for downloading MeSH tree file (optional). Requires ``file_base``.
    :type url_base: str or None
    :param file_base: Base filename (without extension) for the MeSH tree file (optional).
                      Required when either ``s3_base`` or ``url_base`` is provided.
    :type file_base: str or None
    :return: Dictionary mapping MeSH headings to their tree numbers.
             Returns empty dict if all retrieval methods fail.
    :rtype: dict
    :raises: Prints error messages to console but does not raise exceptions.
    .. note::
       - Priority order: S3 cache → URL download
       - S3 files should be pickle format (.pkl extension)
       - URL files should be MeSH ASCII format (.bin extension)
       - MeSH ASCII format uses 'MH =' for headings and 'MN =' for tree numbers

    **Example**

    >>> mesh = tree(s3_base='s3://alethiotx-artemis/data/mesh/', file_base='d2025')
    >>> mesh = tree(url_base='https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/', file_base='d2025')
    """
    # Prioritize S3 if s3_base is provided
    if s3_base is not None:
        if file_base is None:
            print("Error: file_base is required when s3_base is provided")
            return {}
        try:
            print(f"Checking for cached MeSH tree at {s3_base}...")
            mesh_tree = read_pickle(s3_base + file_base + ".pkl")
            print(f"MeSH tree loaded from S3: {len(mesh_tree)} headings")
            return mesh_tree
        except Exception as e:
            print(f"S3 cache not found or could not be loaded: {e}")
            # Fall through to try URL download if available
    
    # If S3 failed or not provided, try URL download
    if url_base is not None and file_base is not None:
        # Continue with downloading from URL
        mesh_url = url_base + file_base + ".bin"
        print(f"Downloading MeSH tree from {mesh_url}...")
    
        try:
            response = requests.get(mesh_url, timeout=60)
            response.raise_for_status()
            
            # Parse the MeSH ASCII file
            mesh_tree = {}
            current_heading = None
            current_trees = []
            
            for line in response.text.split('\n'):
                line = line.strip()
                
                # MH = MeSH Heading
                if line.startswith('MH = '):
                    current_heading = line[5:].strip()
                    current_trees = []
                
                # MN = Tree Number
                elif line.startswith('MN = '):
                    tree_num = line[5:].strip()
                    current_trees.append(tree_num)
                
                # Entry = end of record, store the data
                elif line.startswith('ENTRY') or line.startswith('*NEWRECORD'):
                    if current_heading and current_trees:
                        mesh_tree[current_heading] = current_trees
                        current_heading = None
                        current_trees = []
            
            print(f"MeSH tree loaded: {len(mesh_tree)} headings")
            
            return mesh_tree
            
        except Exception as e:
            print(f"Warning: Could not download MeSH tree: {e}")
            return {}
    
    # Neither S3 nor URL options are available or both failed
    if s3_base is not None:
        print("Error: S3 download failed and no valid URL options provided")
    else:
        print("Error: Either (url_base and file_base) or (s3_base and file_base) are required")
    return {}

def descendants(heading: str, s3_base=None, file_base=None, url_base = None) -> list:
    """
    Get all descendant MeSH headings for a given heading based on tree hierarchy.
    This function retrieves a heading and all of its descendants in the MeSH tree
    structure by comparing tree numbers. A heading is considered a descendant if
    its tree number starts with a parent's tree number followed by a dot.
    :param heading: The MeSH heading to find descendants for
    :type heading: str
    :param s3_base: S3 bucket base path for MeSH data, defaults to None
    :type s3_base: str, optional
    :param file_base: Local file system base path for MeSH data, defaults to None
    :type file_base: str, optional
    :param url_base: URL base path for MeSH data, defaults to None
    :type url_base: str, optional
    :return: List of MeSH headings including the parent and all descendants
    :rtype: list
    :raises: None - Returns list with single heading if no tree numbers found
    .. note::
        The function uses tree numbers to determine hierarchy. A tree number like
        'A01.1.2' is a descendant of 'A01.1'.

    **Example**

    >>> descendants('Nervous System')
    ['Nervous System', 'Central Nervous System', 'Brain', ...]
    """
    # Get the full tree info (headings with their tree numbers)
    full_tree = tree(s3_base = s3_base, url_base = url_base, file_base=file_base)
    
    # Get tree numbers for the requested heading
    parent_trees = full_tree.get(heading)
    if not parent_trees:
        print(f"No tree numbers found for '{heading}', using exact match only")
        return [heading]
    
    print(f"Tree numbers for '{heading}': {parent_trees}")
    
    # Find all descendants recursively based on tree number hierarchy
    descendants = set([heading])
    
    # Check all headings to see if any of their tree numbers are descendants
    for heading, tree_nums in full_tree.items():
        for tree_num in tree_nums:
            # Check if this tree number is under any of the parent tree numbers
            for parent_tree in parent_trees:
                if tree_num.startswith(parent_tree + '.'):
                    descendants.add(heading)
                    break
    
    result = list(descendants)
    print(f"Found {len(result)} MeSH headings for '{heading}' (including all descendants)")
    return result

# run when file is directly executed
if __name__ == '__main__': 

    import pickle

    # this will read S3 if available, else download from URL
    full_tree = tree(
        s3_base = 's3://alethiotx-artemis/data/mesh/', 
        url_base = 'https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/',
        file_base='d2025'
    )
    
    # Save to local disk using pickle
    local_pickle_path = 'd2025.pkl'
    with open(local_pickle_path, 'wb') as f:
        pickle.dump(full_tree, f)
    print(f"Saved MeSH tree to disk: {local_pickle_path}")
    
    d = descendants(
        heading='Breast Neoplasms',
        s3_base = 's3://alethiotx-artemis/data/mesh/', 
        url_base = 'https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/',
        file_base='d2025'
    )
    print(d)