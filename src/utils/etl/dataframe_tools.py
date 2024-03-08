import pandas as pd


def column_filter(df: pd.DataFrame, substrings_dict: dict, match_all: bool = True, names_only: bool = False) -> pd.DataFrame:
    """
    Filters columns in a DataFrame based on a dictionary of substrings.

    Parameters:
    - df: The input DataFrame.
    - substrings_dict: Dictionary where each key holds a list of strings to filter columns.
    - match_all: If True, filters columns that contain all substrings within each list.
                 If False, filters columns that contain any substring within each list.

    Returns:
    - A DataFrame with filtered columns.
    """
    matched_columns = []

    for key, substrings in substrings_dict.items():
        if match_all:
            # Directly use the function to check if all substrings are in the column name
            mask = df.columns.map(lambda col_name: all(sub in col_name for sub in substrings))
        else:
            # Directly use the function to check if any substring is in the column name
            mask = df.columns.map(lambda col_name: any(sub in col_name for sub in substrings))
        matched_columns.extend(df.columns[mask].tolist())

    result = df[matched_columns]

    if result.empty:
        print("No matching columns found. Please double check your entries!")
    elif names_only:
        return matched_columns
    else:
        return result
    
    
    
def create_random_dataset(original_df):
    """
    Create a new DataFrame with the same shape as the original, filled with random values.

    Parameters:
    original_df (pd.DataFrame): Original DataFrame to use as a template for shape.

    Returns:
    pd.DataFrame: New DataFrame with random data.
    """
    # Get the shape of the original DataFrame
    rows, cols = original_df.shape

    # Create a NumPy array with random values of the same shape
    random_array = np.random.rand(rows, cols)

    # Create a DataFrame from the random array with the same column names
    random_df = pd.DataFrame(random_array, columns=original_df.columns)

    return random_df