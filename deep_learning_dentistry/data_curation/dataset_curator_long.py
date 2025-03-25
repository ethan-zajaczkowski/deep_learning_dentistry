import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from deep_learning_dentistry.data_curation.data_processing.utils.config import FULL_DATASET_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import create_tooth_site_columns, \
    save_curated_data_to_csv, load_curated_dataset
from deep_learning_dentistry.data_curation.dataset_curator_wide import load_curated_full_dataset_wide_from_csv, \
    curate_full_dataset_wide, save_curated_dataset_wide_to_csv


def transform_dataset(df_raw, variable = None):
    """ Takes the full wide dataset and melts to a site-level dataframe. """

    final_df = pd.DataFrame()
    cols = df_raw.columns.tolist()
    demo_cols = df_raw.loc[:, "research_id":"past_periodontal_treatment"].columns.tolist()
    num_of_teeth_sites = [col for col in cols if col.startswith("num_of_") and not col.endswith("_sites")]
    num_of_sites = [col for col in cols if col.startswith("num_of_") and not col.endswith("_teeth")]
    tooth_based_columns = [col for col in cols if re.match(r"^(has_|worked_on_)", col)]
    index_cols = [col for col in cols if
                  col.startswith("number_of_") or col.startswith("pcnt_of") or col.endswith("_index")]

    for _, row in df_raw.iterrows():
        # I. Base df
        site_df = create_tooth_site_columns_df()

        # II. df for demographics column
        demographics = row[demo_cols].to_dict()
        df_demo = pd.DataFrame([demographics] * len(site_df))
        for exam_col in ["exam_id", "exam_date", "exam_type"]:
            df_demo[exam_col] = row[exam_col]


        # Skip further processing, as there are no exams for this patient
        if pd.isna(row["exam_id"]):
            final_df = pd.concat([final_df, df_demo], ignore_index=True)
            continue

        # III. df for num_of_teeth type columns
        num_of_teeth_sites_data = row[num_of_teeth_sites].to_dict()
        df_num_of_teeth_sites = pd.DataFrame([num_of_teeth_sites_data] * len(site_df))

        # IV. df for num_of_sites type columns
        num_of_sites_data = row[num_of_sites].to_dict()
        df_num_of_sites = pd.DataFrame([num_of_sites_data] * len(site_df))

        # V. Combine the current dataframes
        combined_df = pd.concat([
            df_demo.reset_index(drop=True),
            df_num_of_teeth_sites.reset_index(drop=True),
            df_num_of_sites.reset_index(drop=True),
            site_df.reset_index(drop=True)
        ], axis=1)

        # VI. Work through missing teeth
        combined_df = assign_columns_by_pattern(combined_df, row, r"^q\d+_\d+_missing_status$", "missing_status")

        # VII. Work through restored status
        restored_tooth_col = row["restored_teeth"]
        combined_df = determine_restored_teeth(combined_df, restored_tooth_col)

        # VIII. Work through every variable
        variable_rows = row_generator_for_variables(row)
        variable_rows_combined = merge_variable_rows(variable_rows)
        combined_df = merge_variable_data(combined_df, variable_rows_combined)

        # IX. Work through "has_" or "worked_on_" columns
        for col_name in tooth_based_columns:
            tooth_id_list_str = row.get(col_name, "")
            combined_df = determine_teeth_in_list(
                combined_df,
                tooth_id_list_str,
                output_col=col_name
            )

        # X. Work through index values
        for col in index_cols:
            if pd.isna(row[col]) or (isinstance(row[col], str) and row[col].strip() == ""):
                combined_df[col] = pd.NA
            else:
                value = row[col]
                combined_df[col] = value

        # XI. Work thorugh notes
        combined_df = assign_columns_by_pattern(combined_df, row, r"^q\d+_\d+_other_issues$", "written_issues")
        combined_df["restoration_notes"] = [row.get("restoration_notes")] * len(combined_df)


        final_df = pd.concat([final_df, combined_df], ignore_index=True)

        print(row['research_id'])


    return final_df


def assign_columns_by_pattern(combined_df, row, col_pattern, output_col):
    """
    Generalized function to assign values from a row to combined_df based on a given column pattern.

    Parameters:
      combined_df (pd.DataFrame): The DataFrame with site-level data; must contain 'quadrant' and 'tooth_id'.
      row (pd.Series): The wide-format row containing the variable columns.
      col_pattern (str): A regex pattern to match columns (e.g., r"^q\d+_\d+_missing_status$").
      output_col (str): The name of the new column to create in combined_df.

    Returns:
      pd.DataFrame: The updated DataFrame with the new column added.

    The function builds a mapping keyed by (quadrant, tooth_id) from the matched columns in the row.
    Then, for every row in combined_df, it assigns the value from the mapping if found; otherwise, np.nan.
    """
    mapping = {}
    # Find all columns in the row that match the pattern
    matched_cols = [col for col in row.index if re.match(col_pattern, col)]
    for col in matched_cols:
        # Extract quadrant and tooth_id from the column name
        match = re.match(r"^q(\d+)_(\d+)_", col)
        if match:
            quadrant = int(match.group(1))
            tooth_id = int(match.group(2))
            mapping[(quadrant, tooth_id)] = row[col]

    # Ensure that "quadrant" and "tooth_id" columns in combined_df are integers
    combined_df["quadrant"] = combined_df["quadrant"].astype(int)
    combined_df["tooth_id"] = combined_df["tooth_id"].astype(int)

    def get_value(r):
        q = int(r["quadrant"])
        t = int(r["tooth_id"])
        return mapping.get((q, t), np.nan)

    combined_df[output_col] = combined_df.apply(get_value, axis=1)
    return combined_df


def determine_restored_teeth(combined_df, restored_tooth_col):
    """
    Determines which teeth are restored and adds a "restored_tooth" column to combined_df.

    If restored_tooth_col is empty, then every row in combined_df gets a 0 in "restored_tooth".
    Otherwise, it expects restored_tooth_col to be a string like "xx" or "xx,yy" (with integers
    as tooth ids) and sets "restored_tooth" to 1 for rows whose tooth_id is in the parsed list,
    and 0 for all other rows.

    Args:
        combined_df (pd.DataFrame): The site-level DataFrame containing a "tooth_id" column.
        restored_tooth_col (str): The cell value from the wide DataFrame indicating restored teeth.

    Returns:
        pd.DataFrame: combined_df with an added "restored_tooth" column.
    """
    # If the cell is empty or not a string, set restored_tooth = 0 for all rows.
    if not isinstance(restored_tooth_col, str) or restored_tooth_col.strip() == "":
        combined_df["restored_tooth"] = 0
    else:
        # Parse the string into a list of integers. For example, "18" becomes [18] and "18,19" becomes [18, 19].
        try:
            restored_list = [int(x.strip()) for x in restored_tooth_col.split(",") if x.strip() != ""]
        except ValueError as e:
            raise ValueError(f"Error parsing restored teeth value: {restored_tooth_col}") from e

        # Ensure tooth_id in combined_df is of type int for proper comparison.
        combined_df["tooth_id"] = combined_df["tooth_id"].astype(int)

        # For each row in combined_df, if the tooth_id is in restored_list, mark it as 1 (restored); otherwise, 0.
        combined_df["restored_tooth"] = combined_df["tooth_id"].isin(restored_list).astype(int)

    return combined_df

def parse_column_name(column_name):
    """
    Given a column name like 'q1_18_DB_PLAQUE', extracts:
      quadrant=1, tooth_id=18, site_id='DB', variable='PLAQUE'
    Adjust the regex to match your naming pattern.
    """
    # Example pattern: q{quadrant}_{tooth_id}_{site_id}_{variable}
    match = re.match(r"^q(\d+)_(\d+)_(\w+)_(\w+)$", column_name)

    quadrant_value = int(match.group(1))
    tooth_id = int(match.group(2))
    site_id = match.group(3)
    variable_name = match.group(4)

    return quadrant_value, tooth_id, site_id, variable_name


def row_generator_for_variables(row):
    """
    Transforms a single wide row into multiple records,
    each with (quadrant, tooth_id, site_id, <variable_name>).
    """
    processed_rows = []

    pattern_cols = [
        col for col in row.index
        if re.match(r"^q\d+_\d+_\w+_\w+$", col)
           and not (col.endswith("_issues") or col.endswith("_missing_status"))
    ]

    for column in pattern_cols:
        quadrant, tooth_id, site_id, variable_name = parse_column_name(column)

        if quadrant is not None:
            value = row[column]
            processed_rows.append({
                "quadrant": quadrant,
                "tooth_id": tooth_id,
                "site_id": site_id,  # naming it 'site_id' to match combined_df
                variable_name: value  # e.g. 'PLAQUE': <some_value>
            })

    return processed_rows


def merge_variable_rows(variable_rows):
    """
    Merges multiple dictionaries with the same (quadrant, tooth_id, site_id)
    into a single dictionary that has all variables in one row.

    Example:
      Input:
        [
          {'bleeding': 0.0, 'quadrant': '1', 'site_id': 'DB', 'tooth_id': '18'},
          {'suppuration': 0.0, 'quadrant': '1', 'site_id': 'DB', 'tooth_id': '18'}
        ]

      Output:
        [
          {'quadrant': '1', 'tooth_id': '18', 'site_id': 'DB', 'bleeding': 0.0, 'suppuration': 0.0}
        ]
    """

    merged_map = {}  # Key: (quadrant, tooth_id, site_id), Value: single merged dict

    for d in variable_rows:
        quad = d["quadrant"]
        tooth = d["tooth_id"]
        site = d["site_id"]
        key = (quad, tooth, site)

        if key not in merged_map:
            # Initialize the base dict for this (quadrant, tooth, site)
            merged_map[key] = {
                "quadrant": quad,
                "tooth_id": tooth,
                "site_id": site
            }

        # Merge in any additional variables (e.g., bleeding, suppuration)
        for k, v in d.items():
            if k not in ("quadrant", "tooth_id", "site_id"):
                merged_map[key][k] = v

    # Return one dictionary per unique (quadrant, tooth_id, site_id)
    return list(merged_map.values())


def determine_teeth_in_list(combined_df, tooth_id_list_str, output_col="restored_tooth"):
    """
    Assigns 1 or 0 to rows in combined_df based on whether their tooth_id
    appears in a comma-separated list of tooth IDs.

    If tooth_id_list_str is empty or not a valid string, then every row in combined_df gets 0.
    Otherwise, it expects a string like "18" or "18,19" (with integers as tooth IDs)
    and sets 'output_col' to 1 for rows whose tooth_id is in the parsed list, 0 otherwise.

    Args:
        combined_df (pd.DataFrame): The site-level DataFrame containing a 'tooth_id' column.
        tooth_id_list_str (str): The cell value from the wide DataFrame indicating tooth IDs (e.g., "18" or "18,19").
        output_col (str): The name of the new/updated column in combined_df (default is 'restored_tooth').

    Returns:
        pd.DataFrame: The updated combined_df with the new column added.
    """
    # If the cell is empty or not a string, set output_col = 0 for all rows.
    if not isinstance(tooth_id_list_str, str) or tooth_id_list_str.strip() == "":
        combined_df[output_col] = 0
    else:
        # Parse the string into a list of integers, e.g. "18" -> [18], "18,19" -> [18, 19]
        try:
            tooth_id_list = [
                int(x.strip()) for x in tooth_id_list_str.split(",")
                if x.strip() != ""
            ]
        except ValueError as e:
            raise ValueError(f"Error parsing tooth ID list: {tooth_id_list_str}") from e

        # Ensure tooth_id in combined_df is int for proper comparison
        combined_df["tooth_id"] = combined_df["tooth_id"].astype(int)

        # 1 if in the list, 0 otherwise
        combined_df[output_col] = combined_df["tooth_id"].isin(tooth_id_list).astype(int)

    return combined_df


def merge_variable_data(combined_df, variable_rows):
    """
    Merges site-level variable data into combined_df.

    Args:
      combined_df (pd.DataFrame): The main site-level DataFrame with columns 'quadrant', 'tooth_id', and 'site_id'.
      variable_rows (list): A list of dictionaries generated by row_generator_for_variables().

    Returns:
      pd.DataFrame: The merged DataFrame with variable columns added.
    """
    # Combine the variable rows into a single dictionary per unique (quadrant, tooth_id, site_id)
    variable_rows_combined = merge_variable_rows(variable_rows)
    df_var = pd.DataFrame(variable_rows_combined)

    # Ensure that the merge keys have the same data type (using string here)
    for col in ["quadrant", "tooth_id", "site_id"]:
        combined_df[col] = combined_df[col].astype(str)
        df_var[col] = df_var[col].astype(str)

    # Merge on the keys
    merged_df = combined_df.merge(df_var, on=["quadrant", "tooth_id", "site_id"], how="left")
    return merged_df




def create_tooth_site_columns_df():
    """ Generates a DataFrame with three columns quadrant, tooth_id, site_id. """

    cols = create_tooth_site_columns()

    data = []
    for col in cols:
        parts = col.split("_")
        quadrant = parts[0].lstrip("q")
        tooth_id = parts[1]
        site_id = parts[2]
        data.append({
            "quadrant": quadrant,
            "tooth_id": tooth_id,
            "site_id": site_id
        })

    return pd.DataFrame(data)


def parse_column_name(col_name):
    """ Splits a column name like 'q1_17_MB_bleeding' into four parts:
      1) quadrant  (e.g. 'q1')
      2) tooth     (e.g. '17')
      3) site      (e.g. 'MB')
      4) measurement (e.g. 'bleeding')
    """
    parts = col_name.split("_", 3)
    while len(parts) < 4:
        parts.append("")
    quadrant_str, tooth_num, site, measurement = parts
    quadrant_num = quadrant_str.lstrip("q")  # 'q1' -> '1'
    return quadrant_num, tooth_num, site, measurement


def curate_full_dataset_long():
    """ Curates full dataset_long. """
    # Load the full wide dataset from CSV
    df = load_curated_full_dataset_wide_from_csv()
    unique_ids = df['research_id'].unique()

    chunks = []
    for i in range(0, len(unique_ids), 400):
        chunk_ids = unique_ids[i:i + 400]
        chunk_df = df[df['research_id'].isin(chunk_ids)]
        chunks.append(chunk_df)

    processed_chunks = []
    for chunk in chunks:
        processed_chunk = transform_dataset(chunk)
        processed_chunks.append(processed_chunk)

    df_full = pd.concat(processed_chunks, ignore_index=True)
    return df_full

def save_curated_dataset_long_to_csv(full_dataset_wide, name):
    """ Saves curated full dataset wide as a CSV file. """
    save_curated_data_to_csv(FULL_DATASET_PATH, full_dataset_wide, name)

def load_curated_full_dataset_long_from_csv():
    return load_curated_dataset(FULL_DATASET_PATH, "full_dataset_long.csv")

def load_curate_save_full_dataset_long():
    """ Curates full dataset_long and saves it. """
    full_dataset = curate_full_dataset_long()
    save_curated_dataset_wide_to_csv(full_dataset, "full_dataset_long.csv")

if __name__ == "__main__":
    load_curate_save_full_dataset_long()