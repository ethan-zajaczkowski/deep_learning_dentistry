import os
import pandas as pd
from collections import defaultdict

def load_curated_dataset(file_path, extension=None):
    """ Loads a dataset from the given file path. Return None if error occurs. """
    if extension is not None:
        # Construct a full path by joining the directory/path and extension
        file_path = os.path.join(file_path, extension)

    try:
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}.")
        return dataset
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed. Ensure it is a valid CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
    return None


def fix_na_except_date(df: pd.DataFrame, date_col: str = "CHART DATE"):
    """ Convert dtypes and replace 'Na', 'NaN', blank strings, etc. with <NA> in all columns EXCEPT for the date_col. """

    df = df.convert_dtypes()
    cols_to_fix = [col for col in df.columns if col != date_col]

    df[cols_to_fix] = (
        df[cols_to_fix]
        .replace(r"^\s*$", pd.NA, regex=True)  # empty string -> <NA>
        .replace("Na", pd.NA)
        .replace("NaN", pd.NA)
        .where(lambda x: x.notnull(), pd.NA)   # set null -> <NA>
        .mask(lambda x: x.isna(), pd.NA)       # ensure consistent <NA>
    )

    return df


def generate_teeth_order():
    """
    Generates teeth order and categorizes teeth into upper-right, upper-left, lower-right, and lower-left.
    Returns a dictionary containing teeth categories and the combined teeth order.
    """
    # Define the ranges for each quadrant in reverse or standard order as needed
    upper_right_teeth = range(18, 10, -1)
    upper_left_teeth = range(21, 29)
    lower_left_teeth = range(38, 30, -1)
    lower_right_teeth = range(41, 49)

    # Combine all ranges into a single list
    teeth_order = list(upper_right_teeth) + list(upper_left_teeth) + \
                  list(lower_left_teeth) + list(lower_right_teeth)

    # Return the results as a dictionary
    return {
        "upper_right_teeth": list(upper_right_teeth),
        "upper_left_teeth": list(upper_left_teeth),
        "lower_left_teeth": list(lower_left_teeth),
        "lower_right_teeth": list(lower_right_teeth),
        "teeth_order": teeth_order
    }


def generate_teeth_site_mapping():
    """ Generate a map of sites for a tooth. """
    upper_mapping = ["DB", "B", "MB", "DP", "P", "MP"]
    lower_mapping = ["DB", "B", "MB", "DL", "L", "ML"]

    return {
    "upper_mapping": upper_mapping,
    "lower_mapping": lower_mapping
    }


def format_tooth_site_column(tooth: int, site: str) -> str:
    """ Given a tooth number, prefix the column with it's quadrant. """
    if 10 <= tooth <= 19:
        quadrant = 1
    elif 20 <= tooth <= 29:
        quadrant = 2
    elif 30 <= tooth <= 39:
        quadrant = 3
    elif 40 <= tooth <= 49:
        quadrant = 4
    else:
        quadrant = 0
    return f"q{quadrant}_{tooth}_{site}"


def create_tooth_site_columns():
    """ Generates a list of all tooth-site column names with site labels differing
    for upper and lower teeth. """
    mapping = generate_teeth_site_mapping()
    upper_site_labels = mapping["upper_mapping"]
    lower_site_labels = mapping["lower_mapping"]

    teeth_info = generate_teeth_order()
    teeth_order = teeth_info["teeth_order"]

    column_names = []
    for tooth in teeth_order:
        if 10 <= tooth <= 29:
            site_labels = upper_site_labels
        elif 30 <= tooth <= 49:
            site_labels = lower_site_labels
        else:
            site_labels = []

        for site in site_labels:
            column_names.append(format_tooth_site_column(tooth, site))

    return column_names

def merge_maxillary_and_mandibular_p_r(maxillary_data: pd.DataFrame, mandibular_data: pd.DataFrame):
    """
    Merges maxillary and mandibular pocket and recession data on specific columns, excluding other columns
    that shouldn't be used as merge keys.
    - Returns a merged DataFrame containing both maxillary and mandibular data.
    """

    # Columns to merge on
    merge_columns = ["ResearchID", "CHART ID", "CHART DATE"]

    # Perform the merge
    merged_data = pd.merge(
        maxillary_data,
        mandibular_data,
        on=merge_columns,
        how="outer",
        suffixes=("_maxillary", "_mandibular")
    )

    # Convert columns containing "Tooth" to strings
    tooth_columns = [col for col in merged_data.columns if "Tooth" in col]
    merged_data[tooth_columns] = merged_data[tooth_columns].applymap(
        lambda x: "Na" if pd.isna(x) else str(x)
    )

    # Uniform Na Values
    merged_data = fix_na_except_date(merged_data)
    final_df = merged_data.drop_duplicates()

    def combine_chart_title(row):
        title_max = row["CHART TITLE_maxillary"]
        title_man = row["CHART TITLE_mandibular"]
        if pd.isna(title_max) or title_max == "":
            return title_man
        return title_max

    final_df["CHART TITLE"] = final_df.apply(combine_chart_title, axis=1)
    final_df.drop(columns=["CHART TITLE_maxillary", "CHART TITLE_mandibular"], inplace=True)

    desired_metadata = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    other_columns = [col for col in final_df.columns if col not in desired_metadata]
    final_df = final_df[desired_metadata + other_columns]

    return final_df

def merge_df_for_maxillary_and_mandibular_bleeding_suppuration(df_dict):
    """ Merges dfs from maxillary and mandibular bleeding and suppuration variables. Returns combined raw df of all
    research_ids with bleeding or suppuration data combined in "Teeth Data" column (each row is one exam). """

    # Concatenate all DataFrames in each category
    maxillary_data = pd.concat(df_dict["maxillary"], axis=0, ignore_index=True)
    mandibular_data = pd.concat(df_dict["mandibular"], axis=0, ignore_index=True)
    df_complete = pd.concat([maxillary_data, mandibular_data], axis=0, ignore_index=True)

    merge_keys = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]

    df_grouped = df_complete.groupby(merge_keys).apply(
        lambda x: x.apply(lambda row: (row["TOOTH NBR"], row["AREA"], row["TOOTH SURFACE"]), axis=1).tolist()
    )
    df_grouped_flat = df_grouped.reset_index(name="Teeth Data")

    df_grouped_flat["CHART DATE"] = pd.to_datetime(df_grouped_flat["CHART DATE"], errors="coerce")
    df_grouped_flat = df_grouped_flat.sort_values(by=["ResearchID", "CHART DATE"])

    return df_grouped_flat


def merge_maxillary_and_mandibular_p_r(df_dict):
    """
    Merges DataFrames from a list (first half are maxillary, second half are mandibular)
    for pocket and recession variables. Returns a merged DataFrame containing both
    maxillary and mandibular data.
    """
    # Concatenate all DataFrames in each category
    maxillary_data = pd.concat(df_dict["maxillary"], axis=0, ignore_index=True)
    mandibular_data = pd.concat(df_dict["mandibular"], axis=0, ignore_index=True)

    # Columns to merge on
    merge_columns = ["ResearchID", "CHART ID", "CHART DATE"]

    # Perform the merge
    merged_data = pd.merge(
        maxillary_data,
        mandibular_data,
        on=merge_columns,
        how="outer",
        suffixes=("_maxillary", "_mandibular")
    )

    # Convert columns containing "Tooth" to strings
    tooth_columns = [col for col in merged_data.columns if "Tooth" in col]
    merged_data[tooth_columns] = merged_data[tooth_columns].applymap(
        lambda x: "Na" if pd.isna(x) else str(x)
    )

    # Uniform Na Values
    merged_data = fix_na_except_date(merged_data)
    final_df = merged_data.drop_duplicates()

    def combine_chart_title(row):
        title_max = row["CHART TITLE_maxillary"]
        title_man = row["CHART TITLE_mandibular"]
        if pd.isna(title_max) or title_max == "":
            return title_man
        return title_max

    final_df["CHART TITLE"] = final_df.apply(combine_chart_title, axis=1)
    final_df.drop(columns=["CHART TITLE_maxillary", "CHART TITLE_mandibular"], inplace=True)

    desired_metadata = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    other_columns = [col for col in final_df.columns if col not in desired_metadata]
    final_df = final_df[desired_metadata + other_columns]

    return final_df


def merge_df_normal(df_list):
    """ Merges dfs from the list. Returns combined raw df of all research_ids."""
    merge_keys = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    df_complete = pd.concat(df_list, axis=0, ignore_index=True)
    df_grouped = df_complete.groupby(merge_keys, as_index=False).first()
    df_grouped["CHART DATE"] = pd.to_datetime(df_grouped["CHART DATE"], errors="coerce")
    df_grouped = df_grouped.sort_values(by=["ResearchID", "CHART DATE"])
    return df_grouped


def merge_demographics_data(df_list):
    """
    Merges two demographics DataFrames based on 'ResearchID' as follows:
      - For each ResearchID present in base_df, if the "Periodontal disease risk"
        value is missing in base_df and exists in other_df, then use the value from other_df.
      - If a ResearchID appears only in other_df, include that row in the merged output.
      - Otherwise, keep the value from base_df.
      - Additionally, if the merged "Periodontal disease risk" value is different from
        the "Periodontal disease risk_other" value (and the other value is not pd.NA),
        then replace it with the value from other_df.

    After merging, the function:
      - Drops the temporary columns.
      - Calculates and prints summary statistics including:
          - Total rows in merged data.
          - Non-missing and missing cells in "Periodontal disease risk".
          - The count of ResearchIDs exclusively in other_df.
          - The number of cells that were filled (if missing in base but available in other_df).
          - The number of cells that were changed (if base had a value and it differed from other_df).
          - The sum of filled and changed cells.

    Returns the merged DataFrame.
    """
    if len(df_list) != 2:
        raise ValueError("This function expects exactly two DataFrames in df_list.")

    base_df, other_df = df_list

    # Set ResearchID as index for both DataFrames.
    base_df_indexed = base_df.set_index("ResearchID")
    other_df_indexed = other_df.set_index("ResearchID")

    # Perform an outer join so that all ResearchIDs from both DataFrames are included.
    merged = base_df_indexed.join(
        other_df_indexed[["Periodontal disease risk"]],
        how="outer",
        lsuffix="_base",
        rsuffix="_other"
    )

    # First, fill missing base values with the value from other_df.
    merged["Periodontal disease risk"] = merged["Periodontal disease risk_base"].combine_first(
        merged["Periodontal disease risk_other"]
    )

    # Now, update the "Periodontal disease risk" column:
    # If "Periodontal disease risk_other" is not missing and differs from the current value,
    # then replace the current value with the one from other_df.
    def update_risk(row):
        if pd.notna(row["Periodontal disease risk_other"]) and row["Periodontal disease risk"] != row[
            "Periodontal disease risk_other"]:
            return row["Periodontal disease risk_other"]
        return row["Periodontal disease risk"]

    merged["Periodontal disease risk"] = merged.apply(update_risk, axis=1)

    # Compute counts for changes:
    filled_count = ((merged["Periodontal disease risk_base"].isna()) &
                    (merged["Periodontal disease risk_other"].notna())).sum()
    changed_count = ((merged["Periodontal disease risk_base"].notna()) &
                     (merged["Periodontal disease risk_other"].notna()) &
                     (merged["Periodontal disease risk_base"] != merged["Periodontal disease risk_other"])).sum()
    total_updates = filled_count + changed_count

    # Drop the temporary columns.
    merged = merged.drop(columns=["Periodontal disease risk_base", "Periodontal disease risk_other"]).reset_index()

    # Calculate summary statistics.
    total_rows = len(merged)
    non_missing = merged["Periodontal disease risk"].notna().sum()
    missing = total_rows - non_missing

    # Count ResearchIDs that are exclusively in other_df.
    other_only_ids = set(other_df_indexed.index) - set(base_df_indexed.index)
    count_other_only = len(other_only_ids)

    # Print summary statistics.
    print(f"Total rows in merged data: {total_rows}")
    print(f"Cells with data (non-missing) in 'Periodontal disease risk': {non_missing}")
    print(f"Cells that are empty in 'Periodontal disease risk': {missing}")
    print(f"ResearchIDs exclusively in other_df: {count_other_only}")
    print(f"Cells filled (missing in base but available in other_df): {filled_count}")
    print(f"Cells changed (base value differed from other_df): {changed_count}")
    print(f"Total updates (filled + changed): {total_updates}")

    return merged


def save_curated_data_to_excel(destination, dataframe):
    """ Saves curated dataframe to an Excel file. """
    try:
        dataframe.to_excel(destination, index=False)
        print(f"Data successfully saved to {destination}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def save_curated_data_to_csv(destination, dataframe, new_filename=None):
    """ Saves curated dataframe to a CSV file. """
    try:
        if new_filename:
            if os.path.isdir(destination):
                destination = os.path.join(destination, new_filename)
                dir_path = os.path.dirname(destination)
                destination = os.path.join(dir_path, new_filename)
        dataframe.to_csv(destination, index=False)
        print(f"Data successfully saved to {destination}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def tooth_quadrant_determiner(tooth_id):
    """
    Determines which quadrant a given tooth belongs to.
    """
    tooth_quadrants = generate_teeth_order()

    if tooth_id in tooth_quadrants["upper_right_teeth"]:
        return "Q1"
    elif tooth_id in tooth_quadrants["upper_left_teeth"]:
        return "Q2"
    elif tooth_id in tooth_quadrants["lower_left_teeth"]:
        return "Q3"
    elif tooth_id in tooth_quadrants["lower_right_teeth"]:
        return "Q4"
    else:
        return "Invalid tooth ID"


def get_largest_number(cell_value):
    """
    Given a string such as "1-B, 1-DP, 2-MP",
    return the largest integer found before the dash in each comma-separated item.
    If no integers are found, return 1 by default.
    """
    # If it's not a string, or it's empty after stripping, return 0.
    if not isinstance(cell_value, str):
        return 0
    cell_value = cell_value.strip()
    if cell_value == "":
        return 0

    max_num = None
    for part in cell_value.split(","):
        part = part.strip()
        if not part:
            continue
        # Get the part before the dash.
        left_side = part.split("-")[0].strip()
        try:
            num = int(left_side)
            if max_num is None or num > max_num:
                max_num = num
        except ValueError:
            # If left_side isn't a valid integer, ignore it.
            continue

    # If no valid integer was found, default to 1 (since the cell had some content)
    return max_num if max_num is not None else 1


def is_non_zero(cell):
    """ Returns True if `cell` is a numeric value != 0, or a numeric string != "0". Returns False if `cell` is
    NaN, "Missing", "Not Available", or zero. """
    if pd.isna(cell):
        return False

    if isinstance(cell, str):
        if cell in ("Missing", "Not Available"):
            return False
        try:
            val = float(cell)
            return val != 0
        except ValueError:
            return False
    else:
        return cell != 0


def count_teeth(row, tooth_columns_map):
    """ Given a single row returns the number of teeth that have at least one non-zero site with some assumptions. """
    count = 0
    for tooth_base, col_list in tooth_columns_map.items():
        # If any column for this tooth is "non-zero" according to is_non_zero, increment count
        if any(is_non_zero(row[c]) for c in col_list):
            count += 1
    return count


def count_sites(row, tooth_columns_map):
    """
    Sums up all bleeding sites (i.e. counts all columns that are 1) for the given row.
    """
    total = 0
    for tooth, col_list in tooth_columns_map.items():
        for c in col_list:
            if pd.notna(row[c]) and row[c] == 1:
                total += 1
    return total


def analyze_raw_df_data(df, variable):
    """ For each exam, find the number of teeth and number of sites that have this variable."""

    num_of_variable_on_teeth = f"num_of_{variable}_teeth"
    num_of_variable_on_site = f"num_of_{variable}_sites"

    ## I. Number of teeth that have at least one site with variable ##
    tooth_columns_map = defaultdict(list)
    for col in df.columns:
        if col.startswith("q"):
            parts = col.split("_")
            if len(parts) == 3:
                tooth_num = parts[1]
                tooth_columns_map[tooth_num].append(col)
    df[num_of_variable_on_teeth] = df.apply(lambda row: count_teeth(row, tooth_columns_map), axis=1)

    ## II. Number of sites with variable ##
    df[num_of_variable_on_site] = df.apply(lambda row: count_sites(row, tooth_columns_map), axis=1)

    return df


def transform_raw_df_data(df, variable, analyzed, need_site):
    """ Final transformation for raw data."""

    num_of_variable_on_teeth = f"num_of_{variable}_teeth"
    num_of_variable_on_site = f"num_of_{variable}_sites"

    # I. Rename columns ##
    rename_dict = {
        "ResearchID": "research_id",
        "CHART ID": "exam_id",
        "CHART DATE": "exam_date",
        "CHART TITLE": "exam_type"
    }
    df.rename(columns=rename_dict, inplace=True)

    # II. Reorder columns so that the analyzed columns immediately follow "chart_title" ##
    if analyzed:
        cols = list(df.columns)
        if "exam_type" in cols:
            chart_type_index = cols.index("exam_type")
            if variable == "missing":
                x = 1
            else:
                x = 0
            if num_of_variable_on_teeth in cols:
                cols.remove(num_of_variable_on_teeth)
                cols.insert(chart_type_index + 1 + x, num_of_variable_on_teeth)
            if num_of_variable_on_site in cols:
                cols.remove(num_of_variable_on_site)
                cols.insert(chart_type_index + 2 + x, num_of_variable_on_site)
            df = df.reindex(columns=cols)

    # III. Convert chart_date to datetime (if not already) and sort by research_id and chart_date
    df["exam_date"] = pd.to_datetime(df["exam_date"], errors="coerce")
    df = df.sort_values(by=["research_id", "exam_date"])

    # IV. Check if you need the site level column
    if not need_site:
        df = df.drop(columns=[num_of_variable_on_site], errors='ignore')

    return df