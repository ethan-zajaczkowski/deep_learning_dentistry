import pandas as pd


def load_cleaned_dataset(file_path):
    """
    Loads a dataset from the given file path.

    Returns the cleaned dataset as a pandas DataFrame, or None if an error occurs.
    """
    try:
        dataset = pd.read_excel(file_path)
        print(f"Dataset loaded successfully from {file_path}.")
        return dataset
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed. Ensure it is a valid CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return None


def load_processed_dataset(file_path):
    """
    Loads a dataset from the given file path.

    Returns the processed dataset as a pandas DataFrame, or None if an error occurs.
    """
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
    """
    Convert dtypes and replace 'Na', 'NaN', blank strings, etc. with <NA>
    in all columns EXCEPT for the specified date_col.
    Returns the modified DataFrame.
    """
    # Convert to pandas extended dtypes (e.g. Int64, string, etc.)
    df = df.convert_dtypes()

    # Identify columns to fix (exclude date_col)
    cols_to_fix = [col for col in df.columns if col != date_col]

    # Replace blank/Na/NaN with <NA> in those columns
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
    upper_mapping = ["DB", "B", "MB", "DP", "P", "MP"]
    lower_mapping = ["DB", "B", "MB", "DL", "L", "ML"]

    return {
    "upper_mapping": upper_mapping,
    "lower_mapping": lower_mapping
    }


def create_tooth_site_columns():
    """
    Generates a list of all tooth-site column names with site labels differing
    for upper and lower teeth.
    """
    mapping = generate_teeth_site_mapping()
    upper_site_labels = mapping["upper_mapping"]
    lower_site_labels = mapping["lower_mapping"]

    # Get the teeth order and categorization
    teeth_info = generate_teeth_order()
    teeth_order = teeth_info["teeth_order"]

    # Create the list of column names
    column_names = []
    for tooth in teeth_order:
        # Use upper site labels for upper teeth (1x series) and lower site labels for lower teeth (3x, 4x series)
        if 10 <= tooth < 30:  # Upper teeth (1x and 2x series)
            column_names.extend([f"{tooth}-{site}" for site in upper_site_labels])
        else:  # Lower teeth (3x and 4x series)
            column_names.extend([f"{tooth}-{site}" for site in lower_site_labels])

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


def merge_maxillary_and_mandibular_b_s(maxillary_data: pd.DataFrame, mandibular_data: pd.DataFrame):
    """
    Merges maxillary and mandibular bleeding and suppuration data on specific columns, excluding other columns
    that shouldn't be used as merge keys.
    - Returns a merged DataFrame containing both maxillary and mandibular data for 
    """
    df_complete = pd.concat([maxillary_data, mandibular_data], axis=0, ignore_index=True)

    df_grouped = df_complete.groupby(["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]).apply(
        lambda x: x.apply(lambda row: (row["TOOTH NBR"], row["AREA"], row["TOOTH SURFACE"]), axis=1).tolist()
    )

    df_grouped_flat = df_grouped.reset_index(name="Teeth Data")
    df_grouped_flat["Teeth Data"] = df_grouped_flat["Teeth Data"].apply(sort_grouped_data)

    return df_grouped_flat
    

def sort_grouped_data(grouped_data):
    """
    Sort a pd series by the teeth order created above.
    """
    tooth_order_dict = {tooth: index for index, tooth in enumerate(generate_teeth_order()["teeth_order"])}
    return sorted(grouped_data, key=lambda x: tooth_order_dict.get(x[0], float('inf')))


def save_cleaned_data_to_excel(destination, dataframe):
    """
    Saves cleaned pockets data DataFrame to an Excel file.
    """
    try:
        dataframe.to_excel(destination, index=False)
        print(f"Data successfully saved to {destination}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


def save_processed_data_to_csv(destination, dataframe):
    """
    Saves processed pockets data DataFrame to a CSV file.
    """
    try:
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

def fill_exam_suppuration(exam):
    # Check if all values in 'suppuration' are missing
    if exam['suppuration'].isna().all():
        # Take the first suppuration_index (assuming consistency across the exam)
        supp_index = exam['suppuration_index'].iloc[0]
        if supp_index == 0:
            exam.loc[:, 'suppuration'] = 0
        elif supp_index > 0 or supp_index < 0:
            exam.loc[:, 'suppuration'] = 'Missing'
    return exam