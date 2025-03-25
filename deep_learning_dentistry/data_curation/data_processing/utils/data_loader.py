import os
import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import fix_na_except_date

from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    BLEEDING_PATH,
    CHART_GENERAL_PATH,
    CHART_RESTORE_PATH,
    DEMOGRAPHIC_PATH,
    MOBILITY_FURCATION_INDEX_MAG_PATH,
    POCKETS_PATH,
    RECESSIONS_PATH,
    SUPPURATION_PATH
)

## Dataframe Processing ##

def process_dataframe(df):
    """
    Process a DataFrame to replace missing values or whitespace-only cells with pandas' <NA>.
    Returns the processed DataFrame.
    """
    # Title Changes
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Date changes
    if 'CHART DATE' in df.columns:
        df['CHART DATE'] = pd.to_datetime(df['CHART DATE'], errors='coerce')
        df = df.sort_values(by='CHART DATE').reset_index(drop=True)

    # Ensure CHART TITLE and CHART ID are consistently string type
    for col in ["CHART TITLE", "CHART ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Na Replacements except to CHART DATE
    df = fix_na_except_date(df, date_col="CHART DATE")

    return df


def load_file_normal(folder_path):
    """ From a variable's folder, for each file, extract the dataframe Return list of dfs. """
    df_list = []

    for filename in os.listdir(folder_path):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            xls = pd.ExcelFile(file_path, engine="openpyxl")
            if xls.sheet_names:
                # Load the first sheet
                df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
                df = process_dataframe(df)
                df_list.append(df)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    return df_list


def load_files_maxillary_and_mandibular(folder_path, sheet_names):
    """ From a variable's folder, for each file, take each maxillary or mandibular sheet and
    store as a df. Return list of dfs. Used in Bleeding and Suppuration variables. """

    df_dict = {"maxillary": [], "mandibular": []}

    for filename in os.listdir(folder_path):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            xls = pd.ExcelFile(file_path, engine="openpyxl")
            for sheet in xls.sheet_names:
                if any(sub.lower() in sheet.lower() for sub in sheet_names):
                    df = pd.read_excel(xls, sheet_name=sheet)
                    df = process_dataframe(df)
                    if "mandibular" in sheet.lower():
                        df_dict["mandibular"].append(df)
                    elif "maxillary" in sheet.lower():
                        df_dict["maxillary"].append(df)
                    else:
                        print(f"Sheet '{sheet}' in file '{filename}' did not match either category.")
        except Exception as e:
            print(f"Error processing file {filename} (matching sheet substring): {e}")

    return df_dict

def load_files_for_specific_variable(folder_path, variable):
    """ From a variable's folder, for each file that has multiple variables for each worksheet, extract the worksheet.
    Return list of dfs. Used in Mobility, Furcation, Index and MAG variables. """
    df_list = []

    for filename in os.listdir(folder_path):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                if variable.lower() in sheet_name.lower():
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df = process_dataframe(df)
                    df_list.append(df)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    return df_list


## Bleeding and Suppuration Data Loading ##

def load_bleeding_or_suppuration(file_path):
    """ Load all related files for variable BLEEDING or SUPPURATION from bleeding folder in the form of a list of dfs. Passes each
    df through filter_tooth_numbers_by_sheet()."""

    sheet_names = ["Maxillary", "Mandibular"]
    df_dict = load_files_maxillary_and_mandibular(file_path, sheet_names)

    for sheet_name, dfs in df_dict.items():
        filtered_dfs = []
        for df in dfs:
            filtered_df = filter_tooth_numbers_by_sheet(df, sheet_name)
            filtered_dfs.append(filtered_df)
        # Replace the original list with the filtered list
        df_dict[sheet_name] = filtered_dfs

    return df_dict

def load_chart_general():
    """ Load all related files for variable CHART GENERAL in the form of a list of dfs. """
    return load_file_normal(CHART_GENERAL_PATH)

def load_chart_restorative():
    """ Load all related files for variable CHART RESTORE in the form of a list of dfs. """
    return load_file_normal(CHART_RESTORE_PATH)

def load_demographic_data():
    """ Load all related files for Demographic Data in the form of a list of dfs. """
    return load_file_normal(DEMOGRAPHIC_PATH)

def load_furcation():
    """ Load all related files for variable FURCATION from mobility_furcation_index_mag folder in the form of a list of dfs. """
    return load_files_for_specific_variable(MOBILITY_FURCATION_INDEX_MAG_PATH, "Furcation")

def load_index():
    """ Load all related files for variable INDEX from mobility_furcation_index_mag folder in the form of a list of dfs. """
    return load_files_for_specific_variable(MOBILITY_FURCATION_INDEX_MAG_PATH, "Index")

def load_mobility():
    """ Load all related files for variable MOBILITY from mobility_furcation_index_mag folder in the form of a list of dfs. """
    return load_files_for_specific_variable(MOBILITY_FURCATION_INDEX_MAG_PATH, "Mobility")

def load_mag():
    """ Load all related files for variable MAG from mobility_furcation_index_mag folder in the form of a list of dfs. """
    return load_files_for_specific_variable(MOBILITY_FURCATION_INDEX_MAG_PATH, "MAG")

def load_pockets():
    """ Load all related files for variable POCKETS from pockets folder in the form of a list of dfs. Passes each
    df through filter_tooth_numbers_by_sheet()."""
    sheet_names = ["Pocket_Maxillary", "Pocket_Mandibular"]
    df_list = load_files_maxillary_and_mandibular(POCKETS_PATH, sheet_names)
    return df_list

def load_recessions():
    """ Load all related files for variable RECESSION from recessions folder in the form of a list of dfs. Passes each
    df through filter_tooth_numbers_by_sheet()."""
    sheet_names = ["Recession_Maxillary", "Recession_Mandibular"]
    df_list = load_files_maxillary_and_mandibular(RECESSIONS_PATH, sheet_names)
    return df_list


## Dataframe Modifications ##

def filter_tooth_numbers_by_sheet(df, sheet_name):
    """ Filters the DataFrame to retain only rows where the "TOOTH NBR" is in the valid range
    for the given sheet name (maxillary is from 11 to 28, mandibular is from 31 to 48). """

    if isinstance(df, str):
        df = pd.read_excel(df, sheet_name=sheet_name)

    # Check if df is really a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a DataFrame, got {type(df)} instead.")

    sheet_name_lower = sheet_name.lower()
    if sheet_name_lower == "maxillary":
        valid_numbers = list(range(18, 10, -1)) + list(range(21, 29))
    elif sheet_name_lower == "mandibular":
        valid_numbers = list(range(38, 30, -1)) + list(range(41, 49))
    else:
        valid_numbers = []

    df_filtered = df[df["TOOTH NBR"].isin(valid_numbers)].copy()
    return df_filtered