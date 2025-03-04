import pandas as pd

from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_index
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset, load_cleaned_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (INDEX_CLEANED_PATH, INDEX_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset_to_clean_index, transform_dataset


def curate_index_data():
    """
    Loads index data from the respective sources.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    index_complete = load_index()

    # Check for violations in the data
    violations = check_violations(index_complete)

    if violations:
        print("Error: Violations Found in the following CHART IDs:")
        print(violations)
        return "Error, Violations Found"

    index_updated = transform_dataset_to_clean_index(index_complete)
    return index_updated


def transform_index_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    index_complete = curate_index_data()
    index_transformed = transform_dataset(index_complete, variable="index")
    return index_transformed


def load_processed_index_data():
    """
    Loads the processed index dataset from the specified file path.
    """
    return load_processed_dataset(INDEX_PROCESSED_PATH)


def load_cleaned_index_data_modified():
    """
    Loads the processed index dataset from the specified file path,
    then renames the columns to standardized names.
    """
    df = load_cleaned_dataset(INDEX_CLEANED_PATH)

    # Original -> New column name mapping
    rename_map = {
        'ResearchID': 'research_id',
        'CHART TITLE': 'exam_type',
        'CHART ID': 'exam_id',
        'CHART DATE': 'exam_date',
        'BLEEDING_INDEX': 'bleeding_index',
        'SUPPURATION_INDEX': 'suppuration_index',
        'PLAQUE_INDEX': 'plaque_index',
        'NVL(GCOUNT_OF_MISSING_TEETH,0)': 'missing_teeth',
        'NVL(PCNT_OF_BLEEDING_SURFACES,0)': 'percent_of_bleeding_surfaces',
        'NVL(PCNT_OF_SUPPURATION_SURFACES,0)': 'percent_of_suppuration_surfaces',
        'NVL(PCNT_OF_PLAQUE_SURFACES,0)': 'percent_of_plaque_surfaces'
    }

    # Rename columns
    df.rename(columns=rename_map, inplace=True)

    df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    return df


def save_cleaned_index_data_to_excel(dataframe):
    """
    Saves cleaned index data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(INDEX_CLEANED_PATH, dataframe)


def save_processed_index_data_to_csv(dataframe):
    """
    Saves processed index data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(INDEX_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_index():
    """
    Curates raw index data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    index_complete = curate_index_data()
    save_cleaned_index_data_to_excel(index_complete)
    transformed_index = transform_dataset(index_complete, variable="index")
    save_processed_index_data_to_csv(transformed_index)


def check_violations(dataframe, id_column="CHART ID", reference_column="CHART DATE"):
    """
    Checks for violations where:
    - One row has a number, and the other row has 0 (valid).
    - Both rows have non-zero numbers, and they are different (violation).

    Returns list of CHART IDs that have violations.
    """
    duplicates_df = dataframe[dataframe.duplicated(subset=[id_column], keep=False)]
    columns_to_check = dataframe.columns[dataframe.columns.get_loc(reference_column) + 1 :]

    violations = []

    grouped = duplicates_df.groupby(id_column)

    for chart_id, group in grouped:
        for col in columns_to_check:
            unique_values = group[col].unique()

            # Check for violation conditions
            if len(unique_values) > 2 or (len(unique_values) == 2 and 0 not in unique_values):
                violations.append(chart_id)
                break  # Stop checking further columns for this CHART ID

    return violations


if __name__ == "__main__":
    curate_clean_transform_process_save_index()