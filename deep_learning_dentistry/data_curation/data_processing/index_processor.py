import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    merge_df_normal,
    load_curated_dataset,
    save_curated_data_to_excel,
    save_curated_data_to_csv,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_index
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    INDEX_CURATED_PATH_EXCEL,
    INDEX_CURATED_PATH_CSV,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import (
    transform_dataset_to_clean_index,
)


def curate_index_data():
    """ Load index data and curate it. """
    df_list = load_index()
    furcation_combined_raw = merge_df_normal(df_list)

    violations = check_violations(furcation_combined_raw)
    if violations:
        print("Error: Violations Found in the following CHART IDs:")
        print(violations)
        return "Error, Violations Found"

    index_updated = transform_dataset_to_clean_index(furcation_combined_raw)
    return index_updated

def save_curated_index_data_to_excel(final_index_df):
    """ Saves curated index data to an Excel file. """
    save_curated_data_to_excel(INDEX_CURATED_PATH_EXCEL, final_index_df)

def save_curated_index_data_to_csv(final_index_df):
    """ Saves curated index data as a CSV file. """
    save_curated_data_to_csv(INDEX_CURATED_PATH_CSV, final_index_df)

def load_curated_index_data_from_csv():
    """ Load curated index data. """
    return load_curated_dataset(INDEX_CURATED_PATH_CSV)

def transform_raw_index_data(index_df):
    """ Final transformation for raw index data."""
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
    index_df.rename(columns=rename_map, inplace=True)
    index_df['exam_date'] = pd.to_datetime(index_df['exam_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return index_df

def check_violations(dataframe, id_column="CHART ID", reference_column="CHART DATE"):
    """ Checks for violations where 1. one row has a number, and the other row has 0 (valid), 2. both rows have
    non-zero numbers, and they are different (violation). Returns list of CHART IDs that have violations. """

    duplicates_df = dataframe[dataframe.duplicated(subset=[id_column], keep=False)]
    columns_to_check = dataframe.columns[dataframe.columns.get_loc(reference_column) + 1 :]

    violations = []

    grouped = duplicates_df.groupby(id_column)

    for chart_id, group in grouped:
        for col in columns_to_check:
            unique_values = group[col].unique()

            if len(unique_values) > 2 or (len(unique_values) == 2 and 0 not in unique_values):
                violations.append(chart_id)
                break

    return violations

def load_curate_save_index_data():
    """ Curates raw index data and saves it. """
    index_complete = curate_index_data()
    index_transformed = transform_raw_index_data(index_complete)
    save_curated_index_data_to_excel(index_transformed)
    save_curated_index_data_to_csv(index_transformed)


if __name__ == "__main__":
    load_curate_save_index_data()