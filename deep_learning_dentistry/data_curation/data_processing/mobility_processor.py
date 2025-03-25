from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    merge_df_normal,
    save_curated_data_to_excel,
    save_curated_data_to_csv,
    load_curated_dataset,
    analyze_raw_df_data,
    transform_raw_df_data,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_mobility
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    MOBILITY_CURATED_PATH_EXCEL,
    MOBILITY_CURATED_PATH_CSV,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import (
    raw_to_cleaned_df_mobility,
)


def load_raw_mobility_data():
    """ Load raw mobility data and curate it. """
    df_list = load_mobility()
    mobility_combined_raw = merge_df_normal(df_list)
    mobility_combined_clean = raw_to_cleaned_df_mobility(mobility_combined_raw, 'mobility')
    return mobility_combined_clean

def save_curated_mobility_data_to_excel(final_mobility_df):
    """ Saves curated mobility data to an Excel file. """
    save_curated_data_to_excel(MOBILITY_CURATED_PATH_EXCEL, final_mobility_df)

def save_curated_mobility_data_to_csv(final_mobility_df):
    """ Saves curated mobility data as a CSV file. """
    save_curated_data_to_csv(MOBILITY_CURATED_PATH_CSV, final_mobility_df)

def load_curated_mobility_data_from_csv():
    """ Load curated mobility data. """
    return load_curated_dataset(MOBILITY_CURATED_PATH_CSV)

def analyze_raw_mobility_data(raw_bleeding_df):
    """ For each exam, find the number of teeth and number of sites that have mobility."""
    return analyze_raw_df_data(raw_bleeding_df, "mobility")

def transform_raw_mobility_data(raw_bleeding_df):
    """ Final transformation for raw bleeding data."""
    return transform_raw_df_data(raw_bleeding_df, "mobility", True, False)

def load_curate_save_mobility_data():
    """ Curates raw mobility data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed. """
    mobility_complete = load_raw_mobility_data()
    mobility_curated = analyze_raw_mobility_data(mobility_complete)
    mobility_transformed = transform_raw_mobility_data(mobility_curated)
    save_curated_mobility_data_to_excel(mobility_transformed)
    save_curated_mobility_data_to_csv(mobility_transformed)


if __name__ == "__main__":
    load_curate_save_mobility_data()