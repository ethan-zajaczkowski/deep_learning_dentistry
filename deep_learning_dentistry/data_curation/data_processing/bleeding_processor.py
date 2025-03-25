from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_bleeding_or_suppuration
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    save_curated_data_to_excel,
    save_curated_data_to_csv,
    load_curated_dataset,
    merge_df_for_maxillary_and_mandibular_bleeding_suppuration,
    analyze_raw_df_data,
    transform_raw_df_data,
)
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    BLEEDING_CURATED_PATH_EXCEL,
    BLEEDING_CURATED_PATH_CSV, BLEEDING_PATH
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import raw_to_cleaned_df_bleeding_and_sup

def load_raw_bleeding_data():
    """ Load raw bleeding data and curate it. """
    df_dict = load_bleeding_or_suppuration(BLEEDING_PATH)
    bleeding_combined_raw = merge_df_for_maxillary_and_mandibular_bleeding_suppuration(df_dict)
    bleeding_combined_clean = raw_to_cleaned_df_bleeding_and_sup(bleeding_combined_raw)
    return bleeding_combined_clean

def save_curated_bleeding_data_to_excel(final_bleeding_df):
    """ Saves curated bleeding data to an Excel file. """
    save_curated_data_to_excel(BLEEDING_CURATED_PATH_EXCEL, final_bleeding_df)

def save_curated_bleeding_data_to_csv(final_bleeding_df):
    """ Saves curated bleeding data as a CSV file. """
    save_curated_data_to_csv(BLEEDING_CURATED_PATH_CSV, final_bleeding_df)

def load_curated_bleeding_data_from_csv():
    """ Load curated bleeding data. """
    return load_curated_dataset(BLEEDING_CURATED_PATH_CSV)

def analyze_raw_bleeding_data(raw_bleeding_df):
    """ For each exam, find the number of teeth and number of sites that have bleeding."""
    return analyze_raw_df_data(raw_bleeding_df, "bleeding")

def transform_raw_bleeding_data(bleeding_df):
    """ Final transformation for raw bleeding data."""
    return transform_raw_df_data(bleeding_df, "bleeding", True, True)

def load_curate_save_bleeding_data():
    """ Curates raw bleeding data and saves it. """
    bleeding_complete = load_raw_bleeding_data()
    bleeding_curated = analyze_raw_bleeding_data(bleeding_complete)
    bleeding_transformed = transform_raw_bleeding_data(bleeding_curated)
    save_curated_bleeding_data_to_excel(bleeding_transformed)
    save_curated_bleeding_data_to_csv(bleeding_transformed)

if __name__ == "__main__":
    load_curate_save_bleeding_data()