from deep_learning_dentistry.data_curation.data_processing.utils.config import FURCATION_CURATED_PATH_EXCEL, \
    MAG_CURATED_PATH_CSV, FURCATION_CURATED_PATH_CSV
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import merge_df_normal, \
    load_curated_dataset, save_curated_data_to_csv, save_curated_data_to_excel, analyze_raw_df_data, \
    transform_raw_df_data
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_furcation
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import raw_to_cleaned_df_furcation


def load_raw_furcation_data():
    """ Load raw mobility data and curate it. """
    df_list = load_furcation()
    furcation_combined_raw = merge_df_normal(df_list)
    furcation_combined_clean = raw_to_cleaned_df_furcation(furcation_combined_raw)
    return furcation_combined_clean

def save_curated_furcation_data_to_excel(final_furcation_df):
    """ Saves curated furcation data to an Excel file. """
    save_curated_data_to_excel(FURCATION_CURATED_PATH_EXCEL, final_furcation_df)

def save_curated_furcation_data_to_csv(final_furcation_df):
    """ Saves curated furcation data as a CSV file. """
    save_curated_data_to_csv(FURCATION_CURATED_PATH_CSV, final_furcation_df)

def load_curated_furcation_data_from_csv():
    """ Load curated furcation data. """
    return load_curated_dataset(FURCATION_CURATED_PATH_CSV)

def analyze_raw_furcation_data(raw_furcation_df):
    """ For each exam, find the number of teeth that have furcation."""
    return analyze_raw_df_data(raw_furcation_df, "furcation")

def transform_raw_furcation_data(furcation_df):
    """ Final transformation for raw furcation data."""
    return transform_raw_df_data(furcation_df, "furcation", True, False)

def load_curate_save_furcation_data():
    """ Curates raw furcation data and saves it. """
    furcation_complete = load_raw_furcation_data()
    furcation_curated = analyze_raw_furcation_data(furcation_complete)
    furcation_transformed = transform_raw_furcation_data(furcation_curated)
    save_curated_furcation_data_to_excel(furcation_transformed)
    save_curated_furcation_data_to_csv(furcation_transformed)


if __name__ == "__main__":
    load_curate_save_furcation_data()