from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    merge_df_normal,
    save_curated_data_to_excel,
    save_curated_data_to_csv,
    load_curated_dataset,
    analyze_raw_df_data,
    transform_raw_df_data,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_mag
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    MAG_CURATED_PATH_CSV,
    MAG_CURATED_PATH_EXCEL,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import (
    raw_to_cleaned_df_mag,
)


def load_raw_mag_data():
    """ Load raw mobility data and curate it. """
    df_list = load_mag()
    mag_combined_raw = merge_df_normal(df_list)
    mobility_combined_clean = raw_to_cleaned_df_mag(mag_combined_raw)
    return mobility_combined_clean

def save_curated_mag_data_to_excel(final_mag_df):
    """ Saves curated mag data to an Excel file. """
    save_curated_data_to_excel(MAG_CURATED_PATH_EXCEL, final_mag_df)

def save_curated_mag_data_to_csv(final_mag_df):
    """ Saves curated mag data as a CSV file. """
    save_curated_data_to_csv(MAG_CURATED_PATH_CSV, final_mag_df)

def load_curated_mag_data_from_csv():
    """ Load curated mag data. """
    return load_curated_dataset(MAG_CURATED_PATH_CSV)

def analyze_raw_mag_data(raw_mag_df):
    """ For each exam, find the number of teeth and number of sites that have mag."""
    return analyze_raw_df_data(raw_mag_df, "mag")

def transform_raw_mag_data(mag_df):
    """ Final transformation for raw mag data."""
    return transform_raw_df_data(mag_df, "mag", True, True)

def load_curate_save_mag_data():
    """ Curates raw mag data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed. """
    mag_complete = load_raw_mag_data()
    mag_curated = analyze_raw_mag_data(mag_complete)
    mag_transformed = transform_raw_mag_data(mag_curated)
    save_curated_mag_data_to_excel(mag_transformed)
    save_curated_mag_data_to_csv(mag_transformed)


if __name__ == "__main__":
    load_curate_save_mag_data()
