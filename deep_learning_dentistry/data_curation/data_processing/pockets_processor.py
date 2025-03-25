from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    POCKETS_CURATED_PATH_EXCEL,
    POCKETS_CURATED_PATH_CSV,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    merge_maxillary_and_mandibular_p_r,
    save_curated_data_to_excel,
    save_curated_data_to_csv,
    load_curated_dataset,
    transform_raw_df_data,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import (
    load_pockets,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import (
    raw_to_cleaned_df_pockets_and_rec,
)

def load_raw_pockets_data():
    """ Load raw bleeding data and curate it. """
    df_list = load_pockets()
    pockets_combined_raw = merge_maxillary_and_mandibular_p_r(df_list)
    pockets_combined_cleaned = raw_to_cleaned_df_pockets_and_rec(pockets_combined_raw)
    return pockets_combined_cleaned

def save_curated_pockets_data_to_excel(final_pockets_df):
    """ Saves curated pockets data to an Excel file. """
    save_curated_data_to_excel(POCKETS_CURATED_PATH_EXCEL, final_pockets_df)

def save_curated_pockets_data_to_csv(final_pockets_df):
    """ Saves curated pockets data as a CSV file. """
    save_curated_data_to_csv(POCKETS_CURATED_PATH_CSV, final_pockets_df)

def load_curated_pockets_data_from_csv():
    """ Load curated pockets data. """
    return load_curated_dataset(POCKETS_CURATED_PATH_CSV)

def transform_raw_pockets_data(pockets_df):
    """ Final transformation for raw pockets data."""
    return transform_raw_df_data(pockets_df, "pockets", False, False)

def load_curate_save_pockets_data():
    """ Curates raw pockets data and saves it. """
    pockets_complete = load_raw_pockets_data()
    pockets_transformed = transform_raw_pockets_data(pockets_complete)
    save_curated_pockets_data_to_excel(pockets_transformed)
    save_curated_pockets_data_to_csv(pockets_transformed)


if __name__ == "__main__":
    load_curate_save_pockets_data()