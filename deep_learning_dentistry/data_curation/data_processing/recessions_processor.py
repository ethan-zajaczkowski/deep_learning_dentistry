from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    merge_maxillary_and_mandibular_p_r,
    save_curated_data_to_excel,
    save_curated_data_to_csv,
    load_curated_dataset,
    transform_raw_df_data,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import (
    load_recessions,
)
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    RECESSIONS_CURATED_PATH_EXCEL,
    RECESSIONS_CURATED_PATH_CSV,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import (
    raw_to_cleaned_df_pockets_and_rec,
)


def load_raw_recessions_data():
    """ Load raw recessions data and curate it. """
    df_list = load_recessions()
    recessions_combined_raw = merge_maxillary_and_mandibular_p_r(df_list)
    recessions_combined_cleaned = raw_to_cleaned_df_pockets_and_rec(recessions_combined_raw)
    return recessions_combined_cleaned

def save_curated_recessions_data_to_excel(final_recessions_df):
    """ Saves curated recessions data to an Excel file. """
    save_curated_data_to_excel(RECESSIONS_CURATED_PATH_EXCEL, final_recessions_df)

def save_curated_recessions_data_to_csv(final_recessions_df):
    """ Saves curated recessions data as a CSV file. """
    save_curated_data_to_csv(RECESSIONS_CURATED_PATH_CSV, final_recessions_df)

def load_curated_recessions_data_from_csv():
    """ Load curated recessions data. """
    return load_curated_dataset(RECESSIONS_CURATED_PATH_CSV)

def transform_raw_recessions_data(recessions_df):
    """ Final transformation for raw recessions data."""
    return transform_raw_df_data(recessions_df, "recessions", False, False)

def load_curate_save_recessions_data():
    """ Curates raw recessions data and saves it. """
    recessions_complete = load_raw_recessions_data()
    recessions_transformed = transform_raw_recessions_data(recessions_complete)
    save_curated_recessions_data_to_excel(recessions_transformed)
    save_curated_recessions_data_to_csv(recessions_transformed)


if __name__ == "__main__":
    load_curate_save_recessions_data()