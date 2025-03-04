from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_pockets_maxillary, load_pockets_mandibular
from deep_learning_dentistry.data_curation.data_processing.utils.functions import merge_maxillary_and_mandibular_p_r, \
    save_cleaned_data_to_excel, save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (POCKETS_CLEANED_PATH, POCKETS_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset


def curate_pockets_data():
    """
    Loads maxillary and mandibular pockets data from the respective sources,
    merges them into a single unified DataFrame, and returns the final curated data.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    pockets_maxillary = load_pockets_maxillary()
    pockets_mandibular = load_pockets_mandibular()
    pockets_complete = merge_maxillary_and_mandibular_p_r(pockets_maxillary, pockets_mandibular)
    return pockets_complete


def transform_pockets_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    pockets_complete = curate_pockets_data()
    pockets_transformed = transform_dataset(pockets_complete, variable="pockets")
    return pockets_transformed


def load_processed_pockets_data():
    """
    Loads the processed pockets dataset from the specified file path.
    """
    return load_processed_dataset(POCKETS_PROCESSED_PATH)


def save_cleaned_pockets_data_to_excel(dataframe):
    """
    Saves cleaned pockets data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(POCKETS_CLEANED_PATH, dataframe)


def save_processed_pockets_data_to_csv(dataframe):
    """
    Saves processed pockets data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(POCKETS_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_pockets():
    """
    Curates raw pockets data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    pockets_complete = curate_pockets_data()
    save_cleaned_pockets_data_to_excel(pockets_complete)
    transformed_pockets = transform_dataset(pockets_complete, variable="pockets")
    save_processed_pockets_data_to_csv(transformed_pockets)


if __name__ == "__main__":
    curate_clean_transform_process_save_pockets()