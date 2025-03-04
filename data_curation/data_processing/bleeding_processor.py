from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_bleeding_maxillary, load_bleeding_mandibular
from deep_learning_dentistry.data_curation.data_processing.utils.functions import merge_maxillary_and_mandibular_b_s, \
    save_cleaned_data_to_excel, save_processed_data_to_csv, load_processed_dataset, load_cleaned_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (BLEEDING_CLEANED_PATH, BLEEDING_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, transform_dataset_to_clean_b_s


def curate_bleeding_data():
    """
    Loads maxillary and mandibular bleeding data from the respective sources,
    merges them into a single unified DataFrame, and returns the final curated data.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    bleeding_maxillary = load_bleeding_maxillary()
    bleeding_mandibular = load_bleeding_mandibular()
    bleeding_complete = merge_maxillary_and_mandibular_b_s(bleeding_maxillary, bleeding_mandibular)
    bleeding_updated = transform_dataset_to_clean_b_s(bleeding_complete)
    return bleeding_updated


def transform_bleeding_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    bleeding_complete = curate_bleeding_data()
    bleeding_transformed = transform_dataset(bleeding_complete, variable="bleeding")
    return bleeding_transformed


def load_cleaned_bleeding_data():
    """
    Loads the cleaned bleeding dataset from the specified file path.
    Returns the processed bleeding dataset.
    """
    return load_cleaned_dataset(BLEEDING_CLEANED_PATH)


def load_processed_bleeding_data():
    """
    Loads the processed bleeding dataset from the specified file path.
    Returns the processed bleeding dataset.
    """
    return load_processed_dataset(BLEEDING_PROCESSED_PATH)


def save_cleaned_bleeding_data_to_excel(dataframe):
    """
    Saves cleaned bleeding data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(BLEEDING_CLEANED_PATH, dataframe)


def save_processed_bleeding_data_to_csv(dataframe):
    """
    Saves processed bleeding data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(BLEEDING_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_bleeding():
    """
    Curates raw bleeding data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    bleeding_complete = curate_bleeding_data()
    save_cleaned_bleeding_data_to_excel(bleeding_complete)
    transformed_bleeding = transform_dataset(bleeding_complete, variable="bleeding")
    save_processed_bleeding_data_to_csv(transformed_bleeding)

if __name__ == "__main__":
    curate_clean_transform_process_save_bleeding()