from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_suppuration_maxillary, load_suppuration_mandibular
from deep_learning_dentistry.data_curation.data_processing.utils.functions import merge_maxillary_and_mandibular_b_s, \
    save_cleaned_data_to_excel, save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (SUPPURATION_CLEANED_PATH, SUPPURATION_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, transform_dataset_to_clean_b_s


def curate_suppuration_data():
    """
    Loads maxillary and mandibular suppuration data from the respective sources,
    merges them into a single unified DataFrame, and returns the final curated data.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    suppuration_maxillary = load_suppuration_maxillary()
    suppuration_mandibular = load_suppuration_mandibular()
    suppuration_complete = merge_maxillary_and_mandibular_b_s(suppuration_maxillary, suppuration_mandibular)
    suppuration_updated = transform_dataset_to_clean_b_s(suppuration_complete)
    return suppuration_updated


def transform_suppuration_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    suppuration_complete = curate_suppuration_data()
    suppuration_transformed = transform_dataset(suppuration_complete, variable="suppuration")
    return suppuration_transformed


def load_processed_suppuration_data():
    """
    Loads the processed suppuration dataset from the specified file path.
    """
    return load_processed_dataset(SUPPURATION_PROCESSED_PATH)


def save_cleaned_suppuration_data_to_excel(dataframe):
    """
    Saves cleaned bleeding data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(SUPPURATION_CLEANED_PATH, dataframe)


def save_processed_suppuration_data_to_csv(dataframe):
    """
    Saves processed bleeding data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(SUPPURATION_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_suppuration():
    """
    Curates raw bleeding data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    suppuration_complete = curate_suppuration_data()
    save_cleaned_suppuration_data_to_excel(suppuration_complete)
    transformed_suppuration = transform_dataset(suppuration_complete, variable="suppuration")
    save_processed_suppuration_data_to_csv(transformed_suppuration)


if __name__ == "__main__":
    curate_clean_transform_process_save_suppuration()