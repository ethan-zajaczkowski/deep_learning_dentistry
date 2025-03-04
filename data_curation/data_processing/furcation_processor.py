from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_furcation
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (FURCATION_CLEANED_PATH, FURCATION_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, \
    transform_dataset_to_clean_furcation


def curate_furcation_data():
    """
    Loads furcation data from the respective sources.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    furcation_data = load_furcation()
    furcation_transformed = transform_dataset_to_clean_furcation(furcation_data)
    return furcation_transformed


def transform_furcation_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    furcation_complete = curate_furcation_data()
    furcation_transformed = transform_dataset(furcation_complete, variable="furcation")
    return furcation_transformed


def load_processed_furcation_data():
    """
    Loads the processed bleeding dataset from the specified file path.
    Returns the processed bleeding dataset.
    """
    return load_processed_dataset(FURCATION_PROCESSED_PATH)


def save_cleaned_furcation_data_to_excel(dataframe):
    """
    Saves cleaned furcation data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(FURCATION_CLEANED_PATH, dataframe)


def save_processed_furcation_data_to_csv(dataframe):
    """
    Saves processed furcation data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(FURCATION_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_furcation():
    """
    Curates raw furcation data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    furcation_complete = curate_furcation_data()
    save_cleaned_furcation_data_to_excel(furcation_complete)
    transformed_furcation = transform_dataset(furcation_complete, variable="furcation")
    save_processed_furcation_data_to_csv(transformed_furcation)


if __name__ == "__main__":
    curate_clean_transform_process_save_furcation()