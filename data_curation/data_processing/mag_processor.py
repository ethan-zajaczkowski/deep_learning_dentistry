from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_mag
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (MAG_CLEANED_PATH, MAG_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, \
    transform_dataset_to_clean_mag


def curate_mag_data():
    """
    Loads MAG data from the respective sources,
    """
    mag_data = load_mag()
    mag_data_transformed = transform_dataset_to_clean_mag(mag_data)
    return mag_data_transformed


def transform_mag_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    mag_complete = curate_mag_data()
    mag_transformed = transform_dataset(mag_complete, variable="mag")
    return mag_transformed


def load_processed_mag_data():
    """
    Loads the processed MAG dataset from the specified file path.
    """
    return load_processed_dataset(MAG_PROCESSED_PATH)


def save_cleaned_mag_data_to_excel(dataframe):
    """
    Saves cleaned MAG data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(MAG_CLEANED_PATH, dataframe)


def save_processed_mag_data_to_csv(dataframe):
    """
    Saves processed MAG data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(MAG_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_mag():
    """
    Curates raw MAG data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    mag_complete = curate_mag_data()
    save_cleaned_mag_data_to_excel(mag_complete)
    transformed_mag = transform_dataset(mag_complete, variable="mag")
    save_processed_mag_data_to_csv(transformed_mag)


if __name__ == "__main__":
    curate_clean_transform_process_save_mag()
