from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_mobility
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (MOBILITY_CLEANED_PATH, MOBILITY_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset


def curate_mobility_data():
    """
    Loads mobility data from the respective sources,
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    mobility_data = load_mobility()
    return mobility_data


def transform_mobility_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    mobility_complete = curate_mobility_data()
    mobility_transformed = transform_dataset(mobility_complete, variable="mobility")
    return mobility_transformed


def load_processed_mobility_data():
    """
    Loads the processed mobility dataset from the specified file path.
    """
    return load_processed_dataset(MOBILITY_PROCESSED_PATH)


def save_cleaned_mobility_data_to_excel(dataframe):
    """
    Saves cleaned mobility data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(MOBILITY_CLEANED_PATH, dataframe)


def save_processed_mobility_data_to_csv(dataframe):
    """
    Saves processed mobility data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(MOBILITY_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_mobility():
    """
    Curates raw mobility data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    mobility_complete = curate_mobility_data()
    save_cleaned_mobility_data_to_excel(mobility_complete)
    transformed_mobility = transform_dataset(mobility_complete, variable="mobility")
    save_processed_mobility_data_to_csv(transformed_mobility)


if __name__ == "__main__":
    curate_clean_transform_process_save_mobility()