from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_demographic_data
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset, fix_na_except_date
from deep_learning_dentistry.data_curation.data_processing.utils.config import DEMOGRAPHIC_CLEANED_PATH, DEMOGRAPHIC_PROCESSED_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset


def curate_demographics_data():
    """
    Loads demographics data from the respective sources,
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    demographics_data = load_demographic_data()
    demographics_data = fix_na_except_date(demographics_data)
    return demographics_data


def transform_demographics_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    demographics_data_complete = curate_demographics_data()
    demographics_data_transformed = transform_dataset(demographics_data_complete, variable="demographic_data")
    return demographics_data_transformed


def load_processed_demographic_data():
    """
    Loads the processed demographic dataset from the specified file path.
    """
    return load_processed_dataset(DEMOGRAPHIC_PROCESSED_PATH)


def save_cleaned_demographics_data_data_to_excel(dataframe):
    """
    Saves cleaned demographics_data data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(DEMOGRAPHIC_CLEANED_PATH, dataframe)


def save_processed_demographics_data_data_to_csv(dataframe):
    """
    Saves processed demographics_data data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(DEMOGRAPHIC_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_demographics_data():
    """
    Curates raw demographics_data data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    demographics_data_complete = curate_demographics_data()
    save_cleaned_demographics_data_data_to_excel(demographics_data_complete)
    demographics_data_transformed = transform_dataset(demographics_data_complete, variable="demographic_data")
    save_processed_demographics_data_data_to_csv(demographics_data_transformed)


if __name__ == "__main__":
    curate_clean_transform_process_save_demographics_data()