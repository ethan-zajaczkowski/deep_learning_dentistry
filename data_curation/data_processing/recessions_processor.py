from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_recession_maxillary, load_recession_mandibular
from deep_learning_dentistry.data_curation.data_processing.utils.functions import merge_maxillary_and_mandibular_p_r, \
    save_cleaned_data_to_excel, save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (RECESSIONS_CLEANED_PATH, RECESSIONS_PROCESSED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset


def curate_recessions_data():
    """
    Loads maxillary and mandibular recessions data from the respective sources,
    merges them into a single unified DataFrame, and returns the final curated data.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    recessions_maxillary = load_recession_maxillary()
    recessions_mandibular = load_recession_mandibular()
    recessions_complete = merge_maxillary_and_mandibular_p_r(recessions_maxillary, recessions_mandibular)
    return recessions_complete


def transform_recessions_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    recessions_complete = curate_recessions_data()
    recessions_transformed = transform_dataset(recessions_complete, variable="recessions")
    return recessions_transformed


def load_processed_recessions_data():
    """
    Loads the processed recessions dataset from the specified file path.
    """
    return load_processed_dataset(RECESSIONS_PROCESSED_PATH)


def save_cleaned_recessions_data_to_excel(dataframe):
    """
    Saves cleaned recessions data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(RECESSIONS_CLEANED_PATH, dataframe)


def save_processed_recessions_data_to_csv(dataframe):
    """
    Saves processed recessions data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(RECESSIONS_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_recessions():
    """
    Curates raw recessions data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    recessions_complete = curate_recessions_data()
    save_cleaned_recessions_data_to_excel(recessions_complete)
    transformed_recessions = transform_dataset(recessions_complete, variable="recessions")
    save_processed_recessions_data_to_csv(transformed_recessions)


if __name__ == "__main__":
    curate_clean_transform_process_save_recessions()