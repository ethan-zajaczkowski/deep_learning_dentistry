from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_chart_endo
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (CHART_ENDO_PROCESSED_PATH, CHART_ENDO_CLEANED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, \
    transform_dataset_to_clean_chart_endo


def curate_chart_endo():
    """
    Loads maxillary and mandibular bleeding data from the respective sources,
    merges them into a single unified DataFrame, and returns the final curated data.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    chart_endo_complete = load_chart_endo()
    chart_endo_transform = transform_dataset_to_clean_chart_endo(chart_endo_complete)
    return chart_endo_transform


def load_processed_chart_endo_data():
    """
    Loads the processed chart endo dataset from the specified file path.
    """
    return load_processed_dataset(CHART_ENDO_PROCESSED_PATH)


def transform_chart_endo_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    chart_endo_complete = curate_chart_endo()
    chart_endo_transformed = transform_dataset(chart_endo_complete, variable="chart_endo")
    return chart_endo_transformed


def save_cleaned_chart_endo_data_to_excel(dataframe):
    """
    Saves cleaned chart_endo data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(CHART_ENDO_CLEANED_PATH, dataframe)


def save_processed_chart_endo_data_to_csv(dataframe):
    """
    Saves processed bleeding data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(CHART_ENDO_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_bleeding():
    """
    Curates raw bleeding data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    chart_endo_complete = curate_chart_endo()
    save_cleaned_chart_endo_data_to_excel(chart_endo_complete)
    transformed_bleeding = transform_dataset(chart_endo_complete, variable="chart_endo")
    save_processed_chart_endo_data_to_csv(transformed_bleeding)


if __name__ == "__main__":
    curate_clean_transform_process_save_bleeding()