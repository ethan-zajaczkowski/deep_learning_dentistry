import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import merge_df_normal, \
    save_curated_data_to_excel, save_curated_data_to_csv, load_curated_dataset, transform_raw_df_data
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_chart_restorative
from deep_learning_dentistry.data_curation.data_processing.utils.config import (CHART_RESTORE_CURATED_PATH_CSV,
                                                                                CHART_RESTORE_CURATED_PATH_EXCEL)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import \
    transform_dataset_to_clean_chart_restore


def load_chart_restorative_data():
    """ Load raw chart_restore data and curate it. """
    df_list = load_chart_restorative()
    chart_restore_raw = merge_df_normal(df_list)
    chart_restorative_clean = transform_dataset_to_clean_chart_restore(chart_restore_raw)
    return chart_restorative_clean

def save_curated_chart_restore_data_to_excel(final_chart_restore_df):
    """ Saves curated chart_restore data to an Excel file. """
    save_curated_data_to_excel(CHART_RESTORE_CURATED_PATH_EXCEL, final_chart_restore_df)

def save_curated_chart_restore_data_to_csv(final_chart_restore_df):
    """ Saves curated chart_restore data as a CSV file. """
    save_curated_data_to_csv(CHART_RESTORE_CURATED_PATH_CSV, final_chart_restore_df)

def load_curated_chart_restore_data_from_csv():
    """ Load curated chart_restore data. """
    return load_curated_dataset(CHART_RESTORE_CURATED_PATH_CSV)

def transform_raw_chart_restore_data(chart_restore_df):
    """Final transformation for chart_restore data."""
    return (
        transform_raw_df_data(chart_restore_df, "chart_restore", False, False)
        .rename(columns={"Complete Notes": "restoration_notes"})
        .drop(columns=["TOOTH CONDITION", "TOOTH NBR & NOTES", "MATERIAL"])
    )

def load_curate_save_chart_restore_data():
    """ Curates raw bleeding data and saves it. """
    chart_restore_complete = load_chart_restorative_data()
    chart_restore_transformed = transform_raw_chart_restore_data(chart_restore_complete)
    save_curated_chart_restore_data_to_excel(chart_restore_transformed)
    save_curated_chart_restore_data_to_csv(chart_restore_transformed)


if __name__ == "__main__":
    load_curate_save_chart_restore_data()