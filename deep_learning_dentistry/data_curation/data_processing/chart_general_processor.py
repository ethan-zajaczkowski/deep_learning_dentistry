from collections import defaultdict

from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import merge_df_normal, \
    save_curated_data_to_csv, save_curated_data_to_excel, load_curated_dataset, transform_raw_df_data, \
    analyze_raw_df_data
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_chart_general
from deep_learning_dentistry.data_curation.data_processing.utils.config import CHART_GENERAL_CURATED_PATH_EXCEL, CHART_GENERAL_CURATED_PATH_CSV
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import raw_to_cleaned_df_chart_general

def load_chart_general_data():
    """ Load raw chart_general data and curate it. """
    df_list = load_chart_general()
    chart_general_raw = merge_df_normal(df_list)
    chart_general_clean = raw_to_cleaned_df_chart_general(chart_general_raw)
    return chart_general_clean

def save_curated_chart_general_data_to_excel(final_chart_general_df):
    """ Saves curated chart_general data to an Excel file. """
    save_curated_data_to_excel(CHART_GENERAL_CURATED_PATH_EXCEL, final_chart_general_df)

def save_curated_chart_general_data_to_csv(final_chart_restore_df):
    """ Saves curated chart_general data as a CSV file. """
    save_curated_data_to_csv(CHART_GENERAL_CURATED_PATH_CSV, final_chart_restore_df)

def load_curated_chart_general_data_from_csv():
    """ Load curated chart_general data. """
    return load_curated_dataset(CHART_GENERAL_CURATED_PATH_CSV)

def analyze_raw_chart_general_data(raw_chart_general_df):
    """
    """
    df = raw_chart_general_df.copy()
    num_of_variable_on_teeth = "num_of_missing_teeth"

    def count_missing_teeth(row):
        count = 0
        for col in row.index:
            if col.endswith("_missing_status"):
                if float(row[col]) == 1:
                    count += 1
        return count

    df[num_of_variable_on_teeth] = df.apply(count_missing_teeth, axis=1)
    return df

def transform_raw_chart_restore_data(chart_restore_df):
    """Final transformation for chart_general data."""
    df = transform_raw_df_data(chart_restore_df, "missing", True, False)
    df = df.drop(columns=["TREATMENT"])
    return df

def load_curate_save_chart_general_data():
    """ Curates raw chart_general data and saves it. """
    chart_general_complete = load_chart_general_data()
    chart_general_curated = analyze_raw_chart_general_data(chart_general_complete)
    chart_general_transformed = transform_raw_chart_restore_data(chart_general_curated)
    save_curated_chart_general_data_to_excel(chart_general_transformed)
    save_curated_chart_general_data_to_csv(chart_general_transformed)

if __name__ == "__main__":
    load_curate_save_chart_general_data()