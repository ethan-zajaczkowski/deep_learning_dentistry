import re
import pandas as pd

from deep_learning_dentistry.data_curation.data_processing.index_processor import load_curated_index_data_from_csv
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    SUPPURATION_PATH,
    SUPPURATION_CURATED_PATH_EXCEL,
    SUPPURATION_CURATED_PATH_CSV,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    merge_df_for_maxillary_and_mandibular_bleeding_suppuration,
    save_curated_data_to_csv,
    save_curated_data_to_excel,
    load_curated_dataset,
    analyze_raw_df_data,
    transform_raw_df_data,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import (
    load_bleeding_or_suppuration,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import (
    raw_to_cleaned_df_bleeding_and_sup,
)


def load_raw_suppuration_data():
    """ Load raw bleeding data and curate it. """
    df_list = load_bleeding_or_suppuration(SUPPURATION_PATH)
    suppuration_combined_raw =  merge_df_for_maxillary_and_mandibular_bleeding_suppuration(df_list)
    suppuration_combined_clean = raw_to_cleaned_df_bleeding_and_sup(suppuration_combined_raw)
    return suppuration_combined_clean

def save_curated_suppuration_data_to_excel(final_suppuration_df):
    """ Saves curated suppuration data to an Excel file. """
    save_curated_data_to_excel(SUPPURATION_CURATED_PATH_EXCEL, final_suppuration_df)

def save_curated_suppuration_data_to_csv(final_suppuration_df):
    """ Saves curated suppuration data as a CSV file. """
    save_curated_data_to_csv(SUPPURATION_CURATED_PATH_CSV, final_suppuration_df)

def load_curated_suppuration_data_from_csv():
    """ Load curated suppuration data. """
    return load_curated_dataset(SUPPURATION_CURATED_PATH_CSV)

def analyze_raw_suppuration_data(raw_suppuration_df):
    """ For each exam, find the number of teeth and number of sites that have suppuration."""
    return analyze_raw_df_data(raw_suppuration_df, "suppuration")

def transform_raw_suppuration_data(suppuration_df):
    """ Final transformation for raw bleeding data."""
    return transform_raw_df_data(suppuration_df, "suppuration", True, True)

def load_curate_save_suppuration_data():
    """ Curates raw bleeding data and saves it. """
    suppuration_complete = load_raw_suppuration_data()
    suppuration_curated = analyze_raw_suppuration_data(suppuration_complete)
    suppuration_transformed = transform_raw_suppuration_data(suppuration_curated)
    save_curated_suppuration_data_to_excel(suppuration_transformed)
    save_curated_suppuration_data_to_csv(suppuration_transformed)

    ## You have to save then reload suppuration because pandas has issues with datetime conversion from csv
    index_complete = load_curated_index_data_from_csv()
    suppuration_complete = load_curated_suppuration_data_from_csv()
    new_suppuration_df = suppuration_imputation(suppuration_complete, index_complete)
    save_curated_suppuration_data_to_excel(new_suppuration_df)
    save_curated_suppuration_data_to_csv(new_suppuration_df)

def suppuration_imputation(suppuration_df, index_df):
    """ Imputes values for suppuration based on the fact the suppuration_index in index_df has values of 0. """
    merge_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']
    merged_df = pd.merge(suppuration_df, index_df, on=merge_cols, how='outer')

    supp_cols = [
        col for col in merged_df.columns
        if re.match(r"^q\d+_\d+_\w+$", col)
    ]

    mask = (
            (merged_df['suppuration_index'] == 0.0) &
            (merged_df[supp_cols].isna().all(axis=1))
    )
    merged_df.loc[mask, supp_cols] = 0

    tooth_groups = {}
    for col in supp_cols:
        match = re.match(r'(q\d+_\d+)', col)
        if match:
            tooth_id = match.group(1)  # e.g., "q4_48"
            tooth_groups.setdefault(tooth_id, []).append(col)

    cols_to_drop = [
        'bleeding_index',
        'suppuration_index',
        'plaque_index',
        'missing_teeth',
        'percent_of_bleeding_surfaces',
        'percent_of_suppuration_surfaces',
        'percent_of_plaque_surfaces',
    ]

    merged_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    merged_df.drop(columns=["num_of_suppuration_teeth", "num_of_suppuration_sites"], inplace=True, errors='ignore')

    merged_df_1 = analyze_raw_df_data(merged_df, "suppuration")
    merged_df_2 = transform_raw_df_data(merged_df_1, "suppuration", True, True)

    return merged_df_2


if __name__ == "__main__":
    load_curate_save_suppuration_data()