import os
import webbrowser

import pandas as pd

from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_chart_restorative
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import (CHART_RESTORE_PROCESSED_PATH, CHART_RESTORE_CLEANED_PATH)
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, transform_dataset_to_clean_chart_restore


def curate_chart_restorative():
    """
    Loads maxillary and mandibular bleeding data from the respective sources,
    merges them into a single unified DataFrame, and returns the final curated data.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    chart_restorative_complete = load_chart_restorative()
    chart_restorative_transform = transform_dataset_to_clean_chart_restore(chart_restorative_complete)
    return chart_restorative_transform


def load_processed_chart_restorative_data():
    """
    Loads the processed chart restorative dataset from the specified file path.
    """
    return load_processed_dataset(CHART_RESTORE_PROCESSED_PATH)


def transform_chart_restorative_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    chart_restorative_complete = curate_chart_restorative()
    chart_restorative_transformed = transform_dataset(chart_restorative_complete, variable="chart_restore")
    return chart_restorative_transformed


def save_cleaned_chart_restorative_data_to_excel(dataframe):
    """
    Saves cleaned chart_endo data DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(CHART_RESTORE_CLEANED_PATH, dataframe)


def save_processed_chart_restorative_data_to_csv(dataframe):
    """
    Saves processed bleeding data DataFrame to a CSV file.
    """
    save_processed_data_to_csv(CHART_RESTORE_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_restore():
    """
    Curates raw bleeding data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    chart_restorative_complete = curate_chart_restorative()
    save_cleaned_chart_restorative_data_to_excel(chart_restorative_complete)
    transformed_restore = transform_dataset(chart_restorative_complete, variable="chart_restore")
    merged_restore = post_process_chart_restore_1(transformed_restore)
    propagate_restore = post_process_chart_restore_2(merged_restore)
    save_processed_chart_restorative_data_to_csv(propagate_restore)


def post_process_chart_restore_1(df):
    """
    For each column in columns_to_fix, replaces 0 with the first non-zero
    value found in the same group (defined by group_cols).
    """
    df = df.copy()

    # 2) Add a helper index to preserve original row order
    df["original_index"] = df.index

    # 3) For each column that can be 0 or string, convert 0 to NaN
    value_columns = [
        "tooth_notes_restorative",
        "has_bridge_retainer_3/4_Crown",
        "has_bridge_retainer_veneer",
        "has_veneer",
        "has_fillings_or_caries",
        "has_inlay",
        "restored_material"
    ]

    for col in value_columns:
        df[f"{col}_str"] = df[col].where(df[col].ne(0))

    # 4) Group by the relevant columns for your "exam" logic
    group_cols = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site"]

    # Build the aggregation dictionary: for each col, pick 'first' non-null
    agg_dict = {
        "original_index": "min"  # We'll use this to restore order
    }
    for col in value_columns:
        agg_dict[f"{col}_str"] = "first"

    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(agg_dict)
    )

    # 5) Fill NaN back with 0 if that group had no non-null values
    for col in value_columns:
        grouped[f"{col}_str"] = grouped[f"{col}_str"].fillna(0)

    # 6) Sort by earliest appearance
    grouped.sort_values("original_index", inplace=True)
    grouped.reset_index(drop=True, inplace=True)

    # 7) Rename columns back and drop helper columns
    for col in value_columns:
        grouped.rename(columns={f"{col}_str": col}, inplace=True)
    grouped.drop(columns="original_index", inplace=True)

    return grouped


def post_process_chart_restore_2(df):
    """
    For each research_id, and for each tooth group (defined by tooth_quadrant, tooth_id, tooth_site),
    applies cummax on the specified columns so that once a 1 is encountered, every subsequent value becomes 1.
    Assumes the DataFrame is already in chronological order.
    """
    def variable_helper(df_research_id):
        # Grouping keys for each tooth
        tooth_index = ["tooth_quadrant", "tooth_id", "tooth_site"]
        columns_to_match = [
            "has_bridge_retainer_3/4_Crown",
            "has_bridge_retainer_veneer",
            "has_veneer",
            "has_fillings_or_caries",
            "has_inlay"
        ]

        # For each column, within each tooth group, apply cummax
        for col in columns_to_match:
            df_research_id[col] = df_research_id.groupby(tooth_index)[col].cummax()
        return df_research_id

    grouped = df.groupby("research_id", group_keys=False).apply(variable_helper)
    return grouped


if __name__ == "__main__":
    curate_clean_transform_process_save_restore()