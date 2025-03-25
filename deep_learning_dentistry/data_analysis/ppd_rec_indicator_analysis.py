import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deep_learning_dentistry.data_curation.data_processing.utils.config import FULL_DATASET_PATH, ANALYSIS_DATASET_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import load_curated_dataset, \
    save_curated_data_to_csv


def load_dataset() -> pd.DataFrame:
    """Import the specified cleaned dataset."""
    df = load_curated_dataset(FULL_DATASET_PATH, "full_dataset_long.csv")
    return df


def create_has_data_columns(df: pd.DataFrame, categorical_var: str, pockets_or_recession: str) -> pd.DataFrame:
    """
    Compute a 'has_data' column based on categorical columns that end with f", {categorical_var}".
    For each row, if any of these columns is not NA, then has_data is recorded as 1;
    otherwise, it is 0.

    Returns a DataFrame containing only the 'has_data' column and all columns ending with
    f", {pockets_or_recession}".
    """
    cat_suffix = f", {categorical_var}"
    p_r_suffix = f", {pockets_or_recession}"

    # Identify columns ending with the specified suffixes.
    cat_cols = [col for col in df.columns if col.endswith(cat_suffix)]
    p_r_cols = [col for col in df.columns if col.endswith(p_r_suffix)]

    # For each row, mark has_data as 1 if any categorical column (cat_suffix) is not NA.
    df["has_data"] = df[cat_cols].apply(lambda row: 1 if row.notna().any() else 0, axis=1)

    # Return only the 'has_data' column and the pocket (or recession) columns.
    selected_columns = ["has_data"] + p_r_cols
    return df[selected_columns]


def custom_ordering(value_counts: pd.Series) -> list:
    """
    Create a custom ordering for the x-axis:
      - Remove "Error" entirely.
      - Sort non-numeric values (excluding "Missing") alphabetically.
      - Sort numeric values in ascending order.
      - Place "Missing" at the far right.
    """
    # Drop "Error" if present.
    if "Error" in value_counts.index:
        value_counts = value_counts.drop("Error")

    def is_floatable(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    all_keys = set(value_counts.index)
    non_numeric = [x for x in all_keys if not is_floatable(x)]
    numeric = [x for x in all_keys if is_floatable(x)]

    non_numeric_sorted = sorted([x for x in non_numeric if x != "Missing"])
    numeric_sorted = sorted(numeric, key=lambda x: float(x))
    final_order = non_numeric_sorted + numeric_sorted
    if "Missing" in non_numeric:
        final_order.append("Missing")
    return final_order


def plot_has_data_histograms(df_new: pd.DataFrame, variable) -> tuple:
    """
    Creates histogram plots for the pocket values (all columns except 'has_data')
    conditioned on has_data == 0 and has_data == 1.

    - Removes any occurrence of "Error".
    - Custom ordering is used to place "Missing" at the far right of the x-axis.

    Returns a tuple of figure objects: (fig_zero, fig_one, fig_combined).
    """
    # Identify pocket columns (all except 'has_data').
    pocket_cols = [col for col in df_new.columns if col != "has_data"]

    # Split the DataFrame by has_data condition.
    df_zero = df_new[df_new["has_data"] == 0]
    df_one = df_new[df_new["has_data"] == 1]

    # Flatten the values in the pocket columns for each condition.
    values_zero = pd.Series(df_zero[pocket_cols].values.flatten()).dropna().astype(str)
    values_one = pd.Series(df_one[pocket_cols].values.flatten()).dropna().astype(str)

    # Remove "Error" values.
    values_zero = values_zero[values_zero != "Error"]
    values_one = values_one[values_one != "Error"]

    # Compute frequency counts.
    counts_zero = values_zero.value_counts()
    counts_one = values_one.value_counts()

    # Create custom x-axis ordering based on the union of keys from both counts.
    combined_keys = set(counts_zero.index).union(set(counts_one.index))
    combined_series = pd.Series(index=combined_keys).fillna(0)
    final_order = custom_ordering(combined_series)

    # Reindex frequency counts to follow the custom order.
    counts_zero = counts_zero.reindex(final_order).fillna(0)
    counts_one = counts_one.reindex(final_order).fillna(0)

    indices = np.arange(len(final_order))
    bar_width = 0.6

    # Plot for has_data == 0.
    fig_zero = plt.figure(figsize=(10, 6))
    plt.bar(indices, counts_zero.values, width=bar_width, color='blue', edgecolor="black")
    plt.xlabel(f"{variable} Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {variable} Values (has_data == 0)")
    plt.xticks(indices, final_order, rotation=45, ha="right")
    plt.tight_layout()

    # Plot for has_data == 1.
    fig_one = plt.figure(figsize=(10, 6))
    plt.bar(indices, counts_one.values, width=bar_width, color='red', edgecolor="black")
    plt.xlabel(f"{variable} Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {variable} Values (has_data == 1)")
    plt.xticks(indices, final_order, rotation=45, ha="right")
    plt.tight_layout()

    # Combined grouped bar chart.
    fig_combined = plt.figure(figsize=(12, 7))
    plt.bar(indices - bar_width / 4, counts_zero.values, width=bar_width / 2,
            color='blue', edgecolor="black", label='has_data == 0')
    plt.bar(indices + bar_width / 4, counts_one.values, width=bar_width / 2,
            color='red', edgecolor="black", label='has_data == 1')
    plt.xlabel(f"{variable} Value")
    plt.ylabel("Frequency")
    plt.title(f"Combined Histogram of {variable} Values by has_data")
    plt.xticks(indices, final_order, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    return fig_zero, fig_one, fig_combined


if __name__ == "__main__":
    # Load dataset.
    df = load_dataset()

    # Create new DataFrame with has_data and pocket columns.
    df_new = create_has_data_columns(df, categorical_var="furcation", pockets_or_recession="recession")

    save_curated_data_to_csv(ANALYSIS_DATASET_PATH, df_new, "ppd_rec_indicator_analysis.xlsx")

    # Generate the histogram figures.
    fig_zero, fig_one, fig_combined = plot_has_data_histograms(df_new, "Recession")

    # Optionally display the figures.
    fig_zero.show()
    fig_one.show()
    fig_combined.show()