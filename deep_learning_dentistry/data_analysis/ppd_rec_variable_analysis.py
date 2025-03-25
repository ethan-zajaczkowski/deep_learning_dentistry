import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deep_learning_dentistry.data_curation.data_processing.utils.config import FULL_DATASET_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import load_curated_dataset


def load_dataset() -> pd.DataFrame:
    """Import the specified cleaned dataset."""
    df = load_curated_dataset(FULL_DATASET_PATH, "full_dataset_long.csv")
    return df


def filter_pocket_columns(df: pd.DataFrame, target_integer: int, categorical_var: str,
                          pockets_or_recession: str) -> pd.DataFrame:
    """
    Filter the DataFrame so that for each column ending with the specified categorical variable,
    if the value equals target_integer then the corresponding pocket column is kept;
    otherwise, both the categorical column and the corresponding pocket column are set to NA.
    Returns a DataFrame containing only the processed pocket columns.
    """
    pocket_cols = []
    cat_var = f", {categorical_var}"
    p_or_r = f", {pockets_or_recession}"

    for col in df.columns:
        if col.endswith(cat_var):
            prefix = col[:-len(cat_var)]
            pocket_col = prefix + p_or_r
        else:
            continue

        if pocket_col in df.columns:
            mask = df[col] != target_integer
            df.loc[mask, col] = pd.NA
            df.loc[mask, pocket_col] = pd.NA
            pocket_cols.append(pocket_col)

    return df[list(set(pocket_cols))]


def custom_ordering(value_counts: pd.Series) -> list:
    """
    Create a custom ordering for the x-axis:
      - Remove "Error" entirely.
      - Sort non-numeric values (excluding "Missing") alphabetically.
      - Sort numeric values in ascending order.
      - Place "Missing" at the far right.
    """
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


def plot_pocket_histograms(df_zero: pd.DataFrame, df_one: pd.DataFrame) -> tuple:
    """
    Given two DataFrames (one filtered for bleeding==0 and one for bleeding==1),
    this function produces three figures:
      1. A histogram (bar chart) of pocket values for bleeding == 0.
      2. A histogram (bar chart) of pocket values for bleeding == 1.
      3. A combined grouped bar chart comparing both conditions, with the blue (bleeding==0)
         bars to the left of the red (bleeding==1) bars.
    Returns a tuple of figure objects: (fig_zero, fig_one, fig_combined).
    """
    # Flatten the pocket columns for each condition.
    pocket_zero = pd.Series(df_zero.values.flatten()).dropna().astype(str)
    pocket_one = pd.Series(df_one.values.flatten()).dropna().astype(str)

    # Compute frequency counts.
    counts_zero = pocket_zero.value_counts()
    counts_one = pocket_one.value_counts()

    # Create a custom x-axis ordering based on the union of keys.
    all_keys = set(counts_zero.index).union(set(counts_one.index))
    combined_counts = pd.Series(index=all_keys).fillna(0)
    final_order = custom_ordering(combined_counts)

    # Reindex frequency counts to the custom order.
    counts_zero = counts_zero.reindex(final_order).fillna(0)
    counts_one = counts_one.reindex(final_order).fillna(0)

    indices = np.arange(len(final_order))
    bar_width = 0.6

    # Figure for Bleeding == 0.
    fig_zero = plt.figure(figsize=(10, 6))
    plt.bar(indices, counts_zero.values, width=bar_width, color='blue', edgecolor="black")
    plt.xlabel("Recession Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Recession Values (Furcation == 0)")
    plt.xticks(indices, final_order, rotation=45, ha="right")
    plt.tight_layout()

    # Figure for Bleeding == 1.
    fig_one = plt.figure(figsize=(10, 6))
    plt.bar(indices, counts_one.values, width=bar_width, color='red', edgecolor="black")
    plt.xlabel("Recession Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Recession Values (Furcation == 1)")
    plt.xticks(indices, final_order, rotation=45, ha="right")
    plt.tight_layout()

    # Combined grouped bar chart.
    fig_combined = plt.figure(figsize=(12, 7))
    # Position blue bars (bleeding == 0) to the left and red bars (bleeding == 1) to the right.
    plt.bar(indices - bar_width / 4, counts_zero.values, width=bar_width / 2,
            color='blue', edgecolor="black", label='Furcation == 0')
    plt.bar(indices + bar_width / 4, counts_one.values, width=bar_width / 2,
            color='red', edgecolor="black", label='Furcation == 1')
    plt.xlabel("Recession Value")
    plt.ylabel("Frequency")
    plt.title("Combined Histogram of Recession Values (Furcation == 0 & 1)")
    plt.xticks(indices, final_order, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    return fig_zero, fig_one, fig_combined


if __name__ == "__main__":
    # Load dataset.
    df = load_dataset()

    # Filter pocket columns for bleeding==0 and bleeding==1.
    df_zero = filter_pocket_columns(df.copy(), target_integer=0, categorical_var="bleeding",
                                    pockets_or_recession="pocket")
    df_one = filter_pocket_columns(df.copy(), target_integer=1, categorical_var="bleeding",
                                   pockets_or_recession="pocket")

    # Generate the three histogram figures.
    fig_zero, fig_one, fig_combined = plot_pocket_histograms(df_zero, df_one)

    # Optionally, display the figures.
    fig_zero.show()
    fig_one.show()
    fig_combined.show()