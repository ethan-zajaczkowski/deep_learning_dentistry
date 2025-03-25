import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_demographic_data


def plot_transition_table_matplotlib(transition_table):
    """
    Renders the given transition_table (a pandas DataFrame) as a matplotlib table.
    Reorders the table so that the rows and columns are in the order:
    High, Moderate, Low, Missing.
    """
    # Define the desired order.
    desired_order = ["High", "Moderate", "Low", "Missing"]

    # Reindex the transition_table to the desired order.
    transition_table = transition_table.reindex(index=desired_order, columns=desired_order, fill_value=0)

    # Create the plot.
    fig, ax = plt.subplots(figsize=(6, 3))  # adjust figsize as needed
    ax.axis('tight')
    ax.axis('off')

    # Create the table in the center of the figure.
    table_data = transition_table.values
    row_labels = transition_table.index
    col_labels = transition_table.columns

    table = ax.table(cellText=table_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)  # adjust as needed
    table.auto_set_column_width(col=list(range(len(col_labels))))

    plt.title("Transition Table (Matplotlib)", pad=20)
    plt.tight_layout()
    plt.show()


def plot_missing_vs_total_side_by_side(df2022, df2023):
    """
    Creates a side-by-side bar chart for each year (2022, 2023):
      - One bar for the total number of ResearchIDs,
      - One bar for the missing count in the 'Periodontal disease risk' column.
    """
    # Calculate totals and missing counts.
    total_2022 = len(df2022)
    total_2023 = len(df2023)

    missing_2022 = df2022["Periodontal disease risk"].isna().sum()
    missing_2023 = df2023["Periodontal disease risk"].isna().sum()

    # We'll create two groups on the x-axis: one for 2022, one for 2023.
    x = np.arange(2)  # [0, 1]
    width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(6, 5))

    # For each group (year), we place two bars side by side:
    # - The "total" bar is shifted slightly left,
    # - The "missing" bar is shifted slightly right.
    bars_total = ax.bar(x - width / 2,
                        [total_2022, total_2023],
                        width=width,
                        label="Total",
                        color="lightblue",
                        alpha=0.8)
    bars_missing = ax.bar(x + width / 2,
                          [missing_2022, missing_2023],
                          width=width,
                          label="Missing",
                          color="darkblue")

    # Configure the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(["2022", "2023"])
    ax.set_ylabel("Number of ResearchIDs")
    ax.set_title("Total vs. Missing in 'Periodontal disease risk' (2022 vs. 2023)")
    ax.legend()

    # Annotate each bar with its value
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", color="black", fontsize=10)

    annotate_bars(bars_total)
    annotate_bars(bars_missing)

    plt.tight_layout()
    plt.show()


def plot_risk_overlay(base_risk, other_risk):
    """
    Plots an overlay comparison of the risk distributions from two Series.

    Missing values are replaced with "Missing". The two distributions (base and other)
    are shown as grouped bars for each category in the order:
      "Missing", "Low", "Moderate", "High".
    """
    # Define the desired order for risk categories.
    desired_order = ["Missing", "Low", "Moderate", "High"]

    # Replace missing values with "Missing"
    base_filled = base_risk.fillna("Missing")
    other_filled = other_risk.fillna("Missing")

    # Count frequencies for each Series.
    base_counts = base_filled.value_counts()
    other_counts = other_filled.value_counts()

    # Ensure both Series have the same set of categories.
    categories = desired_order  # use the desired order explicitly

    # Reindex counts to follow the desired order.
    base_counts = base_counts.reindex(categories, fill_value=0)
    other_counts = other_counts.reindex(categories, fill_value=0)

    # Prepare data for grouped bar plot.
    x = np.arange(len(categories))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, base_counts.values, width, label="2022 Risk Classes", color="blue", alpha=0.7)
    rects2 = ax.bar(x + width / 2, other_counts.values, width, label="2023 Risk Classes", color="red", alpha=0.7)

    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Count")
    ax.set_title("Overlay Comparison of Risk Distributions")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Annotate each bar with the count.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", color="black")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


def demographic_comparisons(df_list):
    """
    Compares the 'Periodontal disease risk' column in two DataFrames and prints:
      - Total rows in merged data
      - Number of non-missing and missing cells
      - A transition table showing how risk values changed.

    The risk values are assumed to be "Low", "Moderate", "High".
    Then, it plots an overlay bar chart comparing the risk distributions from the base
    and the other DataFrame in the order:
         Missing, Low, Moderate, High.
    """
    df1, df2 = df_list

    if len(df1) >= len(df2):
        base_df, other_df = df1, df2
    else:
        base_df, other_df = df2, df1

    # Set ResearchID as index for both DataFrames.
    base_df_indexed = base_df.set_index('ResearchID')
    other_df_indexed = other_df.set_index('ResearchID')

    # Merge the two DataFrames using combine_first, then reset index.
    merged = base_df_indexed.combine_first(other_df_indexed)
    merged.reset_index(inplace=True)

    # Reindex the risk columns to have the same order as in merged.
    base_risk = base_df_indexed["Periodontal disease risk"].reindex(merged["ResearchID"])
    other_risk = other_df_indexed["Periodontal disease risk"].reindex(merged["ResearchID"])

    count_filled = ((base_risk.isna()) & (other_risk.notna())).sum()
    total_rows = len(merged)
    non_missing = merged["Periodontal disease risk"].notna().sum()
    missing = total_rows - non_missing

    print(f"Total rows in merged data: {total_rows}")
    print(f"Cells with data (non-missing) in 'Periodontal disease risk': {non_missing}")
    print(f"Cells that are empty in 'Periodontal disease risk': {missing}")
    print(f"Cells filled from the other DataFrame in 'Periodontal disease risk': {count_filled}\n")

    # Replace missing values with "Missing" in both risk Series.
    base_filled = base_risk.fillna("Missing")
    other_filled = other_risk.fillna("Missing")

    # Create a crosstab to show transitions with new labels.
    transition_table = pd.crosstab(base_filled, other_filled,
                                   rownames=["2022 Risk Classes"],
                                   colnames=["2023 Risk Classes"])
    print("Transition Table:")
    print(transition_table)

    # Plot overlay comparison of risk distributions.
    plot_risk_overlay(base_risk, other_risk)

    return transition_table


def plot_risk_transition_combinations(risk_2022, risk_2023):
    """
    Given two Series for 2022 and 2023 risk classes (which can be "Low", "Moderate", "High", or missing),
    this function creates and plots a horizontal bar chart of the frequency distribution of all possible transitions,
    in the order: Missing, Low, Moderate, High.

    Each bar is labeled with the count and its percentage of the total.
    """
    # Define desired order.
    desired_order = ["Missing", "Low", "Moderate", "High"]

    # Replace missing values with "Missing"
    risk_2022_filled = risk_2022.fillna("Missing")
    risk_2023_filled = risk_2023.fillna("Missing")

    # Create a crosstab with the desired order for both rows and columns.
    ctab = pd.crosstab(risk_2022_filled, risk_2023_filled,
                       rownames=["2022 Risk Classes"],
                       colnames=["2023 Risk Classes"])
    ctab = ctab.reindex(index=desired_order, columns=desired_order, fill_value=0)

    # Flatten the crosstab into a DataFrame.
    flat = ctab.stack().reset_index(name="count")
    flat["transition"] = flat.apply(lambda row: f"{row['2022 Risk Classes']} -> {row['2023 Risk Classes']}", axis=1)

    # Create a custom ordering for the transitions.
    order_mapping = {risk: i for i, risk in enumerate(desired_order)}
    flat["order"] = (flat["2022 Risk Classes"].map(order_mapping) * len(desired_order)
                     + flat["2023 Risk Classes"].map(order_mapping))
    flat = flat.sort_values("order")

    # Plot a horizontal bar chart.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(flat["transition"], flat["count"], color="skyblue")
    ax.set_xlabel("Number of Exams")
    ax.set_title("Distribution of Risk Transitions (2022 -> 2023)")

    # Annotate each bar with the count and percentage.
    total = flat["count"].sum()
    for idx, row in flat.iterrows():
        ax.text(row["count"] + 1, idx,
                f"{row['count']} ({row['count'] / total * 100:.1f}%)",
                va='center', color="black")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load the data
    df_list = load_demographic_data()

    # Extract 2022 and 2023 DataFrames
    df2022, df2023 = df_list

    # 1. Plot side-by-side total vs. missing
    plot_missing_vs_total_side_by_side(df2022, df2023)

    # 2. Print summary & transitions table + overlay chart
    table = demographic_comparisons(df_list)

    # 3. Plot horizontal transitions chart
    plot_risk_transition_combinations(df2022["Periodontal disease risk"],
                                      df2023["Periodontal disease risk"])

    # 4. Transition table
    plot_transition_table_matplotlib(table)