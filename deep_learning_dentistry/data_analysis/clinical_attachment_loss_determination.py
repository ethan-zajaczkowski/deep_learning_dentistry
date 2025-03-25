import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deep_learning_dentistry.data_curation.dataset_curator_long import load_curated_full_dataset_long_from_csv


# from deep_learning_dentistry.data_curation.dataset_curator_long import load_curated_full_dataset_long_from_csv

def classify_exam(df_exam):
    """
    Given a DataFrame for one exam (all sites for that exam),
    classify periodontitis severity as "High", "Moderate", "Low", or "None".

    Aggregates site-level measurements to the tooth level using the maximum CAL and pocket values.
    Thresholds (in mm) are applied as follows:
      - High: ≥2 teeth with CAL ≥ 6 and ≥1 tooth with pocket ≥ 5.
      - Moderate: ≥2 teeth with CAL ≥ 4 or ≥2 teeth with pocket ≥ 5.
      - Low: (≥2 teeth with CAL ≥ 3 and ≥2 teeth with pocket ≥ 4) or if any site has pocket ≥ 5.

    Before applying these thresholds, sites with site_id in ["L", "B", "P"] are excluded.
    """
    # Ensure that "pocket" and "cal" are numeric
    df_exam["pocket"] = pd.to_numeric(df_exam["pocket"], errors="coerce")
    df_exam["cal"] = pd.to_numeric(df_exam["cal"], errors="coerce")

    # Exclude sites with site_id in ["L", "B", "P"]
    df_filtered = df_exam[~df_exam["site_id"].isin(["L", "B", "P"])]

    # High: ≥2 teeth with CAL ≥ 6 and ≥1 tooth with pocket ≥ 5.
    severe_tooth_ids = df_filtered.loc[df_filtered["cal"] >= 6, "tooth_id"].unique()
    pocket5_tooth_ids = df_filtered.loc[df_filtered["pocket"] >= 5, "tooth_id"].unique()
    if len(severe_tooth_ids) >= 2 and len(pocket5_tooth_ids) >= 1:
        return "High"

    # Moderate: ≥2 teeth with CAL ≥ 4 or ≥2 teeth with pocket ≥ 5.
    moderate_tooth_ids_cal = df_filtered.loc[df_filtered["cal"] >= 4, "tooth_id"].unique()
    moderate_tooth_ids_pocket = df_filtered.loc[df_filtered["pocket"] >= 5, "tooth_id"].unique()
    if len(moderate_tooth_ids_cal) >= 2 or len(moderate_tooth_ids_pocket) >= 2:
        return "Moderate"

    # Low (mild): Either (≥2 teeth with CAL ≥ 3 and ≥2 teeth with pocket ≥ 4)
    #          OR even if any site has a pocket ≥ 5.
    mild_tooth_ids_cal = df_filtered.loc[df_filtered["cal"] >= 3, "tooth_id"].unique()
    mild_tooth_ids_pocket = df_filtered.loc[df_filtered["pocket"] >= 4, "tooth_id"].unique()
    if (len(mild_tooth_ids_cal) >= 2 and len(mild_tooth_ids_pocket) >= 2) or (df_filtered["pocket"] >= 5).any():
        return "Low"

    return "None"


def curate_final_exam_classification(df_long):
    """
    Loads the curated long dataset, computes clinical attachment loss (CAL),
    classifies each exam, and returns a DataFrame with one row per exam.

    The final DataFrame includes:
      - research_id, exam_id, exam_date, exam_type,
      - cal_classification (the calculated classification based on CAL),
      - periodontal_disease_risk (the risk value aggregated using the first non-missing value).
    """
    # Convert "pocket" and "recession" to numeric, replacing "Missing" with NaN.
    df_long["pocket"] = pd.to_numeric(df_long["pocket"].replace("Missing", np.nan), errors="coerce")
    df_long["recession"] = pd.to_numeric(df_long["recession"].replace("Missing", np.nan), errors="coerce")

    # Compute CAL as the sum of pocket and recession.
    df_long["cal"] = df_long["pocket"] + df_long["recession"]

    # Group by exam identifiers.
    exam_groups = df_long.groupby(["research_id", "exam_id", "exam_date"])

    num_exams = exam_groups.ngroups
    print("Number of exam groups:", num_exams)

    # Apply the classification function to each exam group.
    exam_classification = exam_groups.apply(classify_exam)

    # Aggregate the periodontal_disease_risk for each exam.
    # We assume the risk value is repeated for every site in an exam, so we take the first value.
    risk = exam_groups["periodontal_disease_risk"].first()

    # Reset indices and merge the results.
    final_exam_df = exam_classification.reset_index(name="cal_classification")
    risk_df = risk.reset_index(name="periodontal_disease_risk")

    final_exam_df = final_exam_df.merge(
        risk_df, on=["research_id", "exam_id", "exam_date"]
    )

    return final_exam_df


def plot_cal_vs_risk_matches(final_exam_df):
    """
    Plots a bar chart of how many rows in final_exam_df have matching classifications
    ('cal_classification' == 'periodontal_disease_risk') vs. those that do not.
    Each bar is annotated inside the bar with the count and percentage of total exams.
    """
    # 1) Create a "match_status" column indicating whether the classifications match.
    final_exam_df["match_status"] = final_exam_df.apply(
        lambda row: "Match" if row["cal_classification"] == row["periodontal_disease_risk"] else "Not Match",
        axis=1
    )

    # 2) Count the "Match" vs. "Not Match" values.
    match_counts = final_exam_df["match_status"].value_counts()
    total_exams = match_counts.sum()

    # 3) Create a bar plot.
    ax = match_counts.plot(kind="bar", color=["green", "red"], figsize=(6, 4))
    plt.title("CAL Classification vs. Periodontal Disease Risk")
    plt.xlabel("Match Status")
    plt.ylabel("Number of Exams")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # 4) Annotate each bar with the count and the percentage (inside the bar at the top).
    for bar in ax.patches:
        height = bar.get_height()
        percentage = height / total_exams * 100
        ax.annotate(
            f'{int(height)} ({percentage:.1f}%)',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, -15),  # Offset text 15 points downward, so it is inside the bar.
            textcoords="offset points",
            ha="center", va="top", color="white", fontsize=10
        )

    plt.show()


def plot_difference_index(final_exam_df):
    """
    For each exam in final_exam_df, computes a difference index defined as:
      difference = (level of cal_classification) - (level of periodontal_disease_risk)
    where the mapping is:
      - "None"      -> 1
      - "Low"       -> 2
      - "Moderate"  -> 3
      - "High"      -> 4

    Then, it plots the distribution of these differences as a bar chart.
    """
    # Define the mapping dictionary.
    level_mapping = {"None": 1, "Low": 2, "Moderate": 3, "High": 4}

    # Create a new column "difference" using the mapping.
    final_exam_df["difference"] = final_exam_df.apply(
        lambda row: level_mapping.get(row["cal_classification"], np.nan) -
                    level_mapping.get(row["periodontal_disease_risk"], np.nan),
        axis=1
    )

    # Count the frequency of each difference.
    diff_counts = final_exam_df["difference"].value_counts().sort_index()
    total_exams = diff_counts.sum()

    # Create a bar plot.
    ax = diff_counts.plot(kind="bar", color="blue", figsize=(6, 4))
    plt.title("Distribution of Difference Index\n(CAL Classification Level - PD Risk Level)")
    plt.xlabel("Difference Index")
    plt.ylabel("Number of Exams")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Annotate each bar with the count and percentage in black text.
    for bar in ax.patches:
        height = bar.get_height()
        percentage = height / total_exams * 100
        ax.annotate(
            f"{int(height)} ({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, -15),  # move text 15 points down inside the bar
            textcoords="offset points",
            ha="center",
            va="top",
            color="black",  # changed to black
            fontsize=10
        )
    plt.show()


def plot_transition_distribution(final_exam_df):
    """
    Creates a new column 'transition' in final_exam_df that records the mapping
    from the baseline risk ('periodontal_disease_risk') to the computed CAL classification
    ('cal_classification') in the format:
       "Periodontal Disease Risk to CAL Classification"

    Then, it plots a horizontal bar chart showing the frequency distribution of these transitions,
    with annotations (in normal orientation) showing the count and percentage just outside each bar.
    """
    # Create a new column "transition" that records the transition.
    final_exam_df["transition"] = final_exam_df.apply(
        lambda row: f"{row['periodontal_disease_risk']} to {row['cal_classification']}",
        axis=1
    )

    # Count the frequency of each transition.
    transition_counts = final_exam_df["transition"].value_counts().sort_index()
    total_exams = transition_counts.sum()

    # Create a horizontal bar plot.
    ax = transition_counts.plot(kind="barh", color="skyblue", figsize=(8, 5))
    plt.title("Distribution of Risk Transitions\n(Periodontal Disease Risk to CAL Classification)")
    plt.xlabel("Number of Exams")
    plt.ylabel("Transition")
    plt.tight_layout()

    # Annotate each bar with the count and percentage, just outside (to the right) of the bar.
    for bar in ax.patches:
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        percentage = width / total_exams * 100
        ax.annotate(
            f"{int(width)} ({percentage:.1f}%)",
            xy=(width, y_center),
            xytext=(3, 0),  # 3 points to the right
            textcoords="offset points",
            ha="left", va="center", color="black", fontsize=10
        )

    plt.show()


if __name__ == '__main__':
    df = load_curated_full_dataset_long_from_csv()
    unique_exam_ids = df["exam_id"].nunique()
    print("Unique exam_ids:", unique_exam_ids)
    filtered_df = df[["research_id", "periodontal_disease_risk", "exam_id", "exam_date", "exam_type", "quadrant",
                      "tooth_id", "site_id", "missing_status", "pocket", "recession"]]
    final_exam_classification_df = curate_final_exam_classification(filtered_df)
    plot_cal_vs_risk_matches(final_exam_classification_df)
    plot_difference_index(final_exam_classification_df)
    plot_transition_distribution(final_exam_classification_df)