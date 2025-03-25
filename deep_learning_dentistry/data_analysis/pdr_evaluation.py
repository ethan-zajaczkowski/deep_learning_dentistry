import pandas as pd
import matplotlib.pyplot as plt
from deep_learning_dentistry.data_curation.data_processing.utils.config import DEMOGRAPHICS_CURATED_PATH_CSV, \
    FULL_DATASET_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import load_curated_dataset


def load_dataset(cleaned_dataset_path):
    """
    Import the specified cleaned dataset.
    """
    df = pd.read_csv(cleaned_dataset_path)
    return df


def count_chart_ids(df):
    """
    Reads an Excel file into a pandas DataFrame and counts the number of unique research_id entries.
    Returns the count of unique research_id values.
    """
    unique_count = df['research_id'].nunique()
    return unique_count


def count_unique_exam_ids(df):
    """
    Reads an Excel file into a pandas DataFrame and counts the number of unique research_id entries.
    Returns the count of unique research_id values.
    """
    unique_count = df['exam_id'].nunique()
    return unique_count


if __name__ == '__main__':
    demographics_data = load_dataset(DEMOGRAPHICS_CURATED_PATH_CSV)

    curated_data = load_curated_dataset(FULL_DATASET_PATH, "full_dataset_long.csv")
    df_missing = (
        curated_data
        .groupby(["research_id", "exam_id"], as_index=False)
        .apply(lambda g: g.loc[g["missing_status"] == 1, "tooth_id"].nunique())
        .rename(columns={0: "missing_teeth"})  # Rename the unnamed column
    )
    df_missing.rename(columns={None: "missing_teeth"}, inplace=True)
    df_last_3 = df_missing.groupby("research_id").tail(3)
    df_max_teeth = (
        df_last_3
        .groupby("research_id", as_index=False)["missing_teeth"]
        .max()
    )

    df_max_teeth.rename(columns={"research_id": "research_id"}, inplace=True)
    demographics_data = demographics_data.merge(df_max_teeth, on="research_id", how="left")
    demographics_data["date_of_birth"] = pd.to_datetime(demographics_data["date_of_birth"])
    exam_date = pd.to_datetime("2023-04-01")
    demographics_data["age_at_exam"] = (exam_date - demographics_data["date_of_birth"]).dt.days / 365.25

    data_high = demographics_data[demographics_data["periodontal_disease_risk"] ==  "High"]
    data_med = demographics_data[demographics_data["periodontal_disease_risk"] == "Moderate"]
    data_low = demographics_data[demographics_data["periodontal_disease_risk"] == "Low"]
    data_blank = demographics_data[demographics_data["periodontal_disease_risk"].isna()]

    # Create a figure for the subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Define the datasets and titles
    datasets = [data_high, data_med, data_low, data_blank]
    titles = ["High Risk", "Medium Risk", "Low Risk", "Blank Risk"]
    colors = ['blue', 'orange', 'green', 'red']

    ### GENDER VS PERIODONTAL DISEASE RISK ###

    # We'll use blue for "M" and magenta for "F"
    gender_colors = ["blue", "magenta"]

    total_men = demographics_data["gender"].eq("M").sum()
    total_women = demographics_data["gender"].eq("F").sum()

    for ax, data, title in zip(axes.flatten(), datasets, titles):
        if not data.empty:
            # Count outcomes within this risk group
            count_M = data["gender"].eq("M").sum()
            count_F = data["gender"].eq("F").sum()

            # Normalize using overall totals (if totals > 0)
            perc_M = (count_M / total_men * 100) if total_men > 0 else 0
            perc_F = (count_F / total_women * 100) if total_women > 0 else 0

            outcomes = ["M", "F"]
            percents = [perc_M, perc_F]

            # Plot the bar chart for percentages
            bars = ax.bar(outcomes, percents, color=gender_colors, alpha=0.7, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("gender")
            ax.set_ylabel("Percentage of Overall (%)")

            # Set y-axis from 0 to 100%
            ax.set_ylim(0, 100)

            # Annotate each bar with the normalized percentage
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height - (0.05 * height),
                    f"{pct:.1f}%",
                    ha='center',
                    va='top',
                    color='white',
                    fontweight='bold'
                )
        else:
            ax.set_title(f"{title} (No Data)")
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    ### AGE VS PERIODONTAL DISEASE RISK ###

    # Generate the bar plots
    # for ax, data, title, color in zip(axes.flatten(), datasets, titles, colors):
    #     if not data.empty:
    #         # Calculate mean and standard deviation
    #         mean_val = data["age_at_exam"].mean()
    #         sd_val = data["age_at_exam"].std()
    #
    #         # Plot the histogram
    #         ax.hist(data["age_at_exam"], bins=20, color=color, alpha=0.7, edgecolor="black")
    #
    #         # Add a vertical dashed line at the mean
    #         ax.axvline(mean_val, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
    #
    #         # Add legend and title with mean and sd information
    #         ax.legend()
    #         ax.set_title(f"{title}\nMean: {mean_val:.1f}, SD: {sd_val:.1f}")
    #         ax.set_xlabel("Age at Exam")
    #         ax.set_ylabel("Count")
    #     else:
    #         ax.set_title(f"{title} (No Data)")
    #         ax.axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    # for ax, data, title in zip(axes.flatten(), datasets, titles):
    #     if not data.empty:
    #         # Create a box plot for the "age_at_exam" column.
    #         # Use patch_artist=True to allow color filling and showmeans=True to display the mean.
    #         bp = ax.boxplot(data["age_at_exam"], patch_artist=True, showmeans=True)
    #
    #         # Customize the box plot colors (optional)
    #         for box in bp['boxes']:
    #             box.set(facecolor='lightblue', color='blue')
    #         for whisker in bp['whiskers']:
    #             whisker.set(color='blue')
    #         for cap in bp['caps']:
    #             cap.set(color='blue')
    #         for median in bp['medians']:
    #             median.set(color='red', linewidth=2)
    #         for mean in bp['means']:
    #             mean.set(marker='D', markerfacecolor='yellow', markeredgecolor='black')
    #
    #         # Calculate statistics for annotation
    #         mean_val = data["age_at_exam"].mean()
    #         sd_val = data["age_at_exam"].std()
    #
    #         # Set title with the computed mean and standard deviation
    #         ax.set_title(f"{title}\nMean: {mean_val:.1f}, SD: {sd_val:.1f}")
    #         ax.set_ylabel("Age at Exam")
    #
    #         # Remove x-axis tick labels (we have one box only)
    #         ax.set_xticklabels([""])
    #     else:
    #         ax.set_title(f"{title} (No Data)")
    #         ax.axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    ### TOBACCO USE VS. PERIODONTAL DISEASE RISK ###

    # for ax, data, title in zip(axes.flatten(), datasets, titles):
    #     if not data.empty:
    #         # Count outcomes: "Yes", "No", and missing values (pd.NA)
    #         count_yes = data["TobaccoUser"].eq("Yes").sum()
    #         count_no = data["TobaccoUser"].eq("No").sum()
    #         count_blank = data["TobaccoUser"].isna().sum()
    #
    #         # Prepare data for bar plot
    #         outcomes = ["Yes", "No", "Blank"]
    #         counts = [count_yes, count_no, count_blank]
    #         colors = ["green", "red", "gray"]
    #
    #         # Plot the bar chart
    #         bars = ax.bar(outcomes, counts, color=colors, alpha=0.7, edgecolor="black")
    #         ax.set_title(title)
    #         ax.set_xlabel("Tobacco User")
    #         ax.set_ylabel("Count")
    #
    #         # Make the y-axis go up to 1700
    #         ax.set_ylim(0, 1700)
    #
    #         # Place labels INSIDE each bar (near the top)
    #         for bar, val in zip(bars, counts):
    #             # Position the label slightly below the bar’s top edge
    #             # so it's inside the bar area
    #             height = bar.get_height()
    #             ax.text(
    #                 bar.get_x() + bar.get_width() / 2.0,
    #                 height - (0.05 * height),  # 5% below the top of the bar
    #                 str(val),
    #                 ha='center',
    #                 va='top',
    #                 color='white',  # White text is visible on darker bars
    #                 fontweight='bold'
    #             )
    #     else:
    #         ax.set_title(f"{title} (No Data)")
    #         ax.axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    ### MISSING TEETH VS PERIODONTAL DISEASE RISK ###

    # for ax, data, title, color in zip(axes.flatten(), datasets, titles, colors):
    #     if not data.empty:
    #         # Calculate statistics on missing_teeth.
    #         mean_val = data["missing_teeth"].mean()
    #         sd_val = data["missing_teeth"].std()
    #
    #         # Define bins based on the range of missing_teeth values.
    #         max_val = data["missing_teeth"].max()
    #         bins = np.arange(0, max_val + 2)  # bins for each integer count
    #
    #         # Plot histogram of missing_teeth.
    #         counts, bins, patches = ax.hist(data["missing_teeth"], bins=bins, color=color,
    #                                         alpha=0.7, edgecolor="black")
    #
    #         # Add a vertical dashed line at the mean.
    #         ax.axvline(mean_val, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}')
    #
    #         # Set title with mean and SD.
    #         ax.set_title(f"{title}\nMean: {mean_val:.1f}, SD: {sd_val:.1f}")
    #         ax.set_xlabel("Missing Teeth")
    #         ax.set_ylabel("Count")
    #
    #         # Set a fixed y-axis maximum (if desired). Here we ensure a minimum upper limit of 50.
    #         ax.set_ylim(0, max(50, counts.max() * 1.2))
    #
    #         # Optionally, annotate the bars by placing count labels inside each bar.
    #         for patch, count in zip(patches, counts):
    #             if count > 0:
    #                 # Place the label at the center of the bar, slightly below the top.
    #                 ax.text(patch.get_x() + patch.get_width() / 2,
    #                         patch.get_height() - (0.05 * patch.get_height()),
    #                         int(count), ha='center', va='top', color='white', fontweight='bold')
    #
    #         ax.legend()
    #     else:
    #         ax.set_title(f"{title} (No Data)")
    #         ax.axis("off")

    plt.tight_layout()
    plt.show()



