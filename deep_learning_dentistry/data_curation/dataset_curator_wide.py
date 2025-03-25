import re
import pandas as pd
import numpy as np
from io import StringIO

from deep_learning_dentistry.data_curation.data_processing.bleeding_processor import (
    load_curated_bleeding_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.chart_general_processor import (
    load_curated_chart_general_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.chart_restorative_processor import (
    load_curated_chart_restore_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.demographics_data_processor import (
    load_curated_demographics_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.furcation_processor import (
    load_curated_furcation_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.index_processor import (
    load_curated_index_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.mag_processor import (
    load_curated_mag_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.mobility_processor import (
    load_curated_mobility_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.pockets_processor import (
    load_curated_pockets_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.recessions_processor import (
    load_curated_recessions_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.suppuration_processor import (
    load_curated_suppuration_data_from_csv,
)
from deep_learning_dentistry.data_curation.data_processing.utils.config import (
    FULL_DATASET_PATH,
    ANALYSIS_DATASET_PATH,
)
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import (
    save_curated_data_to_csv,
    load_curated_dataset,
)


def combine_seven_variables(merge_columns, exam_datasets, variable_suffixes):
    """Combined all seven variables into one dataframe."""
    import pandas as pd

    if len(exam_datasets) != len(variable_suffixes):
        raise ValueError("Number of exam_datasets must match number of variable_suffixes.")

    # Rename columns for each dataset, appending the specific suffix (except merge_columns)
    renamed_datasets = []
    for df, suffix in zip(exam_datasets, variable_suffixes):
        df_renamed = rename_columns_with_suffix(df.copy(), suffix, merge_cols=merge_columns)
        renamed_datasets.append(df_renamed)

    # Merge the renamed datasets using an outer join on the merge_columns
    combined_exam_data = renamed_datasets[0]
    for dataset in renamed_datasets[1:]:
        combined_exam_data = pd.merge(
            combined_exam_data,
            dataset,
            on=merge_columns,
            how='outer'
        )

    combined_exam_data = combined_exam_data.replace('DNE', pd.NA)

    # Sort the merged DataFrame by research_id and exam_date
    combined_exam_data.sort_values(by=['research_id', 'exam_date'], ascending=True, inplace=True)

    return combined_exam_data


def rename_columns_with_suffix(df, suffix, merge_cols):
    """ Appending the end of the column with qX_YY_ZZ with variable name. """
    pattern = re.compile(r"^q\d+_\d+_.+")
    new_cols = {}
    for col in df.columns:
        if col not in merge_cols and pattern.match(col):
            new_cols[col] = f"{col}_{suffix}"
    return df.rename(columns=new_cols)


def combine_variables_and_demographics(df_demographics, exam_df):
    """ Merge demographic data with rest of variables """
    combined_df = df_demographics.merge(exam_df, on='research_id', how='left')
    return combined_df


def determine_restoration_work_done(chart_restore_df, chart_general_df):
    """ """
    merged_columns = ["research_id", "exam_id", "exam_date", "exam_type"]

    # I. Find all exams with possible bridge creation
    relevant_cols = ["has_bridge_retainer_3/4_Crown", "has_bridge_retainer_veneer"]
    valid_ids = (
        chart_restore_df.groupby("research_id")[relevant_cols]
        .apply(lambda df: df.notna().any().any())
        .loc[lambda s: s].index
    )
    sub_df = chart_restore_df[chart_restore_df["research_id"].isin(valid_ids)].copy()

    # II. Find all potential teeth that could have a bridge
    sub_df["potential_teeth"] = sub_df.apply(find_neighbors, axis=1)

    # III. Determine the intersection between missing_teeth and potential_teeth, and label this intersection as restored_teeth
    merged_df = pd.merge(
        sub_df,
        chart_general_df,
        on=merged_columns,
        how="left"
    )

    merged_df["restored_teeth"] = merged_df.apply(find_restored_teeth, axis=1)

    # IV. Sort by research_id and exam_date, then propagate the restored_teeth forward
    merged_df["exam_date"] = pd.to_datetime(merged_df["exam_date"], errors="coerce")
    merged_df.sort_values(by=["research_id", "exam_date"], inplace=True)

    def accumulate_restored_teeth(group):
        """ Cumulative union of 'restored_teeth' sets for a single research_id. """
        accumulated = set()
        results = []
        for val in group["restored_teeth"]:
            current_set = set()
            if pd.notna(val):
                for part in val.split(","):
                    part = part.strip()
                    if part.isdigit():
                        current_set.add(part)
            accumulated |= current_set
            if accumulated:
                results.append(", ".join(sorted(accumulated, key=lambda x: int(x))))
            else:
                results.append(pd.NA)
        return pd.Series(results, index=group.index)

    merged_df["restored_teeth"] = (
        merged_df.groupby("research_id", group_keys=False).apply(accumulate_restored_teeth)
    )

    # V. Return only the columns of interest
    final_df = merged_df[merged_columns + ["restored_teeth"]].copy()

    return final_df


def find_neighbors(row):
    """ Helper function that returns the number above and below the number examined, i.e. 46 -> 45, 47 """
    relevant_cols = ["has_bridge_retainer_3/4_Crown", "has_bridge_retainer_veneer"]
    neighbors_set = set()

    for col in relevant_cols:
        val = row.get(col)
        if pd.notna(val):
            # val might be a single number or a comma-separated string
            for part in str(val).split(","):
                part = part.strip()
                if part:
                    try:
                        tooth_num = int(float(part))  # handle "46" or "46.0"
                        neighbors_set.add(tooth_num - 1)
                        neighbors_set.add(tooth_num + 1)
                    except ValueError:
                        pass

    if neighbors_set:
        return ", ".join(str(x) for x in sorted(neighbors_set))
    else:
        return np.nan


def find_restored_teeth(row):
    """ Find the teeth that have a fake bridge using the intersection "potential_teeth" and "missing_status" on that tooth. """
    if pd.isna(row.get("potential_teeth")):
        return np.nan

    # Parse the potential_teeth string into a list of tooth numbers (integers)
    potential_list = []
    for part in str(row["potential_teeth"]).split(","):
        part = part.strip()
        if part.isdigit():
            potential_list.append(int(part))

    # Build the missing-status column name, e.g. q2_21_missing_status
    restored_list = []
    for tooth_num in potential_list:
        quadrant = tooth_num // 10
        col_name = f"q{quadrant}_{tooth_num}_missing_status"

        # Check if that column exists and equals 1
        if col_name in row.index:
            val = row[col_name]
            try:
                if float(val) == 1:
                    restored_list.append(str(tooth_num))
            except ValueError:
                pass

    if restored_list:
        return ", ".join(restored_list)
    else:
        return np.nan


def combine_with_chart_restored(merge_with_chart_general, restored_teeth_df, df_chart_restore, merge_columns):
    """ Combine main df with chart_restored. """
    restored_teeth_df_new = simulate_csv_reload(restored_teeth_df)

    # I. Merge merged_df_with_demo and restored_teeth_df_new on merge_columns
    combined_df = pd.merge(merge_with_chart_general, restored_teeth_df_new, on=merge_columns, how='outer')

    base_cols = merge_with_chart_general.columns.tolist()
    new_cols = ["num_of_missing_teeth", "restored_teeth"]

    current_cols = combined_df.columns.tolist()
    current_cols = [col for col in current_cols if col not in new_cols]
    exam_type_index = base_cols.index("exam_type")
    exam_type_index = current_cols.index("exam_type")
    current_cols.insert(exam_type_index + 1, new_cols[0])
    current_cols.insert(exam_type_index + 2, new_cols[1])
    combined_df = combined_df[current_cols]

    # III. Merge with df_chart_restore on merge_columns (outer join)
    merged_df = pd.merge(combined_df, df_chart_restore, on=merge_columns, how='outer')

    return merged_df


def combine_with_chart_general(merged_df_with_demo, df_chart_general, merge_columns):
    """ Combine main df with chart_general."""
    combined_df = pd.merge(merged_df_with_demo, df_chart_general, on=merge_columns, how='outer')

    cols = combined_df.columns.tolist()
    missing_status_cols = [col for col in cols if col.endswith('_missing_status')]
    remaining_cols = [col for col in cols if col not in missing_status_cols]
    exam_date_index = remaining_cols.index('exam_type')
    new_order = remaining_cols[:exam_date_index+1] + missing_status_cols + remaining_cols[exam_date_index+1:]
    combined_df = combined_df[new_order]

    return combined_df


def simulate_csv_reload(df):
    """ Simulates saving the given DataFrame to CSV and reloading it without parsing dates. Need this to because
    when loading dataset from CSV, the exam_date column is a string, but some modified objects have exam_date as
    datetime object, which causes merging issues. """
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    df_reloaded = pd.read_csv(buffer, parse_dates=False)
    return df_reloaded


def propagation_forward_in_time(merged_df_with_demo):
    """ For each ressearch_id, propagate information forward in time related to missing_status, restored_teeth, ..."""

    # I. Propagation of Missing Teeth information FORWARD
    missing_cols = [col for col in merged_df_with_demo.columns if col.endswith('_missing_status')]
    merged_df_with_demo = merged_df_with_demo.sort_values(by=['research_id', 'exam_date'])

    for col in missing_cols:
        merged_df_with_demo[col] = merged_df_with_demo.groupby('research_id')[col].transform(
            lambda x: pd.to_numeric(x.ffill(), errors='coerce').cummax()
        )

    # II. Propagate forward for columns that start with "has_" but are not missing_status columns.
    propagation_cols = [col for col in merged_df_with_demo.columns if col.startswith('has_') or col == 'restored_teeth']
    for col in propagation_cols:
        if pd.api.types.is_numeric_dtype(merged_df_with_demo[col]):
            merged_df_with_demo[col] = merged_df_with_demo.groupby('research_id')[col].transform(
                lambda x: x.ffill().cummax()
            )
        else:
            merged_df_with_demo[col] = merged_df_with_demo.groupby('research_id')[col].transform(
                cumulative_string_union
            )

    return merged_df_with_demo


def propagation_backward_in_time(merged_df_with_demo):
    """ For each ressearch_id, propagate information backward in time based on the first examination that is not pd.NA
    """
    # I. Propagation of Missing Teeth information BACKWARD
    missing_cols = [col for col in merged_df_with_demo.columns if col.endswith('_missing_status')]
    merged_df_with_demo = merged_df_with_demo.sort_values(by=['research_id', 'exam_date'])

    for col in missing_cols:
        merged_df_with_demo[col] = pd.to_numeric(merged_df_with_demo[col], errors='coerce')
        merged_df_with_demo[col] = merged_df_with_demo.groupby('research_id')[col].transform(lambda x: x.bfill())

    return merged_df_with_demo


def cumulative_string_union(series):
    """ Helper function for propagation_forward_in_time. """
    cumulative = set()
    results = []
    for val in series:
        if pd.notna(val):
            items = {item.strip() for item in str(val).split(',') if item.strip()}
            cumulative |= items
        if cumulative:
            try:
                sorted_items = sorted(cumulative, key=lambda x: int(x))
            except ValueError:
                sorted_items = sorted(cumulative)
            results.append(", ".join(sorted_items))
        else:
            results.append(np.nan)
    return pd.Series(results, index=series.index)


def check_num_of_missing_teeth(main_df):
    """ Function to check if num_of_missing_teeth column has value for each row. """

    def count_missing_teeth(row):
        missing_cols = [col for col in row.index if col.endswith("_missing_status")]
        if not missing_cols:
            return pd.NA
        if all(pd.isna(row[col]) for col in missing_cols):
            return pd.NA
        count = 0
        for col in missing_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    if float(val) == 1:
                        count += 1
                except Exception:
                    continue
        return count

    main_df['num_of_missing_teeth'] = main_df.apply(count_missing_teeth, axis=1)
    return main_df

def index_curator(main_df, index_df, merge_columns):
    """ Calculates:
       - difference = general_missing_teeth_count - index)missing_teeth
       - number_of_surfaces = (32 - general_missing_teeth_count) * 6
       - number_of_bleeding_surfaces = the precomputed sum_of_teeth_with_bleeding from bleeding_wide
       - bleeding_index = number_of_bleeding_surfaces / number_of_surfaces
    Returns the merged DataFrame with these new columns. """
    df = main_df[merge_columns].copy()

    ## Number of Teeth and Sites ##
    df['num_of_total_teeth'] = 32 - main_df['num_of_missing_teeth']
    df['num_of_total_sites'] = df['num_of_total_teeth'] * 6

    ## Bleeding Index & Percent of Teeth Bleeding ##
    df['number_of_bleeding_teeth'] = main_df['num_of_bleeding_teeth']
    df['number_of_bleeding_sites'] = main_df['num_of_bleeding_sites']
    df['bleeding_index'] = df.apply(
        lambda row: row['number_of_bleeding_sites'] / row['num_of_total_sites']
        if pd.notna(row['num_of_total_sites']) and row['num_of_total_sites'] != 0 else pd.NA,
        axis=1
    )
    df['pcnt_of_teeth_with_bleeding'] = df.apply(
        lambda row: (row['number_of_bleeding_teeth'] / row['num_of_total_teeth']) * 100
        if pd.notna(row['num_of_total_teeth']) and row['num_of_total_teeth'] != 0 else pd.NA,
        axis=1
    )

    ## Suppuration Index & Percent of Teeth with Suppuration ##
    df['number_of_suppuration_teeth'] = main_df['num_of_suppuration_teeth']
    df['number_of_suppuration_sites'] = main_df['num_of_suppuration_sites']
    df['suppuration_index'] = df.apply(
        lambda row: row['number_of_suppuration_sites'] / row['num_of_total_sites']
        if pd.notna(row['num_of_total_sites']) and row['num_of_total_sites'] != 0 else pd.NA,
        axis=1
    )
    df['pcnt_of_teeth_with_suppuration'] = df.apply(
        lambda row: (row['number_of_suppuration_teeth'] / row['num_of_total_teeth']) * 100
        if pd.notna(row['num_of_total_teeth']) and row['num_of_total_teeth'] != 0 else pd.NA,
        axis=1
    )

    ## Plaque Metrics ##
    merge_plaque_cols = merge_columns + ['plaque_index', 'percent_of_plaque_surfaces']
    merged = pd.merge(main_df, index_df[merge_plaque_cols], on=merge_columns, how='left')
    df['plaque_index'] = merged['plaque_index']
    df['pcnt_of_teeth_with_plaque'] = merged['percent_of_plaque_surfaces']

    ## Final Merge ##
    merged_df = pd.merge(main_df, df, on=merge_columns, how='outer')
    cols = merged_df.columns.tolist()

    for col in ["num_of_total_teeth", "num_of_total_sites"]:
        if col in cols:
            cols.remove(col)

    idx = cols.index("exam_type")

    cols.insert(idx + 1, "num_of_total_teeth")
    cols.insert(idx + 3, "num_of_total_sites")
    merged_df = merged_df[cols]

    return merged_df


def final_curator(main_df, merge_columns):
    """ Moves all columsn with 'num_of_..." to right beside num_of_sites. """
    cols = main_df.columns.tolist()

    # 1) Identify the first group: columns that start with 'num_of_',
    #    do not end with '_sites', and are not 'num_of_total_sites'.
    num_of_cols = [
        col for col in cols
        if col.startswith("num_of_")
           and not col.endswith("_sites")
    ]

    # 2) Identify the second group: columns that start with 'num_of_',
    #    do not end with '_teeth', and are not 'num_of_total_sites'.
    num_of_cols_2 = [
        col for col in cols
        if col.startswith("num_of_")
           and not col.endswith("_teeth")
    ]

    # Remove these columns from the current order so we can re-insert them
    for c in num_of_cols:
        if c in cols:
            cols.remove(c)
    for c in num_of_cols_2:
        if c in cols:
            cols.remove(c)

    # 3) Insert the first group right after 'restored_teeth'
    try:
        idx = cols.index("restored_teeth")
    except ValueError:
        # If 'restored_teeth' not found, we can default to appending at the end
        idx = len(cols) - 1
    for i, c in enumerate(num_of_cols):
        cols.insert(idx + 1 + i, c)

    # 4) Insert the second group right after 'num_of_suppuration_teeth'
    try:
        idy = cols.index("num_of_suppuration_teeth")
    except ValueError:
        idy = len(cols) - 1
    for i, c in enumerate(num_of_cols_2):
        cols.insert(idy + 1 + i, c)

    # Re-index main_df with the new column order
    main_df = main_df[cols]

    # Convert columns that start with 'num_of_' to numeric
    for col in main_df.columns:
        if col.startswith("num_of_"):
            main_df[col] = pd.to_numeric(main_df[col], errors='coerce')

    return main_df


def curate_full_dataset_wide():
    """
    Function to curate data.
    """
    df_bleeding = load_curated_bleeding_data_from_csv()
    df_chart_general = load_curated_chart_general_data_from_csv()
    df_chart_restore = load_curated_chart_restore_data_from_csv()
    df_demographics = load_curated_demographics_data_from_csv()
    df_furcation = load_curated_furcation_data_from_csv()
    df_index = load_curated_index_data_from_csv()
    df_mag = load_curated_mag_data_from_csv()
    df_mobility = load_curated_mobility_data_from_csv()
    df_pockets = load_curated_pockets_data_from_csv()
    df_recessions = load_curated_recessions_data_from_csv()
    df_suppuration = load_curated_suppuration_data_from_csv()

    # variables_to_combine = [df_bleeding, df_furcation, df_pockets, df_recessions, df_suppuration]
    # variable_names = ['bleeding', 'furcation', 'pocket', 'recession', 'suppuration']

    variables_to_combine = [df_bleeding, df_furcation, df_mag, df_mobility, df_pockets, df_recessions, df_suppuration]
    variable_names = ['bleeding', 'furcation', 'mag', 'mobility', 'pocket', 'recession', 'suppuration']

    merge_columns = ['research_id', 'exam_id', 'exam_date', 'exam_type']

    merged_df = combine_seven_variables(merge_columns, variables_to_combine, variable_names)
    merge_with_chart_general = combine_with_chart_general(merged_df, df_chart_general, merge_columns)

    restored_teeth_df = determine_restoration_work_done(df_chart_restore, df_chart_general)
    merge_with_chart_restored = combine_with_chart_restored(merge_with_chart_general, restored_teeth_df, df_chart_restore, merge_columns)

    merged_df_with_demo = combine_variables_and_demographics(df_demographics, merge_with_chart_restored)
    propped_forward_df = propagation_forward_in_time(merged_df_with_demo)
    propped_backward_df = propagation_backward_in_time(propped_forward_df)
    draft_df = check_num_of_missing_teeth(propped_backward_df)
    indexed_df = index_curator(draft_df, df_index, merge_columns)
    final_df = final_curator(indexed_df, merge_columns)

    return final_df

def save_curated_dataset_wide_to_csv(full_dataset_wide, name):
    """ Saves curated full dataset wide as a CSV file. """
    save_curated_data_to_csv(FULL_DATASET_PATH, full_dataset_wide, name)

def load_curated_full_dataset_wide_from_csv():
    return load_curated_dataset(FULL_DATASET_PATH, "full_dataset_wide.csv")

def load_curate_save_full_dataset_wide():
    """ Curates raw bleeding data and saves it. """
    full_dataset = curate_full_dataset_wide()
    save_curated_dataset_wide_to_csv(full_dataset, "full_dataset_wide.csv")


if __name__ == "__main__":
    load_curate_save_full_dataset_wide()