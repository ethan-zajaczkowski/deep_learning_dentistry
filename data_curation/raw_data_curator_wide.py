import re
import webbrowser
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from deep_learning_dentistry.data_curation.data_processing.bleeding_processor import load_processed_bleeding_data
from deep_learning_dentistry.data_curation.data_processing.chart_general_processor import \
    load_processed_chart_general_data
from deep_learning_dentistry.data_curation.data_processing.chart_restorative_processor import \
    load_processed_chart_restorative_data
from deep_learning_dentistry.data_curation.data_processing.demographics_data_processor import \
    load_processed_demographic_data
from deep_learning_dentistry.data_curation.data_processing.furcation_processor import load_processed_furcation_data
from deep_learning_dentistry.data_curation.data_processing.index_processor import load_processed_index_data, \
    load_cleaned_index_data_modified
from deep_learning_dentistry.data_curation.data_processing.mag_processor import load_processed_mag_data
from deep_learning_dentistry.data_curation.data_processing.mobility_processor import load_processed_mobility_data
from deep_learning_dentistry.data_curation.data_processing.pockets_processor import load_processed_pockets_data
from deep_learning_dentistry.data_curation.data_processing.recessions_processor import load_processed_recessions_data
from deep_learning_dentistry.data_curation.data_processing.suppuration_processor import load_processed_suppuration_data
from deep_learning_dentistry.data_curation.data_processing.utils.config import CURATED_PROCESSED_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_demographic_data, \
    load_furcation
from deep_learning_dentistry.data_curation.data_processing.utils.functions import load_cleaned_dataset, \
    generate_teeth_order, create_tooth_site_columns, fill_exam_suppuration, save_processed_data_to_csv
from deep_learning_dentistry.deep_learning_analysis.variable_analysis.variable_analysis import load_suppuration


## Preparing Datasets For Merging ##

def create_chart_general_df(chart_general):
    """
    Takes chart_general and creates a wide-format DataFrame for missing_status,
    with one row per exam and one column per tooth (based on tooth_id).
    """
    exam_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']

    df_wide = chart_general.pivot_table(
        index=exam_cols,
        columns='tooth_id',
        values='missing_status',
        aggfunc='first'
    ).reset_index()
    df_wide.columns.name = None

    tooth_order_numeric = generate_teeth_order()["teeth_order"]
    mapping = {t: f"q{t // 10}_{t}, missing_status" for t in tooth_order_numeric}
    df_wide.rename(columns=mapping, inplace=True)

    col_order = exam_cols + [mapping[t] for t in tooth_order_numeric]
    df_wide = df_wide.reindex(columns=col_order, fill_value=0)

    missing_cols = [mapping[t] for t in tooth_order_numeric]
    df_wide["general_missing_teeth_count"] = df_wide[missing_cols].sum(axis=1)

    return df_wide


def create_restoration_exam_level_data(df):
    """
    Extract exam level data for general_chart dataframe.
    """
    exam_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']

    df_exam_level = (
        df
        .groupby(exam_cols, as_index=False)
        .first()
    )
    df_exam_level.drop(columns=['tooth_quadrant', 'tooth_id', 'tooth_site'], inplace=True)

    return df_exam_level


def create_tooth_level_variable_data(df, variable=None):
    """
    Take a variable from a processed DataFrame and turn it into a wide format with only tooth-level data.
    """

    def numeric_to_fdi(tooth_id, variable):
        """
        Converts a numeric tooth id (e.g., 18) into a string in the format "Q(quadrant), T(tooth_id), variable".
        """
        quadrant = tooth_id // 10
        return f"q{quadrant}_{tooth_id}, {variable}"

    tooth_order_numeric = generate_teeth_order()["teeth_order"]
    tooth_order = [numeric_to_fdi(t, variable) for t in tooth_order_numeric]

    df_wide = df.pivot_table(
        index=['research_id', 'exam_id', 'exam_date', 'exam_type'],
        columns='tooth_id',
        values=variable,
        aggfunc='first'
    )
    df_wide = df_wide.reset_index()
    df_wide.columns.name = None

    mapping = {t: numeric_to_fdi(t, variable) for t in tooth_order_numeric}
    df_wide.rename(columns=mapping, inplace=True)

    col_order = ['research_id', 'exam_id', 'exam_date', 'exam_type'] + tooth_order
    df_wide = df_wide.reindex(columns=col_order, fill_value=np.nan)

    df_wide[f"sum_of_teeth_with_{variable}"] = df_wide[tooth_order].sum(axis=1, skipna=True)

    return df_wide


def create_tooth_site_level_variable_data(df, variable=None):
    """
    Takes a variable from a processed DataFrame and turns it into a wide format
    that includes tooth-site level data. Also computes:
      - sum_of_surfaces_with_<variable>: total number of '1's across all tooth-site columns
      - sum_of_teeth_with_<variable>: number of distinct teeth that have at least one site = 1
    """
    def tooth_indicator(row):
        count = 0
        for tooth_base, col_list in tooth_groups.items():
            # If ANY site in col_list is 1, we count that tooth once
            if any(row[c] == 1 for c in col_list if pd.notna(row[c])):
                count += 1
        return count

    def tooth_indicator_furc(row):
        count = 0
        for tooth_base, col_list in tooth_groups.items():
            if any(row[c] == 1 for c in col_list if pd.notna(row[c])):
                count += 1
        return count

    df = df.copy()
    df['tooth_id_site'] = df['tooth_id'].astype(str) + '-' + df['tooth_site'].astype(str)

    df_wide = df.pivot_table(
        index=['research_id', 'exam_id', 'exam_date', 'exam_type'],
        columns='tooth_id_site',
        values=variable,
        aggfunc='first'
    ).reset_index()

    df_wide.columns.name = None

    ordered_tooth_site_columns = create_tooth_site_columns()
    mapping = {}
    for col in ordered_tooth_site_columns:
        try:
            tooth_id, tooth_site = col.split('-')
            quadrant = int(tooth_id) // 10
            mapping[col] = f"q{quadrant}_{tooth_id}_{tooth_site}, {variable}"
        except ValueError:
            continue

    df_wide.rename(columns=mapping, inplace=True)

    formatted_columns = [mapping[col] for col in ordered_tooth_site_columns if col in mapping]
    final_order = ['research_id', 'exam_id', 'exam_date', 'exam_type'] + formatted_columns
    df_wide = df_wide.reindex(columns=final_order, fill_value=np.nan)

    tooth_groups = {}
    for col in formatted_columns:
        parts = col.split(", ")
        if len(parts) < 2:
            continue
        base_site = parts[0]
        if "_" in base_site:
            tooth_base, _ = base_site.rsplit("_", 1)
            tooth_groups.setdefault(tooth_base, []).append(col)

    if variable == "mobility":
        df_wide[formatted_columns] = df_wide[formatted_columns].apply(pd.to_numeric, errors='coerce')

    if variable == "furcation":
        df_furc_numeric = df_wide[formatted_columns].replace("Not Available", pd.NA)
        df_wide[f"sum_of_teeth_with_{variable}"] = df_furc_numeric.apply(tooth_indicator_furc, axis=1)

    if variable not in ['pocket', 'recession', 'furcation']:
        df_wide[f"sum_of_teeth_with_{variable}"] = df_wide.apply(tooth_indicator, axis=1)

    if variable not in ['pocket', 'recession', 'mag', 'mobility', 'furcation']:
        df_wide[f"sum_of_surfaces_with_{variable}"] = df_wide[formatted_columns].sum(axis=1, skipna=True)

    return df_wide


def create_tooth_level_variable_data_suppuration_df(suppuration_df, index_df):
    """
    Takes suppuration and creates a modified dataframe of it, that imputes some
    suppuration values with 0 based on values from index_df, then calculates:
      - sum_of_surfaces_with_suppuration: total 1's across all tooth-site columns
      - sum_of_teeth_with_suppuration: count of teeth that have at least one site = 1
    """
    def tooth_indicator(row):
        """Return how many teeth in this row have at least one site with suppuration = 1."""
        count = 0
        for t_id, cols in tooth_groups.items():
            if any((row[col] == 1) for col in cols if pd.notna(row[col])):
                count += 1
        return count

    suppuration_df_wide = create_tooth_site_level_variable_data(suppuration_df, variable="suppuration")

    merge_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']
    merged_df = pd.merge(suppuration_df_wide, index_df, on=merge_cols, how='outer')

    supp_cols = [col for col in merged_df.columns if col.endswith(', suppuration')]

    mask = (
        (merged_df['suppuration_index'] == 0.0) &
        (merged_df[supp_cols].isna().all(axis=1))
    )
    merged_df.loc[mask, supp_cols] = 0

    merged_df['sum_of_surfaces_with_suppuration'] = merged_df[supp_cols].sum(axis=1, skipna=True)

    tooth_groups = {}
    for col in supp_cols:
        match = re.match(r'(q\d+_\d+)', col)
        if match:
            tooth_id = match.group(1)  # e.g., "q4_48"
            tooth_groups.setdefault(tooth_id, []).append(col)

    merged_df['sum_of_teeth_with_suppuration'] = merged_df.apply(tooth_indicator, axis=1)

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

    return merged_df


def create_tooth_level_restorative_data(df_restore):
    """
    Pivots multiple restorative columns by tooth_id so each exam is one row,
    and each tooth’s restorative attributes occupy separate columns.
    Columns are grouped first by the restorative attribute, then by tooth.
    """
    exam_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']

    restorative_cols = [
        'tooth_notes_restorative',
        'has_bridge_retainer_3/4_Crown',
        'has_bridge_retainer_veneer',
        'has_veneer',
        'has_fillings_or_caries',
        'has_inlay',
        'restored_material'
    ]

    # Pivot the DataFrame
    df_wide = df_restore.pivot_table(
        index=exam_cols,
        columns='tooth_id',
        values=restorative_cols,
        aggfunc='first'
    ).reset_index()

    # Flatten the resulting multi-level columns
    df_wide.columns = [
        '_'.join(str(level) for level in col if level)
        for col in df_wide.columns
    ]

    tooth_order_numeric = generate_teeth_order()["teeth_order"]

    rename_dict = {}
    for tooth_id in tooth_order_numeric:
        quadrant = tooth_id // 10
        for rcol in restorative_cols:
            old_name = f"{rcol}_{tooth_id}"  # e.g., "has_inlay_18"
            new_name = f"q{quadrant}_{tooth_id}, {rcol}"  # e.g., "q1_18, has_inlay"
            rename_dict[old_name] = new_name

    # Rename the flattened columns using the rename dictionary
    df_wide.rename(columns=rename_dict, inplace=True)

    meta_cols = exam_cols
    desired_cols = []

    # Loop over each restorative col FIRST, then each tooth
    for rcol in restorative_cols:
        for tooth_id in tooth_order_numeric:
            quadrant = tooth_id // 10
            col_name = f"q{quadrant}_{tooth_id}, {rcol}"
            desired_cols.append(col_name)

    # Reindex to enforce the final column order
    final_order = meta_cols + [col for col in desired_cols if col in df_wide.columns]
    df_wide = df_wide.reindex(columns=final_order, fill_value=pd.NA)

    return df_wide


def parse_tooth_string(t_str):
    qpart, toothnum = t_str.split('_')
    quadrant = int(qpart.replace('q', ''))
    return quadrant, int(toothnum)


def determine_restoration_work_done(chart_restore_wide):
    """
    Creates a new DataFrame from chart_restore_wide.
    For each row:
      1) Identify if a tooth has "has_bridge_retainer_3/4_Crown" or "has_bridge_retainer_veneer".
      2) If so, mark the new 'potential_restorative_work' column for that tooth AND its adjacent teeth (±1).
    The final DataFrame is returned with columns ordered by your generate_teeth_order() for the newly added columns.
    """
    new_df = chart_restore_wide[['research_id', 'exam_id', 'exam_date', 'exam_type']].copy()

    tooth_order_numeric = generate_teeth_order()["teeth_order"]
    tooth_info_list = [f"q{tnum // 10}_{tnum}" for tnum in tooth_order_numeric]

    for t_str in tooth_info_list:
        col_name = f"{t_str}, potential_restorative_work"
        new_df[col_name] = 0

    relevant_cols = [
        c for c in chart_restore_wide.columns
        if ("has_bridge_retainer_3/4_Crown" in c) or ("has_bridge_retainer_veneer" in c)
    ]

    pattern = r"^(q\d+_\d+),"

    for idx in chart_restore_wide.index:
        for col in relevant_cols:
            val = chart_restore_wide.at[idx, col]
            if pd.notna(val) and val == 1:
                match = re.match(pattern, col.strip())
                if match:
                    tooth_str = match.group(1)  # e.g. "q1_17"
                    quadrant, tooth_id = parse_tooth_string(tooth_str)

                    # Mark the current tooth
                    current_col = f"{tooth_str}, potential_restorative_work"
                    if current_col in new_df.columns:
                        new_df.at[idx, current_col] = 1

                    # Mark neighbors (±1) in the same quadrant
                    for neighbor in [tooth_id - 1, tooth_id + 1]:
                        neighbor_str = f"q{quadrant}_{neighbor}"
                        neighbor_col = f"{neighbor_str}, potential_restorative_work"
                        if neighbor_col in new_df.columns:
                            new_df.at[idx, neighbor_col] = 1

    pot_work_cols = [col for col in new_df.columns if col.endswith(", potential_restorative_work")]
    mask = (new_df[pot_work_cols] == 1).any(axis=1)
    filtered_df = new_df[mask].copy()

    return filtered_df


def missing_tooth_restored(df_general, df_restorative):
    """
    Determine if there was a missing tooth here and it has been restored with a fake bridge of some kind.
    """
    merge_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']

    df_merged = pd.merge(
        df_general,
        df_restorative,
        on=merge_cols,
        how='inner'
    )

    tooth_order_numeric = generate_teeth_order()["teeth_order"]
    for tnum in tooth_order_numeric:
        quadrant = tnum // 10
        missing_col = f"q{quadrant}_{tnum}, missing_status"
        potwork_col = f"q{quadrant}_{tnum}, potential_restorative_work"

        new_col = f"q{quadrant}_{tnum}, missing_tooth_restored"

        if missing_col not in df_merged.columns or potwork_col not in df_merged.columns:
            df_merged[new_col] = 0
        else:
            df_merged[new_col] = (
                    (df_merged[missing_col] == 1) & (df_merged[potwork_col] == 1)
            ).astype(int)

    missing_restored_cols = [col for col in df_merged.columns if col.endswith(", missing_tooth_restored")]
    final_cols = merge_cols + missing_restored_cols

    return df_merged[final_cols]


def index_curator(chart_general_wide, bleeding_wide, suppuration_wide, index_wide):
    """
    Calculates:
       - difference = general_missing_teeth_count - index)missing_teeth
       - number_of_surfaces = (32 - general_missing_teeth_count) * 6
       - number_of_bleeding_surfaces = the precomputed sum_of_teeth_with_bleeding from bleeding_wide
       - bleeding_index = number_of_bleeding_surfaces / number_of_surfaces
    Returns the merged DataFrame with these new columns.
    """
    ## Merging ##
    merge_cols = ['research_id', 'exam_id', 'exam_date', 'exam_type']
    merged_df = pd.merge(chart_general_wide, index_wide, on=merge_cols, how='outer')
    merged_df = pd.merge(merged_df, bleeding_wide, on=merge_cols, how='outer')
    merged_df = pd.merge(merged_df, suppuration_wide, on=merge_cols, how='outer')

    rename_map = {
        'missing_teeth': 'index_missing_teeth',
    }

    merged_df.rename(columns=rename_map, inplace=True)

    ## Compute Difference ##
    merged_df['general_index_missing_teeth_difference'] = merged_df.apply(
        lambda row: row['general_missing_teeth_count'] - row['index_missing_teeth']
        if pd.notna(row['general_missing_teeth_count']) and pd.notna(row['index_missing_teeth'])
        else pd.NA,
        axis=1
    )

    ## Number of Missing Teeth ##
    merged_df['number_of_teeth'] = (32 - merged_df['general_missing_teeth_count'])
    merged_df['number_of_surfaces'] = merged_df['number_of_teeth'] * 6

    ## Bleeding Index ##
    merged_df['number_of_bleeding_surfaces'] = merged_df['sum_of_surfaces_with_bleeding']
    merged_df['bleeding_index'] = merged_df['number_of_bleeding_surfaces'] / merged_df['number_of_surfaces']

    ## Bleeding Teeth ##
    merged_df['number_of_bleeding_teeth'] = merged_df['sum_of_teeth_with_bleeding']
    merged_df['pcnt_of_teeth_with_bleeding'] = (merged_df['number_of_bleeding_teeth'] / (32 - merged_df['general_missing_teeth_count']))*100

    ## Suppuration Index ##
    merged_df['number_of_suppuration_surfaces'] = merged_df['sum_of_surfaces_with_suppuration']
    merged_df['suppuration_index'] = merged_df['number_of_suppuration_surfaces'] / merged_df['number_of_surfaces']

    ## Suppuration Teeth ##
    merged_df['number_of_suppuration_teeth'] = merged_df['sum_of_teeth_with_suppuration']
    merged_df['pcnt_of_teeth_with_suppuration'] = (merged_df['number_of_suppuration_teeth'] / (32 - merged_df['general_missing_teeth_count']))*100

    new = merged_df[['research_id', 'exam_id', 'exam_date', 'exam_type',
                     'number_of_teeth', 'number_of_surfaces',
                     'number_of_bleeding_teeth', 'number_of_bleeding_surfaces',
                     'bleeding_index', 'pcnt_of_teeth_with_bleeding',
                     'number_of_suppuration_surfaces', 'suppuration_index',
                     'number_of_suppuration_teeth', 'pcnt_of_teeth_with_suppuration', 'plaque_index', 'percent_of_plaque_surfaces'
                     ]].copy()

    return new


## Combine Variables ##

def combine_demographics_and_variables_wide(merge_columns, demographic_data, exam_datasets):
    """
    Combine exam-level datasets (multiple exams per patient) with patient-level demographic data.
    Exam data is merged on exam-level columns (e.g., 'research_id', 'exam_id', 'exam_date', 'exam_type'),
    and then demographic data is merged using 'research_id', ensuring that patients without exams are retained.
    """
    # First, merge the exam-level datasets using an outer join.
    combined_exam_data = exam_datasets[0]
    for dataset in exam_datasets[1:]:
        combined_exam_data = pd.merge(combined_exam_data, dataset, on=merge_columns, how='outer')

    combined_exam_data = combined_exam_data.replace('DNE', pd.NA)

    # Now merge the demographic data with the exam-level data.
    # We use a left join so that all patient demographic records are included even if there are no matching exam records.
    combined_data = pd.merge(demographic_data, combined_exam_data, on='research_id', how='left')

    # Sort the data per patient and then by exam_date (assuming exam_date is in a suitable datetime format).
    combined_data.sort_values(by=['research_id', 'exam_date'], ascending=True, inplace=True)

    return combined_data


def apply_one_hot_encoding_on_demographics(merged_df):
    """
    Performs one-hot encoding on the demographics data for specific columns:
    - gender: F -> 0, M -> 1
    - active: FALSE -> 0, TRUE -> 1
    - periodontal_disease_risk: Creates four columns (missing, high, moderate, low)
    - past_periodontal_treatment: Creates three columns (missing, true, false)
    """
    # Copy the DataFrame to avoid modifying the original
    df = merged_df.copy()

    # One-hot encode gender
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

    # Age at exam
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')
    age_series = (df['exam_date'] - df['date_of_birth']).dt.days / 365.25

    # Insert age_at_exam right after date_of_birth
    df.insert(
        df.columns.get_loc('date_of_birth') + 1,  # position in columns
        'age_at_exam',
        age_series
    )

    # One-hot encode active
    df['active'] = df['active'].map({False: 0, True: 1})

    columns_with_categories = {
        'tobacco_user': ['Missing', 'Yes', 'No'],
        'periodontal_disease_risk': ['Missing', 'High', 'Moderate', 'Low'],
        'past_periodontal_treatment': ['Missing', 'Yes', 'No']
    }

    for col, categories in columns_with_categories.items():
        col_index = df.columns.get_loc(col)

        for cat in categories:
            new_col_name = f'{col}_{cat.lower()}'
            df.insert(
                col_index + 1,                       # position for new column
                new_col_name,
                (df[col] == cat).astype(int)         # 1 if matches category, else 0
            )
            col_index += 1
        df.drop(columns=[col], inplace=True)

    return df


def curate_raw_data_wide():
    """
    Function to curate data.
    """
    df_bleeding_pre = load_processed_bleeding_data()
    df_chart_general = load_processed_chart_general_data()
    df_chart_restoration = load_processed_chart_restorative_data()
    df_furcation_pre = load_processed_furcation_data()
    df_index_pre = load_cleaned_index_data_modified()
    df_mag_pre = load_processed_mag_data()
    df_mobility_pre = load_processed_mobility_data()
    df_pocket_raw = load_processed_pockets_data()
    df_recession_raw = load_processed_recessions_data()
    df_suppuration_raw = load_processed_suppuration_data()

    df_bleeding = create_tooth_site_level_variable_data(df_bleeding_pre, "bleeding")
    df_general = create_chart_general_df(df_chart_general)
    df_restore = create_tooth_level_restorative_data(df_chart_restoration)
    df_demographic_data = load_processed_demographic_data()
    df_furcation = create_tooth_site_level_variable_data(df_furcation_pre, "furcation")
    df_mag = create_tooth_site_level_variable_data(df_mag_pre, "mag")
    df_mobility = create_tooth_site_level_variable_data(df_mobility_pre, "mobility")
    df_pocket = create_tooth_site_level_variable_data(df_pocket_raw, "pocket")
    df_recessions = create_tooth_site_level_variable_data(df_recession_raw, "recession")
    df_suppuration = create_tooth_level_variable_data_suppuration_df(df_suppuration_raw, df_index_pre)
    df_index = index_curator(df_general, df_bleeding, df_suppuration, df_index_pre)

    variables_to_combine = [df_bleeding, df_furcation, df_mag, df_mobility, df_pocket, df_recessions, df_suppuration, df_index]
    merge_columns = ['research_id', 'exam_id', 'exam_date', 'exam_type']

    merged_df = combine_demographics_and_variables_wide(merge_columns, df_demographic_data, variables_to_combine)
    merged_df_demo_modified = apply_one_hot_encoding_on_demographics(merged_df)


    return merged_df_demo_modified


if __name__ == "__main__":

    df = curate_raw_data_wide()
    save_processed_data_to_csv(CURATED_PROCESSED_PATH, df)

    # html_content = df.to_html(index=False)
    #
    # # Save the HTML to a file:
    # html_file = "df_wide.html"
    # with open(html_file, "w") as f:
    #     f.write(html_content)
    #
    # try:
    #     # Try to get the Safari browser. This works on macOS.
    #     safari = webbrowser.get('safari')
    #     safari.open('file://' + os.path.abspath(html_file))
    # except webbrowser.Error:
    #     print("Safari browser not found, please open the file manually.")