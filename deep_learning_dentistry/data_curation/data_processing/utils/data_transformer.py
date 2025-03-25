import pandas as pd
import numpy as np

from deep_learning_dentistry.data_curation.data_processing.utils.data_parser import extract_surface_type_regex, \
    process_mag_value, process_mobility_value, process_tooth_integer_values, \
    map_surface_to_site_endo, clean_value, map_surface_to_site_restore
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import create_tooth_site_columns, \
    generate_teeth_order, tooth_quadrant_determiner, generate_teeth_site_mapping, get_largest_number, \
    format_tooth_site_column
from deep_learning_dentistry.data_curation.data_processing.utils.teeth_mapping import tooth_value_mapper


def transform_dataset_to_clean_chart_restore(df):
    """ Transforms raw chart restore df to clean chart restore format. """

    def union_teeth(series):
        """ Each cell might contain '12, 24'. We want the final exam-level cell to have the union of all such strings. """
        all_teeth = set()
        for val in series.dropna():
            for t in val.split(","):
                t_str = t.strip()
                if t_str:
                    all_teeth.add(t_str)
        return ", ".join(sorted(all_teeth))

    def cumulative_union(series):
        """ For each exam row, merge (union) the current row's teeth with everything found so far for that patient. """
        union_so_far = set()
        results = []
        for val in series:
            current_set = set()
            if isinstance(val, str) and val.strip():
                current_set = {t.strip() for t in val.split(",")}
            union_so_far |= current_set
            results.append(", ".join(sorted(union_so_far)))
        return results

    def cumulative_concat(series):
        """ For each exam row, concatenate the current row's notes with all previous exam notes for that patient.
        Join using ' || ' as the delimiter. """
        accum = []
        results = []
        for val in series:
            if isinstance(val, str) and val.strip():
                accum.append(val.strip())
            results.append(" || ".join(accum))
        return results

    restoration_record = [
        "has_bridge_retainer_3/4_Crown",
        "has_bridge_retainer_veneer",
        "has_veneer",
        "has_fillings_or_caries",
        "has_inlay"
    ]

    work_done_that_day_record = [
        "worked_on_bridge_retainer_3/4_Crown",
        "worked_on_bridge_retainer_veneer",
        "worked_on_veneer",
        "worked_on_fillings_or_caries",
        "worked_on_inlay"
    ]

    restoration_mapping = {
        "has_bridge_retainer_3/4_Crown": "3/4 crown",
        "has_bridge_retainer_veneer": "bridge retainer veneer",
        "has_veneer": "veneer",
        "has_fillings_or_caries": "filling",  # or "caries"
        "has_inlay": "inlay"
    }

    work_done_mapping = {
        "worked_on_bridge_retainer_3/4_Crown": "3/4 crown",
        "worked_on_bridge_retainer_veneer": "bridge retainer veneer",
        "worked_on_veneer": "veneer",
        "worked_on_fillings_or_caries": "filling",  # or "caries"
        "worked_on_inlay": "inlay"
    }

    # Initial cleaning
    df = df.drop(columns=['TOOTH SURFACE'])
    df = df.drop_duplicates()
    df['TOOTH NBR & NOTES'] = df['TOOTH NBR'].astype(str) + ": " + df['TOOTH NOTES']

    # Initialize new columns for restorations and work-done
    for col in restoration_record:
        df[col] = ""
    for col in work_done_that_day_record:
        df[col] = np.nan  # So that if no work is found, it remains NaN

    # Ensure rows are sorted by patient and exam date
    df = df.sort_values(by=["ResearchID", "CHART DATE"])

    # Populate restoration and work-done columns from TOOTH CONDITION
    for idx, row in df.iterrows():
        tooth = str(row['TOOTH NBR'])
        condition = str(row['TOOTH CONDITION']).lower() if pd.notnull(row['TOOTH CONDITION']) else ""

        for col, keyword in restoration_mapping.items():
            if keyword in condition:
                current_val = df.at[idx, col]
                if current_val:
                    existing_teeth = {x.strip() for x in current_val.split(",")}
                    if tooth not in existing_teeth:
                        df.at[idx, col] = current_val + ", " + tooth
                else:
                    df.at[idx, col] = tooth

        for col, keyword in work_done_mapping.items():
            if keyword in condition:
                df.at[idx, col] = 1

    # Define grouping keys for exam-level aggregation
    group_keys = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]

    # Build aggregation dictionary for exam-level fields
    agg_dict = {
        "TOOTH NBR & NOTES": lambda x: " | ".join(sorted(x.dropna().unique())),
        "TOOTH CONDITION": lambda x: " | ".join(sorted(x.dropna().unique())),
        "MATERIAL": lambda x: " | ".join(sorted(x.dropna().unique()))
    }
    for col in restoration_record:
        agg_dict[col] = union_teeth
    for col in work_done_that_day_record:
        agg_dict[col] = lambda x: 1 if (x == 1).any() else np.nan

    # Aggregate to one row per exam
    exam_df = df.groupby(group_keys, dropna=False, as_index=False).agg(agg_dict)
    exam_df = exam_df.sort_values(by=["ResearchID", "CHART DATE"])

    # Carry forward restoration columns across exams
    for col in restoration_record:
        exam_df[col] = exam_df.groupby("ResearchID")[col].transform(cumulative_union)

    # Create Complete Notes column by cumulatively concatenating "TOOTH NBR & NOTES"
    exam_df["Complete Notes"] = exam_df.groupby("ResearchID")["TOOTH NBR & NOTES"].transform(cumulative_concat)

    # Reorder columns: move "Complete Notes" to be immediately before "TOOTH NBR & NOTES"
    cols = list(exam_df.columns)
    if "TOOTH NBR & NOTES" in cols and "Complete Notes" in cols:
        tn_index = cols.index("TOOTH NBR & NOTES")
        cols.remove("Complete Notes")
        cols.insert(tn_index, "Complete Notes")
        exam_df = exam_df[cols]

    return exam_df

def raw_to_cleaned_df_bleeding_and_sup(df):
    """ Iterates over rows of bleeding_combined_raw or suppuration_combined_raw maps variables to exact tooth site levels.
    Returns this ready-for-use df. """

    tooth_site_columns = create_tooth_site_columns()
    final_df = pd.DataFrame(0, index=range(len(df)), columns=tooth_site_columns)

    for idx, row in df.iterrows():
        df_row = row["Teeth Data"]
        output =  row_generator_b_s_cleaned_data(df_row, tooth_site_columns)
        final_df.loc[idx] = output.loc[0]

    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    metadata = df[metadata_columns]

    combined_df = pd.concat([metadata.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)
    combined_df = combined_df.drop_duplicates()

    return combined_df


def raw_to_cleaned_df_mobility(df, variable):
    """ Iterates over rows of dataframe. Since the raw dataframe only has tooth level data, the function maps to all sites
    of that tooth. """

    tooth_site_columns = create_tooth_site_columns()
    final_df = pd.DataFrame(0.0, index=range(len(df)), columns=tooth_site_columns)

    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    possible_tooth_columns = [col for col in df.columns if col not in metadata_columns]

    for idx, row in df.iterrows():
        for tooth_col in possible_tooth_columns:
            value = row[tooth_col]
            determine_mobility_value(idx, tooth_col, value, final_df)


    metadata = df.reindex(columns=metadata_columns).copy()

    combined_df = pd.concat([metadata.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)
    combined_df.drop_duplicates(inplace=True)

    return combined_df


def raw_to_cleaned_df_furcation(df):
    """ Transforms raw furcation dataset to a cleaned dataset. """

    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    special_teeth = {18, 17, 16, 14, 24, 26, 27, 28, 48, 47, 46, 36, 37, 38}

    tooth_site_columns = create_tooth_site_columns()
    all_columns = metadata_columns + tooth_site_columns

    final_df = pd.DataFrame("", index=df.index, columns=all_columns)

    for meta_col in metadata_columns:
        final_df[meta_col] = df[meta_col]

    for col in tooth_site_columns:
        parts = col.split("_")
        if len(parts) == 3:
            tooth_str = parts[1]
            try:
                tooth_num = int(tooth_str)
            except ValueError:
                tooth_num = None

            if tooth_num and tooth_num not in special_teeth:
                final_df[col] = "Not Available"

    possible_tooth_columns = [col for col in df.columns if col not in metadata_columns]
    for idx, row in df.iterrows():
        for tooth_col in possible_tooth_columns:
            furcation_value = get_largest_number(row[tooth_col])
            matching_cols = [col_name for col_name in tooth_site_columns if f"_{tooth_col}_" in col_name]
            for col_name in matching_cols:
                final_df.at[idx, col_name] = furcation_value


    final_df.reset_index(drop=True, inplace=True)
    return final_df


def raw_to_cleaned_df_mag(df):
    """ Iterate over rows of mag df. Some special cases to deal with on MAG. """

    df_new = df.replace({'': pd.NA})
    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    consolidated_df = (
        df_new.groupby(metadata_columns, as_index=False)
        .apply(lambda x: x.ffill().tail(1))
        .reset_index(drop=True)
    )

    tooth_site_columns = create_tooth_site_columns()
    final_df = pd.DataFrame(0, index=range(len(consolidated_df)), columns=tooth_site_columns)

    possible_tooth_columns = [
        col for col in consolidated_df.columns
        if col not in metadata_columns
    ]

    for idx, row in consolidated_df.iterrows():
        for tooth_col in possible_tooth_columns:
            cell_value = row[tooth_col]
            processed_values = process_mag_value(cell_value)
            matching_site_cols = [
                col_name for col_name in tooth_site_columns
                if f"_{tooth_col}_" in col_name
            ]
            for site_col in matching_site_cols:
                final_df.at[idx, site_col] = processed_values


    metadata = consolidated_df[metadata_columns].copy()
    combined_df = pd.concat([metadata.reset_index(drop=True),
                             final_df.reset_index(drop=True)], axis=1)
    combined_df.drop_duplicates(inplace=True)

    return combined_df


def raw_to_cleaned_df_pockets_and_rec(df):
    """ """

    # Prepare metadata columns for later use
    metadata_cols = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]

    teeth_labels = generate_teeth_order()
    teeth_site_mapping = generate_teeth_site_mapping()

    rows_data = []

    for idx, row in df.iterrows():
        row_dict = row[metadata_cols].to_dict()

        for tooth_id in teeth_labels["teeth_order"]:
            frontside_col = f"Tooth {tooth_id} B"

            if tooth_id in teeth_labels['upper_right_teeth'] or tooth_id in teeth_labels['upper_left_teeth']:
                backside_col = f"Tooth {tooth_id} P"
            elif tooth_id in teeth_labels['lower_right_teeth'] or tooth_id in teeth_labels['lower_left_teeth']:
                backside_col = f"Tooth {tooth_id} L"

            frontside_data = row[frontside_col]
            backside_data = row[backside_col]

            buccal_values = process_tooth_integer_values(frontside_data)
            backside_values = process_tooth_integer_values(backside_data)

            transformed_buccal_values = tooth_value_mapper(buccal_values, tooth_id, variable="pockets and recession")
            transformed_backside_values = tooth_value_mapper(backside_values, tooth_id, variable="pockets and recession")

            site_values = transformed_buccal_values + transformed_backside_values

            quadrant = tooth_quadrant_determiner(tooth_id)
            mapping = (teeth_site_mapping["upper_mapping"]
                       if quadrant in ["Q1", "Q2"]
                       else teeth_site_mapping["lower_mapping"])

            for i, site_label in enumerate(mapping):
                tooth_site_col = format_tooth_site_column(tooth_id, site_label)
                row_dict[tooth_site_col] = site_values[i]

        rows_data.append(row_dict)

    combined_df = pd.DataFrame(rows_data)
    combined_df.drop_duplicates(inplace=True)

    return combined_df

def raw_to_cleaned_df_chart_general(df):
    """
    Transforms the dataset to generate a clean DataFrame.
    Each row includes metadata and tooth-level columns for 'Missing' and 'Other'.
    Rows are grouped by CHART ID, and values are aggregated as necessary.
    Columns are rearranged according to teeth_order.
    The returned DataFrame is sorted by ResearchID and, within each ResearchID, by CHART DATE (oldest to newest).
    """
    teeth_order = generate_teeth_order()["teeth_order"]

    tooth_columns = [f"T{tooth}" for tooth in teeth_order]
    all_tooth_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in tooth_columns)]

    transformed_data = []

    for idx, row in df.iterrows():
        metadata = {
            "ResearchID": row['ResearchID'],
            "TREATMENT": row['TREATMENT'],
            "CHART TITLE": row['CHART TITLE'],
            "CHART ID": row['CHART ID'],
            "CHART DATE": row['CHART DATE'],
        }

        tooth_data = {}
        for tooth in teeth_order:
            tooth_col_prefix = f"T{tooth}"
            relevant_cols = [col for col in all_tooth_cols if col.startswith(tooth_col_prefix)]

            # Initialize flags for missing and other issues
            missing_flag = 0
            other_content = []

            for col in relevant_cols:
                cell_value = row[col]
                if isinstance(cell_value, str):
                    if 'Missing' in cell_value:
                        missing_flag = 1
                    else:
                        other_content.append(cell_value.strip())

            # New column names using the desired naming format.
            missing_col = f"q{tooth // 10}_{tooth}_missing_status"
            other_col = f"q{tooth // 10}_{tooth}_other_issues"
            tooth_data[missing_col] = missing_flag
            tooth_data[other_col] = ' // '.join(other_content) if other_content else '0'

        # Combine metadata and tooth data
        transformed_row = {**metadata, **tooth_data}
        transformed_data.append(transformed_row)

    # Convert list of dictionaries to DataFrame
    temp_df = pd.DataFrame(transformed_data)

    # Group by CHART ID and aggregate
    grouped_df = (
        temp_df.groupby("CHART ID", as_index=False)
        .agg({
            # Metadata: Take the first occurrence
            "ResearchID": "first",
            "TREATMENT": "first",
            "CHART TITLE": "first",
            "CHART DATE": "first",
            # Tooth-related columns: Aggregate by concatenation for object types or maximum for numeric types.
            **{
                col: (lambda x: ' // '.join(
                        filter(lambda v: v != '0', map(str, x.dropna().unique()))
                    ) or '0'
                    if temp_df[col].dtype == 'object' or temp_df[col].dtype == 'O'
                    else max(x)
                   )
                for col in temp_df.columns if col.startswith("q")
            }
        })
    )

    # Rearrange columns according to teeth_order.
    # Build final tooth columns order using the new naming convention.
    metadata_columns = ["ResearchID", "CHART ID", "TREATMENT", "CHART TITLE", "CHART DATE"]
    missing_columns = [f"q{tooth // 10}_{tooth}_missing_status" for tooth in teeth_order]
    other_columns = [f"q{tooth // 10}_{tooth}_other_issues" for tooth in teeth_order]

    new_column_order = metadata_columns + missing_columns + other_columns
    grouped_df = grouped_df.reindex(columns=new_column_order, fill_value=0)

    # Convert CHART DATE to datetime if not already, then sort by ResearchID and CHART DATE (oldest to newest)
    grouped_df["CHART DATE"] = pd.to_datetime(grouped_df["CHART DATE"], errors='coerce')
    grouped_df = grouped_df.sort_values(by=["ResearchID", "CHART DATE"], ascending=[True, True]).reset_index(drop=True)

    return grouped_df


def transform_dataset_to_clean_index(dataframe):
    """
    Merges rows based on "ResearchID", "CHART TITLE", "CHART ID", "CHART DATE". For the remaining rows, merge.
    Returns a dataframe with merged rows.
    """
    merge_columns = ["ResearchID", "CHART TITLE", "CHART ID", "CHART DATE"]

    reference_column = "CHART DATE"
    metadata_columns = dataframe.columns[:dataframe.columns.get_loc(reference_column) + 1]

    # Identify value columns (after "CHART DATE")
    value_columns = dataframe.columns[dataframe.columns.get_loc(reference_column) + 1:]

    # Group by the merge columns and aggregate
    merged_df = (
        dataframe.groupby(merge_columns, as_index=False)
        .agg({**{col: "first" for col in metadata_columns}, **{col: "max" for col in value_columns}})
    )

    return merged_df













## Helper Functions

def row_generator_b_s_cleaned_data(grouped_data_list, tooth_site_columns):
    """
    Processes grouped data to populate a tooth-site matrix where 1 indicates a match (bleeding or suppuration).
    - Using grouped_data (list of tuples): Each tuple contains (tooth, area, surface), return a dataFrame with
      columns for each tooth-site and values indicating matches.
    """
    tooth_site_matrix = pd.DataFrame(0, index=[0], columns=tooth_site_columns)

    for group in grouped_data_list:
        tooth = group[0]  # i.e. 18 or 22
        area = group[1]  # i.e. 'Maxillary - Buccal'
        surface = group[2]  # i.e. 'DB'

        location = extract_surface_type_regex(area)
        grouped_data = [location, surface]

        site = tooth_value_mapper(grouped_data, tooth, variable="bleeding and suppuration")

        if site is not None:
            column_name = format_tooth_site_column(tooth, site)
            if column_name in tooth_site_matrix.columns:
                tooth_site_matrix.loc[0, column_name] = 1  # Mark a match as 1.
            else:
                print(f"Column {column_name} not found in the matrix.")

    return tooth_site_matrix


def determine_mobility_value(idx, tooth_col, value, final_df):
    """
    """
    processed_val = process_mobility_value(value)
    tooth_num = int(tooth_col)

    if 10 <= tooth_num <= 29:
        site_labels = generate_teeth_site_mapping()["upper_mapping"]
    else:
        site_labels = generate_teeth_site_mapping()["lower_mapping"]

    for site in site_labels:
        tooth_site_col = format_tooth_site_column(tooth_num, site)
        if tooth_site_col in final_df.columns:
            final_df.loc[idx, tooth_site_col] = processed_val