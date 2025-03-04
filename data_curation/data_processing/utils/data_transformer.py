import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.data_parser import extract_surface_type_regex, \
    process_mag_value, process_mobility_value, process_tooth_integer_values, \
    map_surface_to_site_endo, clean_value, map_surface_to_site_restore
from deep_learning_dentistry.data_curation.data_processing.utils.functions import create_tooth_site_columns, \
    generate_teeth_order, tooth_quadrant_determiner, generate_teeth_site_mapping, get_largest_number
from deep_learning_dentistry.data_curation.data_processing.utils.teeth_mapping import tooth_value_mapper


## Application Wide Dataset Transformation Function ##

def transform_dataset(df_raw,
                      research_id="ResearchID",
                      date_col="CHART DATE",
                      exam_type_col="CHART TITLE",
                      exam_col = "CHART ID",
                      variable = None):
    """
    Takes a wide dataframe (raw dental data), uses your process functions, and melts to a site-level
    dataframe (like your 'yellow table').

    Returns the transformed, long-format site-level data.
    """
    final_rows = []

    for _, row in df_raw.iterrows():
        search_id = row.get(research_id, None)
        exam_id = row.get(exam_col, None)
        exam_date = row.get(date_col, None)
        exam_type = row.get(exam_type_col, None)

        if variable == "bleeding":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "bleeding"]
            exam_df = row_generator_bleeding_and_suppuration(row, "bleeding")
        elif variable == "chart_endo":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "tooth_condition_endodontics", "tooth_notes_endodontics", "endo_incisal", "endo_occlusal"]
            exam_df = row_generator_chart_endo(row)
        elif variable == "chart_general":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "treatment", "tooth_quadrant", "tooth_id", "tooth_site", "missing_status", "teeth_issues"]
            exam_df = row_generator_chart_general(row)
        elif variable == "chart_restore":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "tooth_notes_restorative","has_bridge_retainer_3/4_Crown",
                            "has_bridge_retainer_veneer", "has_veneer", "has_fillings_or_caries", "has_inlay", "restored_material"]
            exam_df = row_generator_chart_restore(row)
        elif variable == "demographic_data":
            column_order = ["research_id", "gender", "date_of_birth", "city", "postal_code", "state", "country", "education_level", "occupation", "active", "tobacco_user", "periodontal_disease_risk", "past_periodontal_treatment"]
            exam_df = row_generator_demographics_data(row)
        elif variable == "furcation":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "furcation"]
            exam_df = row_generator_furcation(row, "furcation")
        elif variable == "index":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "bleeding_index", "suppuration_index", "plaque_index",
                "missing_teeth", "percent_of_bleeding_surfaces","percent_of_suppuration_surfaces", "percent_of_plaque_surfaces"]
            exam_df = row_generator_index(row)
        elif variable == "mobility":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "mobility"]
            exam_df = row_generator_mobility_and_mag(row, "mobility")
        elif variable == "mag":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "mag"]
            exam_df = row_generator_mobility_and_mag(row, "mag")
        elif variable == "pockets":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "pocket"]
            exam_df = row_generator_pockets_and_recession(row, "pocket")
        elif variable == "recessions":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "recession"]
            exam_df = row_generator_pockets_and_recession(row, "recession")
        elif variable == "suppuration":
            column_order = ["research_id", "exam_id", "exam_date", "exam_type", "tooth_quadrant", "tooth_id", "tooth_site", "suppuration"]
            exam_df = row_generator_bleeding_and_suppuration(row, "suppuration")

        if variable != "demographic_data":
            for tooth_row in exam_df:
                tooth_row.update({
                    "research_id": search_id,
                    "exam_date": exam_date,
                    "exam_type": exam_type,
                    "exam_id": exam_id,
                })

        final_rows.extend(exam_df)

    final_df = pd.DataFrame(final_rows, columns=column_order)

    return final_df

## Row Generators For Processed Dataset ##

def row_generator_bleeding_and_suppuration(row, variable=None):
    """
    Transforms a single row of the dataset into a long format where each site
    of each tooth is represented as an individual row.

    Returns: A list of dictionaries representing the expanded data for each site.
    """
    processed_rows = []

    # Identify columns representing teeth
    teeth_columns = [col for col in row.index if "-" in col]

    for column in teeth_columns:
        # Split the column name into tooth ID and site ID
        tooth_id, site_id = column.split("-")
        tooth_id = int(tooth_id)

        # Get the quadrant using the tooth_quadrant_determiner function
        quadrant = tooth_quadrant_determiner(tooth_id)
        quadrant_value = int(quadrant[-1])

        # Get the value for the current tooth-site
        value = int(row[column])

        # Append the processed row data
        processed_rows.append({
            "tooth_quadrant": quadrant_value,
            "tooth_id": tooth_id,
            "tooth_site": site_id,
            variable: value
        })

    return processed_rows


def row_generator_chart_endo(row):
    """
    Transforms a single row of the dataset into a long format for endodontics.
    """
    processed_rows = []
    teeth_columns = [col for col in row.index if "-" in col]
    tooth_condition = row['TOOTH CONDITION']

    for column in teeth_columns:
        tooth_id, site_id = column.split("-")
        tooth_id = int(tooth_id)

        value = row[column]

        quadrant = tooth_quadrant_determiner(tooth_id)
        quadrant_value = int(quadrant[-1])

        tooth_notes = row['TOOTH NOTES']
        value_parsed = map_surface_to_site_endo(value)
        processed_rows.append({
            "tooth_condition_endodontics": tooth_condition,
            "tooth_notes_endodontics": tooth_notes,
            "tooth_quadrant": quadrant_value,
            "tooth_id": tooth_id,
            "tooth_site": site_id,
            "endo_incisal": value_parsed[0],
            "endo_occlusal": value_parsed[1]
        })
    return processed_rows


def row_generator_chart_general(row):
    """
    Transforms a single row of the dataset in chart_general into a long format where each site
    of each tooth is represented as an individual row.

    Returns: A list of dictionaries representing the expanded data for each site.
    """
    rows = []

    teeth_labels = generate_teeth_order()
    teeth_site_mapping = generate_teeth_site_mapping()

    for tooth_id in teeth_labels["teeth_order"]:
        treatment_col = "TREATMENT"
        missing_col = f"Tooth {tooth_id} - Missing Status"
        other_col  = f"Tooth {tooth_id} - Other Issues"

        treatment_data = row[treatment_col]
        missing_data = row[missing_col]
        other_data = row[other_col]

        quadrant = tooth_quadrant_determiner(tooth_id)
        quadrant_value = int(quadrant[-1])
        # Choose the appropriate mapping based on the quadrant
        site_mapping = teeth_site_mapping["upper_mapping"] if quadrant in ["Q1", "Q2"] else teeth_site_mapping["lower_mapping"]

        for site in range(6):
            rows.append({
                "treatment": treatment_data,
                "tooth_quadrant": quadrant_value,
                "tooth_id": tooth_id,
                "tooth_site": site_mapping[site],
                "missing_status": missing_data,
                "teeth_issues": other_data
            })

    return rows


def row_generator_chart_restore(row):
    """
    Transforms a single row of the chart_restore dataset into a long format
    where each site of each tooth is represented as an individual row.

    Returns: A list of dictionaries representing the expanded data for each site.
    """
    processed_rows = []

    # Mapping of tooth condition descriptions to the corresponding restoration flag
    conditions_map = {
        "Bridge Retainer 3/4 Crown": "has_bridge_retainer_3/4_Crown",
        "Bridge Retainer Veneer": "has_bridge_retainer_veneer",
        "Veneer": "has_veneer",
        "Fillings / Caries": "has_fillings_or_caries",
        "Inlay": "has_inlay"
    }

    # Identify columns representing tooth sites (those containing '-')
    teeth_columns = [col for col in row.index if "-" in col]

    for column in teeth_columns:
        # Split the column name into tooth ID and site ID (e.g., "18-A")
        tooth_id_str, site_id = column.split("-")
        tooth_id = int(tooth_id_str)

        # Determine the quadrant using your tooth_quadrant_determiner function
        quadrant = tooth_quadrant_determiner(tooth_id)
        quadrant_value = int(quadrant[-1])  # e.g., "Q2" -> 2

        # Get the value for the current tooth-site (expected 0 or 1)
        value = int(row[column])

        # Create a base dictionary with all restoration flags set to 0 (or None for notes/material)
        row_dict = {
            "tooth_quadrant": quadrant_value,
            "tooth_id": tooth_id,
            "tooth_site": site_id,
            "tooth_notes_restorative": 0,
            "has_bridge_retainer_3/4_Crown": 0,
            "has_bridge_retainer_veneer": 0,
            "has_veneer": 0,
            "has_fillings_or_caries": 0,
            "has_inlay": 0,
            "restored_material": 0
        }

        # If the tooth-site is marked (i.e. value == 1), apply the condition from the row
        if value == 1:
            # Get the condition (and material) from the row; adjust key names as necessary
            condition = row.get("TOOTH CONDITION", None)
            material = row.get("MATERIAL", None)
            notes = row.get("TOOTH NBR & NOTES", None)
            if condition in conditions_map:
                condition_key = conditions_map[condition]
                row_dict[condition_key] = 1
                row_dict['tooth_notes_restorative'] = notes
            # If material info is provided, include it
            if material is not None:
                row_dict["restored_material"] = material

        processed_rows.append(row_dict)

    return processed_rows

def row_generator_furcation(row, variable=None):
    """
    Transforms a single row of the dataset into a long format where each site
    of each tooth is represented as an individual row.

    Returns: A list of dictionaries representing the expanded data for each site.
    """
    processed_rows = []

    # Identify columns representing teeth
    teeth_columns = [col for col in row.index if "-" in col]

    for column in teeth_columns:
        # Split the column name into tooth ID and site ID
        tooth_id, site_id = column.split("-")
        tooth_id = int(tooth_id)

        # Get the quadrant using the tooth_quadrant_determiner function
        quadrant = tooth_quadrant_determiner(tooth_id)
        quadrant_value = int(quadrant[-1])

        # Get the value for the current tooth-site
        try:
            value = int(row[column])
        except ValueError:
            value = "Not Available"

        # Append the processed row data
        processed_rows.append({
            "tooth_quadrant": quadrant_value,
            "tooth_id": tooth_id,
            "tooth_site": site_id,
            variable: value
        })

    return processed_rows


def row_generator_demographics_data(row):
    """
    Generates a row for demographic data, replacing missing values (pd.NA or "") with "Missing".
    """
    rows = [{
        "research_id": clean_value(row["ResearchID"]),
        "gender": clean_value(row["Gender"]),
        "date_of_birth": clean_value(row["DateOfBirth"]),
        "city": clean_value(row["City"]),
        "postal_code": clean_value(row["PostalCode"]),
        "state": clean_value(row["State"]),
        "country": clean_value(row["Country"]),
        "education_level": clean_value(row["Education Level"]),
        "occupation": clean_value(row["Occupation"]),
        "active": clean_value(row["Active"]),
        "tobacco_user": clean_value(row["TobaccoUser"]),
        "periodontal_disease_risk": clean_value(row["Periodontal disease risk"]),
        "past_periodontal_treatment": clean_value(row["Past Periodontal Treatmen"])
    }]

    return rows


def row_generator_index(row):
    """
    Generates a row for index data.
    """
    rows = []
    teeth_labels = generate_teeth_order()
    teeth_site_mapping = generate_teeth_site_mapping()

    for tooth_id in teeth_labels["teeth_order"]:
        quadrant = tooth_quadrant_determiner(tooth_id)
        mapping = teeth_site_mapping["upper_mapping"] if quadrant in ["Q1", "Q2"] else teeth_site_mapping["lower_mapping"]
        quadrant_value = int(quadrant[-1])

        for site in range(6):
            rows.append({
                "tooth_quadrant": quadrant_value,
                "tooth_id": tooth_id,
                "tooth_site": mapping[site],
                "bleeding_index": row['BLEEDING_INDEX'],
                "suppuration_index": row['SUPPURATION_INDEX'],
                "plaque_index": row['PLAQUE_INDEX'],
                "missing_teeth": row['NVL(GCOUNT_OF_MISSING_TEETH,0)'],
                "percent_of_bleeding_surfaces": row['NVL(PCNT_OF_BLEEDING_SURFACES,0)'],
                "percent_of_suppuration_surfaces": row['NVL(PCNT_OF_SUPPURATION_SURFACES,0)'],
                "percent_of_plaque_surfaces": row['NVL(PCNT_OF_PLAQUE_SURFACES,0)']
            })

    return rows


def row_generator_mobility_and_mag(row, variable=None):
    """
    Processes all teeth for a single row, which is actually a patient exam, and generates site-level data for pockets and recession.
    Returns a list of dictionaries representing the processed rows for each site in all teeth.
    """
    rows = []

    teeth_labels = generate_teeth_order()
    teeth_site_mapping = generate_teeth_site_mapping()

    for tooth_id in teeth_labels["teeth_order"]:
        cell_value = row[f'{tooth_id}']

        if variable == "mobility":
            parsed_value = process_mobility_value(cell_value)
        elif variable == "mag":
            parsed_value = process_mag_value(cell_value)
        else:
            parsed_value = cell_value  # Fallback

        if isinstance(parsed_value, (int, float)):
            parsed_value = round(parsed_value, 1)

        quadrant = tooth_quadrant_determiner(tooth_id)
        mapping = (teeth_site_mapping["upper_mapping"]
                   if quadrant in ["Q1", "Q2"]
                   else teeth_site_mapping["lower_mapping"])
        quadrant_value = int(quadrant[-1])

        for site in range(6):
            rows.append({
                "tooth_quadrant": quadrant_value,
                "tooth_id": tooth_id,
                "tooth_site": mapping[site],
                variable: parsed_value,
            })

    return rows


def row_generator_pockets_and_recession(row, variable=None):
    """
    Processes all teeth for a single row, which is actually a patient exam, and generates site-level data for pockets and recession.
    Returns a list of dictionaries representing the processed rows for each site in all teeth.
    """
    rows = []

    teeth_labels = generate_teeth_order()
    # Retrieve the mapping dictionary containing both upper and lower mappings
    teeth_site_mapping = generate_teeth_site_mapping()

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

        quadrant = tooth_quadrant_determiner(tooth_id)
        site_values = transformed_buccal_values + transformed_backside_values

        # Select the appropriate mapping based on the quadrant
        mapping = teeth_site_mapping["upper_mapping"] if quadrant in ["Q1", "Q2"] else teeth_site_mapping["lower_mapping"]
        quadrant_value = int(quadrant[-1])

        for site in range(6):
            rows.append({
                "tooth_quadrant": quadrant_value,
                "tooth_id": tooth_id,
                "tooth_site": mapping[site],
                variable: site_values[site],
            })

    return rows


## Row Generator For Cleaned Data ##

def row_generator_chart_endo_cleaned_data(row, tooth_site_columns):
    tooth_site_matrix = pd.DataFrame(pd.NA, index=[0], columns=tooth_site_columns)

    # Extract tooth number
    tooth_number = row['TOOTH NBR']  # Column with tooth number

    # Identify all columns corresponding to the given tooth number
    matching_columns = [col for col in tooth_site_columns if col.startswith(f"{tooth_number}-")]

    # Set these columns to 1
    tooth_site_matrix.loc[0, matching_columns] = 1

    return tooth_site_matrix


def row_generator_chart_restore_cleaned_data(row, tooth_site_columns):
    tooth_site_matrix = pd.DataFrame(pd.NA, index=[0], columns=tooth_site_columns)

    # Extract tooth number
    tooth_number = row['TOOTH NBR']  # Column with tooth number
    tooth_surface = row['TOOTH SURFACE']

    # Identify all columns corresponding to the given tooth number
    matching_columns = [col for col in tooth_site_columns if col.startswith(f"{tooth_number}-")]

    # Set these columns to 1
    tooth_site_matrix.loc[0, matching_columns] = tooth_surface

    return tooth_site_matrix


def row_generator_b_s_cleaned_data(grouped_data_list, tooth_site_columns):
    """
    Processes grouped data to populate a tooth-site matrix where 1 indicates a match (bleeding or suppuration).
    - Using grouped_data (list of tuples): Each tuple contains (tooth, area, surface), return a dataFrame with
      columns for each tooth-site and values indicating matches.
    """
    tooth_site_matrix = pd.DataFrame(0, index=[0], columns=tooth_site_columns)
    for group in grouped_data_list:
        tooth = group[0]  # First element is the tooth number
        area = group[1]  # Second element is the area (e.g., 'Maxillary - Buccal')
        surface = group[2]  # Third element is the tooth surface

        location = extract_surface_type_regex(area)
        grouped_data = [location, surface]

        site = tooth_value_mapper(grouped_data, tooth, variable = "bleeding and suppuration")

        if site is not None:
            column_name = f"{tooth}-{site}"
            if column_name in tooth_site_matrix.columns:
                tooth_site_matrix.loc[0, column_name] = int("1")

    return tooth_site_matrix

## Transform Into Cleaned Dataset ##

def transform_dataset_to_clean_chart_endo(df):
    tooth_site_columns = create_tooth_site_columns()
    final_df = pd.DataFrame(pd.NA, index=range(len(df)), columns=tooth_site_columns)

    for idx, row in df.iterrows():
        output = row_generator_chart_endo_cleaned_data(row, tooth_site_columns)
        final_df.loc[idx] = output.loc[0]

    # Include metadata from original dataset
    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE", "TOOTH CONDITION", "TOOTH NOTES"]
    metadata = df[metadata_columns]

    # Combine metadata with the transformed tooth-site matrix
    combined_df = pd.concat([metadata.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)

    return combined_df


def transform_dataset_to_clean_chart_restore(df, variable=None):
    df = df.drop(columns=['TOOTH SURFACE'])
    df = df.drop_duplicates()
    df['TOOTH NBR & NOTES'] = df['TOOTH NBR'].astype(str) + ": " + df['TOOTH NOTES']
    df['MATERIAL'] = df['MATERIAL'].str.strip()

    grouped_df = df.groupby(
        [col for col in df.columns if col != 'MATERIAL'], dropna=False
    )['MATERIAL'].agg(lambda x: ', '.join(set(x.dropna()))).reset_index()

    new_df = grouped_df.drop(columns=['TOOTH NOTES'])

    tooth_site_columns = create_tooth_site_columns()
    final_df = pd.DataFrame(0, index=range(len(new_df)), columns=tooth_site_columns)

    for idx, row in new_df.iterrows():
        tooth_number = row['TOOTH NBR']
        matching_columns = [
            col for col in tooth_site_columns if col.startswith(f"{tooth_number}-")
        ]
        final_df.loc[idx, matching_columns] = 1

    metadata_columns = [
        "ResearchID", "CHART ID", "CHART DATE", "CHART TITLE",
        "TOOTH CONDITION", "MATERIAL", "TOOTH NBR & NOTES"
    ]
    metadata = new_df[metadata_columns]
    combined_df = pd.concat(
        [metadata.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1
    )

    columns_before_notes = combined_df.columns[:combined_df.columns.get_loc('TOOTH NBR & NOTES')].tolist()

    agg_dict = {"TOOTH NBR & NOTES": lambda x: " | ".join(x)}
    for col in tooth_site_columns:
        agg_dict[col] = "max"  # or "sum", if you prefer

    merged_df = (
        combined_df
        .groupby(columns_before_notes, dropna=False, as_index=False)
        .agg(agg_dict)
    )

    return merged_df


def transform_dataset_to_clean_chart_general(df):
    """
    Transforms the dataset to generate a clean DataFrame.
    Each row includes metadata and tooth-level columns for 'Missing' and 'Other'.
    Rows are grouped by CHART ID, and values are aggregated as necessary.
    Columns are rearranged according to teeth_order.
    """
    # Ensure teeth order is defined
    teeth_order = generate_teeth_order()["teeth_order"]

    # Precompute relevant columns
    tooth_columns = [f"T{tooth}" for tooth in teeth_order]
    all_tooth_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in tooth_columns)]

    # Initialize the transformed data
    transformed_data = []

    # Iterate over rows
    for idx, row in df.iterrows():
        # Extract metadata
        metadata = {
            "ResearchID": row['ResearchID'],
            "TREATMENT": row['TREATMENT'],
            "CHART TITLE": row['CHART TITLE'],
            "CHART ID": row['CHART ID'],
            "CHART DATE": row['CHART DATE'],
        }

        # Process tooth-related data
        tooth_data = {}
        for tooth in teeth_order:
            tooth_col_prefix = f"T{tooth}"
            relevant_cols = [col for col in all_tooth_cols if col.startswith(tooth_col_prefix)]

            # Initialize flags for Missing and Other
            missing_flag = 0
            other_content = []

            for col in relevant_cols:
                cell_value = row[col]
                if isinstance(cell_value, str):
                    if 'Missing' in cell_value:
                        missing_flag = 1
                    else:
                        other_content.append(cell_value.strip())

            # Add Missing and Other data to the row
            tooth_data[f"Tooth {tooth} - Missing Status"] = missing_flag
            tooth_data[f"Tooth {tooth} - Other Issues"] = ' // '.join(other_content) if other_content else '0'

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
            # Tooth-related columns: Aggregate
            **{
                col: lambda x: ' // '.join(
                    filter(lambda v: v != '0', map(str, x.dropna().unique()))  # Remove '0' before concatenating
                ) or '0'  # Default back to '0' if the result is empty
                if temp_df[col].dtype == 'object' or temp_df[col].dtype == 'O'
                else max(x)  # Take the maximum for numeric columns
                for col in temp_df.columns if col.startswith("Tooth")
            }
        })
    )

    # Rearrange columns according to teeth_order
    metadata_columns = ["ResearchID", "CHART ID", "TREATMENT", "CHART TITLE", "CHART DATE"]
    tooth_columns = [col for col in grouped_df.columns if col.startswith("Tooth")]

    # Separate Missing and Other columns
    missing_columns = [col for col in tooth_columns if 'Missing' in col]
    other_columns = [col for col in tooth_columns if 'Other' in col]

    # Sort Missing and Other columns based on teeth_order
    sorted_missing_columns = [f"Tooth {tooth} - Missing Status" for tooth in teeth_order if f"Tooth {tooth} - Missing Status" in missing_columns]
    sorted_other_columns = [f"Tooth {tooth} - Other Issues" for tooth in teeth_order if f"Tooth {tooth} - Other Issues" in other_columns]

    # Combine columns in the correct order
    new_column_order = metadata_columns + sorted_missing_columns + sorted_other_columns
    grouped_df = grouped_df[new_column_order]

    return grouped_df


def transform_dataset_to_clean_b_s(df):
    """
    Iterates over rows of the bleeding data and generates tooth-site matrices for each row.
    Combines the results into a single DataFrame where each row corresponds to a row in the input DataFrame.

    Returns a combined tooth-site matrix for all rows in the input DataFrame.
    """
    tooth_site_columns = create_tooth_site_columns()
    final_df = pd.DataFrame("0", index=range(len(df)), columns=tooth_site_columns)

    for idx, row in df.iterrows():
        df_row = row["Teeth Data"]
        output =  row_generator_b_s_cleaned_data(df_row, tooth_site_columns)
        final_df.loc[idx] = output.loc[0]

    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    metadata = df[metadata_columns]

    combined_df = pd.concat([metadata.reset_index(drop=True), final_df.reset_index(drop=True)], axis=1)

    return combined_df


def transform_dataset_to_clean_furcation(df):
    """
    Transforms raw furcation dataset to a cleaned dataset.
    """
    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    special_teeth = {18, 17, 16, 14, 24, 26, 27, 28, 48, 47, 46, 36, 37, 38}

    tooth_site_columns = create_tooth_site_columns()
    all_columns = metadata_columns + tooth_site_columns
    final_df = pd.DataFrame("0", index=df.index, columns=all_columns)

    for idx, row in df.iterrows():
        for meta_col in metadata_columns:
            final_df.loc[idx, meta_col] = row[meta_col]

    for col in tooth_site_columns:
        tooth_str = col.split("-")[0]
        try:
            tooth_num = int(tooth_str)
        except ValueError:
            continue

        if tooth_num not in special_teeth:
            final_df.loc[:, col] = "Not Available"

    for idx, row in df.iterrows():
        for tooth in [c for c in row.index if c not in metadata_columns]:
            value = get_largest_number(row[tooth])
            for col in tooth_site_columns:
                if col.startswith(f"{tooth}-"):
                    final_df.loc[idx, col] = value

    final_df.reset_index(drop=True, inplace=True)
    return final_df

def transform_dataset_to_clean_mag(dataframe):
    """
    Merges rows based on "ResearchID", "CHART TITLE", "CHART ID", "CHART DATE". For the remaining rows, merge.
    Returns a dataframe with merged rows. Specific for mag.
    """
    df = dataframe.replace({'':pd.NA})
    metadata_columns = ["ResearchID", "CHART ID", "CHART DATE", "CHART TITLE"]
    consolidated_df = df.groupby(metadata_columns, as_index=False).apply(lambda x: x.ffill().tail(1)).reset_index(drop=True)

    return consolidated_df


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