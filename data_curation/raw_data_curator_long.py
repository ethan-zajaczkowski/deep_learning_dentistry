import pandas as pd
import numpy as np
from deep_learning_dentistry.data_curation.data_processing.bleeding_processor import load_processed_bleeding_data
from deep_learning_dentistry.data_curation.data_processing.chart_general_processor import \
    load_processed_chart_general_data
from deep_learning_dentistry.data_curation.data_processing.chart_restorative_processor import \
    load_processed_chart_restorative_data
from deep_learning_dentistry.data_curation.data_processing.demographics_data_processor import \
    load_processed_demographic_data
from deep_learning_dentistry.data_curation.data_processing.furcation_processor import load_processed_furcation_data
from deep_learning_dentistry.data_curation.data_processing.index_processor import load_processed_index_data
from deep_learning_dentistry.data_curation.data_processing.mobility_processor import load_processed_mobility_data
from deep_learning_dentistry.data_curation.data_processing.mag_processor import load_processed_mag_data
from deep_learning_dentistry.data_curation.data_processing.pockets_processor import load_processed_pockets_data
from deep_learning_dentistry.data_curation.data_processing.recessions_processor import load_processed_recessions_data
from deep_learning_dentistry.data_curation.data_processing.suppuration_processor import load_processed_suppuration_data
from deep_learning_dentistry.data_curation.data_processing.utils.config import CURATED_PROCESSED_PATH, TEMP_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_processed_data_to_csv, \
    save_cleaned_data_to_excel, generate_teeth_order, generate_teeth_site_mapping, fill_exam_suppuration


def combine_variables(datasets):
    """
    Combine datasets of bleeding, furcation, mag, mobility, pockets, recessions, suppuration
    """
    merge_columns = ['research_id', 'exam_id', 'exam_date', 'exam_type', 'tooth_quadrant', 'tooth_id', 'tooth_site']

    # Merge datasets iteratively to reduce memory usage
    combined_data = datasets[0]
    for dataset in datasets[1:]:
        combined_data = pd.merge(combined_data, dataset, on=merge_columns, how='outer')

    # Fill missing values with pd.NA
    combined_data = combined_data.replace('DNE', pd.NA)
    combined_data.sort_values(by='exam_date', ascending=True, inplace=True)

    final_df = rearrange_tooth_site(combined_data)
    return final_df


def rearrange_tooth_site(df):
    """
    Rearranges the tooth_site column based on tooth_id and provided mappings,
    grouped by research_id and exam_id, and ordered by teeth_order.
    """
    # Retrieve mappings for tooth_site from generate_teeth_site_mapping
    mapping_dict = generate_teeth_site_mapping()
    upper_mapping = mapping_dict["upper_mapping"]
    lower_mapping = mapping_dict["lower_mapping"]

    # Generate teeth order (ensure it returns a dictionary)
    teeth_order_dict = generate_teeth_order()
    teeth_order = teeth_order_dict["teeth_order"]

    # Ensure teeth_order is a dictionary
    if isinstance(teeth_order, list):
        # Convert list to dictionary where keys are tooth_id and values are their order
        teeth_order = {tooth_id: idx for idx, tooth_id in enumerate(teeth_order)}

    # Create a mapping function for sorting tooth_site
    def get_site_order(row):
        # Adjust this condition as needed for your data.
        # Here, if tooth_id is <= 30 we use the upper mapping; otherwise, the lower mapping.
        if row['tooth_id'] <= 30:
            return upper_mapping.index(row['tooth_site'])
        else:
            return lower_mapping.index(row['tooth_site'])

    # Add temporary columns for sorting
    df['site_order'] = df.apply(get_site_order, axis=1)
    df['tooth_order'] = df['tooth_id'].map(teeth_order).fillna(-1).astype(int)

    # Sort by research_id, exam_id, tooth_order, and site_order
    df.sort_values(by=['research_id', 'exam_id', 'tooth_order', 'site_order'], inplace=True)

    # Drop the temporary columns
    df.drop(columns=['site_order', 'tooth_order'], inplace=True)

    return df

def preprocess_merged_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the merged data by converting exam_date to datetime, sorting, and propagating missing_status.
    """
    # Ensure exam_date is a datetime object for proper comparison
    merged_data["exam_date"] = pd.to_datetime(merged_data["exam_date"])

    # Sort data by research_id, tooth_id, and exam_date to simplify processing
    merged_data = merged_data.sort_values(by=["research_id", "tooth_id", "exam_date"])

    # List of variables to propagate forward in time
    variables_to_propagate = [
        "missing_status",
        "has_bridge_retainer_3/4_Crown",
        "has_bridge_retainer_veneer",
        "has_veneer",
        "has_fillings_or_caries",
        "has_inlay",
        "restored_material",
    ]

    # Convert 'DNE' to pd.NA and ensure numeric dtype for each variable
    for var in variables_to_propagate:
        if var in merged_data.columns:  # Check if the column exists in the DataFrame
            merged_data[var] = pd.to_numeric(merged_data[var].replace('DNE', pd.NA), errors='coerce')

    # Propagate each variable forward in time using groupby and cummax
    for var in variables_to_propagate:
        if var in merged_data.columns:  # Check if the column exists in the DataFrame
            merged_data[var] = merged_data.groupby(["research_id", "tooth_id"])[var].cummax()

    return merged_data


def create_work_done_on_missing_teeth_column(merged_data: pd.DataFrame, columns_to_check: list) -> pd.DataFrame:
    """
    Creates the "work_done_on_missing_teeth" column based on the specified conditions.
    """
    merged_data["work_done_on_missing_teeth"] = (
        (merged_data["missing_status"] == 1) &
        (merged_data[columns_to_check].sum(axis=1) >= 1)
    ).astype(int)  # Convert boolean to integer (0 for False, 1 for True)

    return merged_data


def combined_general_restorative_var(chart_general_data, chart_restorative_data, dataset):
    """
    Main function to load, merge all datasets, preprocess, and analyze the combined data.
    """
    # Define the columns to merge on
    merge_columns = [
        "research_id",
        "exam_id",
        "exam_date",
        "exam_type",
        "tooth_quadrant",
        "tooth_id",
        "tooth_site"
    ]

    # Merge datasets iteratively to reduce memory usage
    merged_data = chart_general_data
    for dataset in [chart_restorative_data, dataset]:
        merged_data = pd.merge(merged_data, dataset, on=merge_columns, how='outer')

    # Preprocess the merged data
    merged_data = preprocess_merged_data(merged_data)

    # Define the three columns to check for work done on missing teeth
    columns_to_check = ["has_bridge_retainer_3/4_Crown", "has_bridge_retainer_veneer", "has_veneer"]

    # Create the new column "work_done_on_missing_teeth"
    merged_data = create_work_done_on_missing_teeth_column(merged_data, columns_to_check)

    # Fill missing values with pd.NA
    merged_data = merged_data.replace('DNE', pd.NA)
    merged_data.sort_values(by='exam_date', ascending=True, inplace=True)

    final_df = rearrange_tooth_site(merged_data)
    return final_df


def one_hot_encode_demographics(demographics_data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs one-hot encoding on the demographics data for specific columns:
    - gender: F -> 0, M -> 1
    - active: FALSE -> 0, TRUE -> 1
    - periodontal_disease_risk: Creates four columns (missing, high, moderate, low)
    - past_periodontal_treatment: Creates three columns (missing, true, false)
    """
    # Copy the DataFrame to avoid modifying the original
    df = demographics_data.copy()

    # One-hot encode gender
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

    # One-hot encode active
    df['active'] = df['active'].map({False: 0, True: 1})

    # One-hot encode periodontal_disease_risk
    periodontal_risk_categories = ['Missing', 'High', 'Moderate', 'Low']
    for category in periodontal_risk_categories:
        df[f'periodontal_disease_risk_{category.lower()}'] = (df['periodontal_disease_risk'] == category).astype(int)

    # Drop the original periodontal_disease_risk column
    df.drop(columns=['periodontal_disease_risk'], inplace=True)

    # One-hot encode past_periodontal_treatment
    treatment_categories = ['Missing', 'Yes', 'No']
    for category in treatment_categories:
        df[f'past_periodontal_treatment_{category.lower()}'] = (df['past_periodontal_treatment'] == category).astype(int)

    # Drop the original past_periodontal_treatment column
    df.drop(columns=['past_periodontal_treatment'], inplace=True)

    return df


def merge_demographics_full(demographics_data, full_data):
    """
    Merges demographics data with the full dataset.
    """
    merge_columns = ["research_id"]
    encoded_df = one_hot_encode_demographics(demographics_data)
    merged_data = pd.merge(encoded_df, full_data, on=merge_columns, how='outer')

    merged_data["exam_date"] = pd.to_datetime(merged_data["exam_date"])
    merged_data["date_of_birth"] = pd.to_datetime(merged_data["date_of_birth"])

    merged_data['tooth_id'] = merged_data['tooth_id'].astype('Int64')
    merged_data['tooth_quadrant'] = merged_data['tooth_quadrant'].astype('Int64')
    merged_data['exam_id'] = merged_data['exam_id'].astype('Int64')
    merged_data['missing_status'] = merged_data['missing_status'].astype('Int64')
    merged_data['work_done_on_missing_teeth'] = merged_data['work_done_on_missing_teeth'].astype('Int64')

    final_df = rearrange_columns_keep_first_half(merged_data)

    return final_df


def rearrange_columns_keep_first_half(df):
    """
    Rearranges columns of the DataFrame such that the first half remains unchanged,
    and columns after 'missing_status' are rearranged according to a specified order.
    """
    # Define the columns to move after 'missing_status'
    columns_to_rearrange = [
        "work_done_on_missing_teeth", "teeth_issues",
        "bleeding", "furcation", "mobility", "mag", "pocket", "recession", "suppuration",
        "bleeding_index", "suppuration_index", "plaque_index", "missing_teeth",
        "percent_of_bleeding_surfaces", "percent_of_suppuration_surfaces", "percent_of_plaque_surfaces",
        "tooth_notes_restorative", "has_bridge_retainer_3/4_Crown",
        "has_bridge_retainer_veneer", "has_veneer", "has_fillings_or_caries",
        "has_inlay", "restored_material"
    ]

    # Identify the first half of the columns (before and including 'missing_status')
    missing_status_index = list(df.columns).index("missing_status")
    first_half_columns = list(df.columns[:missing_status_index + 1])  # Include 'missing_status'

    # Identify remaining columns that were not explicitly rearranged
    remaining_columns = [col for col in df.columns if col not in first_half_columns + columns_to_rearrange]

    # Create the new column order
    new_column_order = first_half_columns + columns_to_rearrange + remaining_columns

    # Rearrange the DataFrame
    return df[new_column_order]


def process_columns(df):
    """
    Processes columns in the DataFrame according to specified rules.
    """
    # Columns to convert to numpy.int64 without checks
    int_columns = ['research_id', 'exam_id', 'tooth_id', 'tooth_side']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(np.int64)

    # Convert `exam_date` to pd.datetime with format YYYY-MM-DD
    if 'exam_date' in df.columns:
        df['exam_date'] = pd.to_datetime(df['exam_date'], format='%Y-%m-%d', errors='coerce')

    # Ensure `exam_type` is string
    if 'exam_type' in df.columns:
        df['exam_type'] = df['exam_type'].astype(str)

    # Columns to process with checks
    checked_columns = [
        'pocket', 'recession', 'suppuration', 'bleeding', 'mobility', 'mag', 'furcation',
        'missing_status', 'bleeding_index', 'suppuration_index', 'plaque_index',
        'missing_teeth', 'percent_of_bleeding_surfaces',
        'percent_of_suppuration_surfaces', 'percent_of_plaque_surfaces'
    ]
    for col in checked_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if x in ['Missing', 'DNE', 'Error'] else int(float(x))
            ).astype('object')  # Ensure consistency with mixed types

    return df


def save_curated_processed_data_data_to_csv(dataframe):
    """
    Saves processed demographics_data data DataFrame to a CSV file.
    """
    dataframe.to_csv(CURATED_PROCESSED_PATH, index=False)


def fill_suppuration(final_df):
    """
    For each exam (grouped by research_id and exam_id) in final_df:
      - If all values in exam['suppuration'] are missing (pd.NA),
          then:
              if exam['suppuration_index'] equals 0, fill exam['suppuration'] with 0;
              else, fill exam['suppuration'] with 'Missing'.
    """
    # Apply the filling logic per exam
    final_df = final_df.groupby(['research_id', 'exam_id'], group_keys=False).apply(fill_exam_suppuration)
    return final_df


if __name__ == "__main__":
    # Load the datasets
    bleeding_data = load_processed_bleeding_data()
    chart_general_data = load_processed_chart_general_data()
    chart_restorative_data = load_processed_chart_restorative_data()
    demographic_data = load_processed_demographic_data()
    furcation_data = load_processed_furcation_data()
    index_data = load_processed_index_data()
    mobility_data = load_processed_mobility_data()
    mag_data = load_processed_mag_data()
    pockets_data = load_processed_pockets_data()
    recessions_data = load_processed_recessions_data()
    suppuration_data = load_processed_suppuration_data()

    # List of datasets to combine
    variable_datasets = [
        bleeding_data,
        furcation_data,
        index_data,
        mag_data,
        mobility_data,
        pockets_data,
        recessions_data,
        suppuration_data,
    ]

    var_df = combine_variables(variable_datasets)
    middle_df = combined_general_restorative_var(chart_general_data, chart_restorative_data, dataset=var_df)
    final_df = merge_demographics_full(demographic_data, middle_df)
    final_df = fill_suppuration(final_df)

    save_processed_data_to_csv(CURATED_PROCESSED_PATH, final_df)

    # Filter to only include rows with research_id equal to 1
    final_df = final_df[final_df['research_id'].between(1, 10)]

    final_df_size = final_df.shape
    print(f"Size of final_df: {final_df_size}")

    # Save the filtered dataset to an Excel file
    save_cleaned_data_to_excel(TEMP_PATH, final_df)