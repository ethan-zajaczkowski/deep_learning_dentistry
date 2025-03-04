import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.config import CURATED_PROCESSED_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_processed_data_to_csv, \
    save_cleaned_data_to_excel


def load_dataset():
    df = pd.read_csv(CURATED_PROCESSED_PATH)
    return df

def compute_new_bleeding_index(group):
    # Calculate total teeth available for the exam:
    missing_teeth = group.loc[group['missing_status'] == 1, 'tooth_id'].nunique()
    total_teeth = 32 - missing_teeth
    total_sites = total_teeth * 6

    # Calculate the two bleeding indices using the provided functions
    index_bleeding = bleeding_index_calculator(group, total_sites)
    index_bleeding_teeth = bleeding_teeth_calculator(group, total_teeth)

    # Make a copy of the group to avoid SettingWithCopyWarning and assign the new index
    group = group.copy()
    group["new_bleeding_index"] = index_bleeding
    group["new_bleeding_teeth_percent"] = index_bleeding_teeth
    return group

def new_bleeding_index(df):
    # Apply the computation to each exam group
    df = df.groupby(["research_id", "exam_id"]).apply(compute_new_bleeding_index)

    cols = list(df.columns)
    cols.remove("new_bleeding_index")
    cols.remove("new_bleeding_teeth_percent")
    idx = cols.index("bleeding_index")
    cols.insert(idx + 1, "new_bleeding_index")
    cols.insert(idx + 1, "new_bleeding_teeth_percent")
    df = df[cols]

    return df

def bleeding_index_calculator(exam_df, total_sites):
    """
    Calculate the bleeding index of a given exam dataframe.
    """
    x = exam_df.loc[exam_df['bleeding'] == 1, 'bleeding'].count()
    return x / total_sites

def bleeding_teeth_calculator(each_exam_df, total_teeth):
    """
    Counts the % of distinct tooth_id where at least one site has `bleeding == 1` of total sites.
    """
    bleeding_teeth = (
        each_exam_df.groupby('tooth_id')['bleeding']
        .apply(lambda x: (x == 1).any())
        .sum()
    )

    return bleeding_teeth / total_teeth


def compute_new_suppuration_values(exam):
    value = exam['suppuration_index'].iloc[0]

    sum_of_suppuration = (
        exam.groupby('tooth_id')['suppuration']
             .apply(lambda x: (x == 1).any())
             .sum()
    )

    if value == 0 and sum_of_suppuration < 1:
        exam['suppuration'] = 0

    return exam


def check_exam(exam):
    # Check if all values in 'suppuration' are not missing
    if exam['suppuration'].notna().all():
        return exam['exam_id'].iloc[0]
    else:
        return pd.NA


def get_complete_suppuration_exam_ids(df):
    """
    For each exam (grouped by research_id and exam_id) in df, if all values in
    'suppuration' are not missing (pd.NA), then collect the exam_id.
    Returns:
        list: Unique exam_ids where suppuration is completely filled.
    """
    exam_ids = df.groupby(["research_id", "exam_id"]).apply(check_exam)
    return exam_ids.dropna().unique().tolist()


def create_exam_availability_df(df):
    """
    Creates a new DataFrame with one row per exam and columns:
      research_id, exam_id, exam_date, bleeding, furcation, mobility, mag, pocket, recession, suppuration

    For each exam (grouped by research_id and exam_id), if all values for a variable
    are non-missing, that variable is marked as 1; otherwise, 0.
    """
    # Define the variables to check.
    variables = ["bleeding", "furcation", "mobility", "mag", "pocket", "recession", "suppuration"]

    # Group the data by research_id and exam_id.
    grouped = df.groupby(["research_id", "exam_id"])
    records = []

    for (rid, eid), group in grouped:
        # Use the first exam_date in the group (adjust if needed)
        exam_date = group["exam_date"].iloc[0] if "exam_date" in group.columns else None

        row = {
            "research_id": rid,
            "exam_id": eid,
            "exam_date": exam_date
        }

        # For each variable, if the entire group's column for that variable is non-NA, mark 1, else 0.
        for var in variables:
            if var in group.columns:
                # If all values in the column for this exam are not missing, mark as available (1)
                available = 1 if group[var].notna().all() else 0
            else:
                available = 0  # if the variable isn't present, mark as 0.
            row[var] = available

        records.append(row)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # import os
    # BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    # NEW_PATH = os.path.join(BASE_DIR, "deep_learning_dentistry/deep_learning_analysis.csv")
    # NEW_PATH_2 = os.path.join(BASE_DIR, "deep_learning_dentistry/deep_learning_analysis_2.xlsx")

    df = load_dataset()  # loads from CURATED_PROCESSED_PATH, for instance.

    # Create the exam availability DataFrame.
    exam_availability_df = create_exam_availability_df(df)

    # Show the results.
    print(exam_availability_df)

    # Optionally, save the results to Excel:
    exam_availability_df.to_excel("exam_availability.xlsx", index=False)
