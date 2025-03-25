from deep_learning_dentistry.data_curation.data_processing.utils.config import DEMOGRAPHICS_CURATED_PATH_EXCEL, \
    DEMOGRAPHICS_CURATED_PATH_CSV
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import merge_df_normal, \
    merge_demographics_data, save_curated_data_to_excel, save_curated_data_to_csv, \
    load_curated_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_demographic_data


def curate_demographics_data():
    """ Load all demographic data and curate it. """
    df_list = load_demographic_data()
    demographics_data_raw = merge_demographics_data(df_list)
    return demographics_data_raw

def save_curated_demographics_data_to_excel(demographics_data):
    """ Saves curated demographics data to an Excel file. """
    save_curated_data_to_excel(DEMOGRAPHICS_CURATED_PATH_EXCEL, demographics_data)

def save_curated_demographics_data_to_csv(demographics_data):
    """ Saves curated demographics data as a CSV file. """
    save_curated_data_to_csv(DEMOGRAPHICS_CURATED_PATH_CSV, demographics_data)

def load_curated_demographics_data_from_csv():
    """ Load curated demographics data. """
    return load_curated_dataset(DEMOGRAPHICS_CURATED_PATH_CSV)

def transform_demographics_restore_data(demographics_data):
    """Final transformation for demographics data."""
    col_map = {
        "ResearchID": "research_id",
        "Gender": "gender",
        "DateOfBirth": "date_of_birth",
        "City": "city",
        "PostalCode": "postal_code",
        "State": "state",
        "Country": "country",
        "Education Level": "education_level",
        "Occupation": "occupation",
        "TobaccoUser": "tobacco_user",
        "Periodontal disease risk": "periodontal_disease_risk",
        "Past Periodontal Treatmen": "past_periodontal_treatment"
    }
    df = demographics_data.rename(columns=col_map)
    desired_order = [
        "research_id",
        "gender",
        "date_of_birth",
        "city",
        "postal_code",
        "state",
        "country",
        "education_level",
        "occupation",
        "tobacco_user",
        "periodontal_disease_risk",
        "past_periodontal_treatment"
    ]
    final_columns = [col for col in desired_order if col in df.columns]
    df = df[final_columns]

    return df

def load_curate_save_demographics_data():
    """ Curates raw chart_general data and saves it. """
    demographics_data_curated = curate_demographics_data()
    demographics_data_transformed = transform_demographics_restore_data(demographics_data_curated)
    save_curated_demographics_data_to_excel(demographics_data_transformed)
    save_curated_demographics_data_to_csv(demographics_data_transformed)


if __name__ == "__main__":
    load_curate_save_demographics_data()