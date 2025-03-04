from deep_learning_dentistry.data_curation.data_processing.utils.data_loader import load_chart_general
from deep_learning_dentistry.data_curation.data_processing.utils.functions import save_cleaned_data_to_excel, \
    save_processed_data_to_csv, load_processed_dataset, load_cleaned_dataset
from deep_learning_dentistry.data_curation.data_processing.utils.config import CHART_GENERAL_CLEANED_PATH, CHART_GENERAL_PROCESSED_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_transformer import transform_dataset, transform_dataset_to_clean_chart_general


def curate_chart_general_data():
    """
    Loads chart_general data from the respective sources.
    ** SAVE RESULTING DF TO "CLEANED" FOLDER **
    """
    chart_general_complete = load_chart_general()
    chart_general_updated = transform_dataset_to_clean_chart_general(chart_general_complete)
    return chart_general_updated


def transform_chart_general_data():
    """
    Curate and transform the dataframe, have it ready for use.
    """
    chart_general_complete = curate_chart_general_data()
    chart_general_transformed = transform_dataset(chart_general_complete, variable="chart_general")
    return chart_general_transformed


def load_processed_chart_general_data():
    """
    Loads the processed chart general dataset from the specified file path.
    """
    return load_processed_dataset(CHART_GENERAL_PROCESSED_PATH)


def save_cleaned_chart_general_data_to_excel(dataframe):
    """
    Saves cleaned chart_general DataFrame to an Excel file.
    """
    save_cleaned_data_to_excel(CHART_GENERAL_CLEANED_PATH, dataframe)


def save_processed_chart_general_data_to_csv(dataframe):
    """
    Saves processed chart_general DataFrame to a CSV file.
    """
    save_processed_data_to_csv(CHART_GENERAL_PROCESSED_PATH, dataframe)


def curate_clean_transform_process_save_chart_general():
    """
    Curates raw index data, saves it to cleaned file, transforms it, then saves the transformed dataset to processed.
    """
    index_complete = curate_chart_general_data()
    save_cleaned_chart_general_data_to_excel(index_complete)
    transformed_index = transform_dataset(index_complete, variable="chart_general")
    post_processed_data = post_process_chart_general(transformed_index)
    save_processed_chart_general_data_to_csv(post_processed_data)
    return transformed_index


def post_process_chart_general(df):
    """
    - Ensure that if a tooth was indicated as missing in the past, that this information is always propagated forward across all future exams (inside of the chart_general dataset). This is necessary because there exists some patients that had a tooth reported as missing in the past but not in the future (outside of the chart_restoration analysis dataset).
    """
    df["missing_status"] = df["missing_status"].astype(int)
    df["missing_status"] = df.groupby(["research_id", "tooth_quadrant", "tooth_id"])["missing_status"].cummax()

    return df


if __name__ == "__main__":
    index_df = curate_clean_transform_process_save_chart_general()