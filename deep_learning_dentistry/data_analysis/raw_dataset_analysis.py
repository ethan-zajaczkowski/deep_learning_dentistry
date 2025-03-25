import pandas as pd
from deep_learning_dentistry.data_curation.data_processing.utils.config import FULL_DATASET_PATH
from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import load_curated_dataset


def load_dataset():
    df = load_curated_dataset(FULL_DATASET_PATH, "full_dataset_long.csv")
    return df

def count_unique_research_ids(df):
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

def analysis():
    df = load_dataset()
    print(count_unique_research_ids(df))
    print(count_unique_exam_ids(df))

if __name__ == '__main__':
    analysis()
