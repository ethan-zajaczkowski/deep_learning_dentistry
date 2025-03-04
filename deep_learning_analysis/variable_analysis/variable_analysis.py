import pandas as pd
import itertools
from deep_learning_dentistry.data_curation.data_processing.utils.config import BLEEDING_CLEANED_PATH, \
    MOBILITY_CLEANED_PATH, POCKETS_CLEANED_PATH, RECESSIONS_CLEANED_PATH, MAG_CLEANED_PATH, FURCATION_CLEANED_PATH, \
    CURATED_PROCESSED_PATH
from deep_learning_dentistry.data_curation.index_customization import get_complete_suppuration_exam_ids


def load_dataset(cleaned_dataset_path):
    """
    Import the specified cleaned dataset.
    """
    df = pd.read_excel(cleaned_dataset_path)
    return df


def load_suppuration():
    """
    Handle suppuration exams
    """
    import os
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    NEW_PATH = os.path.join(BASE_DIR, "deep_learning_analysis.csv")
    list = pd.read_csv(NEW_PATH)
    return list

def unique_research_ids_count(dataframe):
    """
    Count the number of unique research_ids in this dataset.
    """
    unique_count = dataframe['ResearchID'].nunique()
    return unique_count


def unique_exam_ids_count(dataframe):
    """
    Count the number of unique exam_ids in this dataset.
    """
    unique_count = dataframe['CHART ID'].nunique()
    return unique_count


def list_of_exam_ids(dataframe):
    """
    Return a list of the exam_ids in this dataset.
    """
    exam_ids = dataframe['CHART ID'].unique().tolist()
    return exam_ids


def row_count(df):
    """
    Return the number of rows in the given DataFrame.
    """
    return len(df)


def find_non_unique_values(df):
    """
    Find and return a list of distinct values in 'column_name'
    that appear more than once in the DataFrame.
    """
    counts = df['CHART ID'].value_counts()
    non_unique = counts[counts > 1].index.tolist()

    return non_unique


def compute_exam_id_combinations():
    """
    Given a dictionary mapping variable names to their exam ID sets,
    compute the intersection for every combination of 2 to all sets.

    Returns:
        A DataFrame with columns:
         - "Combination": Tuple of variable names.
         - "Size of Combination": How many sets are in the combination.
         - "Common Exam IDs Count": The number of exam IDs common to all sets in the combination.
    """
    records = []

    # Load the datasets
    bleeding_data = load_dataset(BLEEDING_CLEANED_PATH)
    mag_data = load_dataset(MAG_CLEANED_PATH)
    mobility_data = load_dataset(MOBILITY_CLEANED_PATH)
    furcation_data = load_dataset(FURCATION_CLEANED_PATH)
    pockets_data = load_dataset(POCKETS_CLEANED_PATH)
    recessions_data = load_dataset(RECESSIONS_CLEANED_PATH)
    suppuration_data = load_suppuration()

    # Extract exam IDs from each dataset
    bleeding_list = list_of_exam_ids(bleeding_data)
    mag_list = list_of_exam_ids(mag_data)
    mobility_list = list_of_exam_ids(mobility_data)
    furcation_list = list_of_exam_ids(furcation_data)
    pockets_list = list_of_exam_ids(pockets_data)
    recessions_list = list_of_exam_ids(recessions_data)
    suppuration_list = get_complete_suppuration_exam_ids(suppuration_data)

    # Convert lists to sets
    bleeding_set = set(bleeding_list)
    furcation_set = set(furcation_list)
    mag_set = set(mag_list)
    mobility_set = set(mobility_list)
    pockets_set = set(pockets_list)
    recessions_set = set(recessions_list)
    suppuration_set = set(suppuration_list)

    # Create a dictionary mapping variable names to their exam ID sets.
    exam_sets = {
        "bleeding": bleeding_set,
        "furcation": furcation_set,
        "mag": mag_set,
        "mobility": mobility_set,
        "pockets": pockets_set,
        "recessions": recessions_set,
        "suppuration": suppuration_set
    }

    # Generate combinations of sizes 2 up to len(exam_sets)
    for r in range(2, len(exam_sets) + 1):
        for combo in itertools.combinations(exam_sets.keys(), r):
            # Retrieve the corresponding sets for this combination.
            sets_in_combo = [exam_sets[name] for name in combo]
            # Compute the intersection.
            common_ids = set.intersection(*sets_in_combo)
            records.append({
                "Combination": combo,
                "Size of Combination": r,
                "Common Exam IDs Count": len(common_ids)
            })

    # Create and sort the DataFrame
    df_combos = pd.DataFrame(records)
    df_combos_sorted = df_combos.sort_values(by="Common Exam IDs Count", ascending=False)

    output_path = "exam_id_combinations.xlsx"
    df_combos_sorted.to_excel(output_path, index=False)
    print(f"Exam ID combinations saved to {output_path}")

    return df_combos_sorted


def extract_common_exam_ids():
    # Load the datasets
    bleeding_data = load_dataset(BLEEDING_CLEANED_PATH)
    mag_data = load_dataset(MAG_CLEANED_PATH)
    mobility_data = load_dataset(MOBILITY_CLEANED_PATH)
    furcation_data = load_dataset(FURCATION_CLEANED_PATH)
    pockets_data = load_dataset(POCKETS_CLEANED_PATH)
    recessions_data = load_dataset(RECESSIONS_CLEANED_PATH)
    suppuration_data = load_suppuration()

    # Extract exam IDs from each dataset
    bleeding_list = list_of_exam_ids(bleeding_data)
    mag_list = list_of_exam_ids(mag_data)
    mobility_list = list_of_exam_ids(mobility_data)
    furcation_list = list_of_exam_ids(furcation_data)
    pockets_list = list_of_exam_ids(pockets_data)
    recessions_list = list_of_exam_ids(recessions_data)
    suppuration_list = get_complete_suppuration_exam_ids(suppuration_data)

    # Compute the intersection (common exam IDs) among these five sets
    common_exam_ids = set(bleeding_list) & set(furcation_list) & set(pockets_list) & set(recessions_list) & set(
        suppuration_list) & set(mag_list) & set(mobility_list)

    return common_exam_ids


def filter_curated_data(common_exam_ids, curated_path):
    """
    Load the curated processed data and filter it to only include rows whose exam ID
    is in common_exam_ids.
    """
    curated_df = pd.read_csv(CURATED_PROCESSED_PATH)
    # Adjust the column name if needed; here we assume the exam id column is named 'exam_id'
    filtered_df = curated_df[curated_df['exam_id'].isin(common_exam_ids)]

    output_path = "filtered_curated_data_complete.csv"
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered curated data saved to {output_path}")

    return filtered_df


if __name__ == "__main__":
    pass