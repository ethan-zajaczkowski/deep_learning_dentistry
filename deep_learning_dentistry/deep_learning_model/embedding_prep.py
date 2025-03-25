

def apply_one_hot_encoding_on_demographics(merged_df):
    """ Performs one-hot encoding on the demographics data for specific columns:
    - gender: M = 0, F = 1
    - active: FALSE = 0, TRUE = 1
    - periodontal_disease_risk: Creates four columns (missing, high, moderate, low)
    - past_periodontal_treatment: Creates three columns (missing, true, false)
    """
    # Copy the DataFrame to avoid modifying the original
    df = merged_df.copy()

    # One-hot encode gender
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})

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