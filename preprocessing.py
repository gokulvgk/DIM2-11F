import pandas as pd
import numpy as np


def load_and_preprocess(file_path):

    df = pd.read_excel(file_path)

    print("Original Shape:", df.shape)

    # Remove timestamp columns if present
    drop_cols = [
        col for col in df.columns
        if 'Timestamp' in col
    ]

    df.drop(
        columns=drop_cols,
        inplace=True,
        errors='ignore'
    )

    # Preserve Gender column
    gender_col = None

    if 'Gender' in df.columns:
        gender_col = df['Gender'].copy()

    # Likert Scale Mapping
    mapping = {
        'Strongly Agree': 5,
        'Agree': 4,
        'Neutral': 3,
        'Disagree': 2,
        'Strongly Disagree': 1
    }

    df.replace(mapping, inplace=True)

    # Convert survey responses to numeric
    df = df.apply(
        pd.to_numeric,
        errors='coerce'
    )

    # Restore Gender column
    if gender_col is not None:
        df['Gender'] = gender_col

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].mean()
    )

    print("Processed Shape:", df.shape)

    return df