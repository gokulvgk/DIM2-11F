import pandas as pd
import numpy as np


def load_and_preprocess(file_path):

    df = pd.read_excel(file_path)

    print("Original Shape:", df.shape)

    # Remove timestamp if present
    drop_cols = [col for col in df.columns if 'Timestamp' in col]

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Likert Mapping
    mapping = {
        'Strongly Agree': 5,
        'Agree': 4,
        'Neutral': 3,
        'Disagree': 2,
        'Strongly Disagree': 1
    }

    # Replace text responses
    df.replace(mapping, inplace=True)

    # Convert everything possible to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values
    df.fillna(df.mean(), inplace=True)

    print("Processed Shape:", df.shape)

    print(df.head())

    return df