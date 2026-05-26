import numpy as np


FACTOR_COLUMNS = [
    'Math_Anxiety',
    'Curriculum_Complexity',
    'Teaching_Methodology',
    'School_Foundation',
    'Career_Awareness',
    'Employability_Perception',
    'Skill_Based_Preference',
    'Academic_Pressure',
    'Enrollment_Decline',
    'Curriculum_Inflexibility',
    'Market_Driven_Policies'
]



def entropy(row):
    values = row.values

    values = np.abs(values)

    total = np.sum(values)

    if total == 0:
        return 0

    probabilities = values / total

    probabilities = probabilities[probabilities > 0]

    return -np.sum(probabilities * np.log(probabilities))



def apply_dim2_model(df):

    # Equal weights
    weights = np.ones(len(FACTOR_COLUMNS)) / len(FACTOR_COLUMNS)

    # Initial Interest Score
    df['I0'] = np.dot(df[FACTOR_COLUMNS], weights)

    # Entropy Calculation
    df['Entropy'] = df[FACTOR_COLUMNS].apply(entropy, axis=1)

    # Lambda parameter
    lambda_param = 0.5

    # Adjusted Interest
    df['Adjusted_Interest'] = df['I0'] - (
        lambda_param * df['Entropy']
    )

    # Institutional Pressure
    df['Institutional_Pressure'] = (
        df['Enrollment_Decline'] +
        df['Curriculum_Inflexibility'] +
        df['Market_Driven_Policies']
    ) / 3

    # Final DIM² Output
    df['DIM2_Output'] = (
        df['Adjusted_Interest'] -
        df['Institutional_Pressure']
    )

    df.to_excel('outputs/processed_data/dim2_results.xlsx')

    return df
