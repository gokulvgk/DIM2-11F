import pandas as pd


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


def descriptive_statistics(df):
    stats = df[FACTOR_COLUMNS].describe().T

    stats.to_excel('outputs/tables/descriptive_statistics.xlsx')

    print("Descriptive Statistics Saved")



def correlation_analysis(df):
    corr = df[FACTOR_COLUMNS].corr()

    corr.to_excel('outputs/tables/correlation_matrix.xlsx')

    print("Correlation Matrix Saved")
