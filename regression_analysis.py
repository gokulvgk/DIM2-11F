from sklearn.linear_model import LinearRegression
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
    'Curriculum_Inflexibility',
    'Market_Driven_Policies'
]

def run_regression_analysis(df):

    X = df[FACTOR_COLUMNS]

    y = df['Enrollment_Decline']

    model = LinearRegression()

    model.fit(X, y)

    coefficients = pd.DataFrame({
        'Factor': FACTOR_COLUMNS,
        'Coefficient': model.coef_
    })

    coefficients.sort_values(
        by='Coefficient',
        ascending=False,
        inplace=True
    )

    coefficients.to_excel(
        'outputs/tables/regression_coefficients.xlsx',
        index=False
    )

    print(coefficients)
