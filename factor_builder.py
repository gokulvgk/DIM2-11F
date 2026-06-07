import pandas as pd

def build_factors(df):

    # Actual survey questions are columns 6 through 34
    question_cols = df.columns[6:35]

    q = df[question_cols]

    # Factor 1: Q1-Q4
    df['Math_Anxiety'] = q.iloc[:, 0:4].mean(axis=1)

    # Factor 2: Q5-Q8
    df['Curriculum_Complexity'] = q.iloc[:, 4:8].mean(axis=1)

    # Factor 3: Q9-Q12
    df['Teaching_Methodology'] = q.iloc[:, 8:12].mean(axis=1)

    # Factor 4: Q13-Q15
    df['School_Foundation'] = q.iloc[:, 12:15].mean(axis=1)

    # Factor 5: Q16-Q18
    df['Career_Awareness'] = q.iloc[:, 15:18].mean(axis=1)

    # Factor 6: Q19-Q21
    df['Employability_Perception'] = q.iloc[:, 18:21].mean(axis=1)

    # Factor 7: Q22-Q24
    df['Skill_Based_Preference'] = q.iloc[:, 21:24].mean(axis=1)

    # Factor 8
    df['Academic_Pressure'] = q.iloc[:, 24:27].mean(axis=1)

    # Factor 9
    df['Enrollment_Decline'] = q.iloc[:, 27]

    # Factor 10
    df['Curriculum_Inflexibility'] = q.iloc[:, 28]

    # Factor 11
    df['Market_Driven_Policies'] = (
        q.iloc[:, 25:29].mean(axis=1)
    )

    return df