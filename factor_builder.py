def build_factors(df):

    # Factor 1
    df['Math_Anxiety'] = df.iloc[:, 0:4].mean(axis=1)

    # Factor 2
    df['Curriculum_Complexity'] = df.iloc[:, 4:8].mean(axis=1)

    # Factor 3
    df['Teaching_Methodology'] = df.iloc[:, 8:12].mean(axis=1)

    # Factor 4
    df['School_Foundation'] = df.iloc[:, 12:15].mean(axis=1)

    # Factor 5
    df['Career_Awareness'] = df.iloc[:, 15:18].mean(axis=1)

    # Factor 6
    df['Employability_Perception'] = df.iloc[:, 18:21].mean(axis=1)

    # Factor 7
    df['Skill_Based_Preference'] = df.iloc[:, 21:24].mean(axis=1)

    # Factor 8
    df['Academic_Pressure'] = df.iloc[:, 24:27].mean(axis=1)

    # Factor 9
    df['Enrollment_Decline'] = df.iloc[:, 27]

    # Factor 10
    df['Curriculum_Inflexibility'] = df.iloc[:, 28]

    # Factor 11
    df['Market_Driven_Policies'] = (
        df['Enrollment_Decline'] +
        df['Curriculum_Inflexibility']
    ) / 2

    return df
