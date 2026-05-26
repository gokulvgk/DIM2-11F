from preprocessing import load_and_preprocess
from factor_builder import build_factors
from statistics_analysis import descriptive_statistics, correlation_analysis
from dim2_model import apply_dim2_model
from regression_analysis import run_regression_analysis
from visualization import generate_all_graphs


def main():
    print("Loading and preprocessing dataset...")
    df = load_and_preprocess("data/survey.xlsx")

    print("Building 11 factors...")
    df = build_factors(df)

    print("Running statistical analysis...")
    descriptive_statistics(df)
    correlation_analysis(df)

    print("Applying DIM²–11F model...")
    df = apply_dim2_model(df)

    print("Running regression analysis...")
    run_regression_analysis(df)

    print("Generating graphs...")
    generate_all_graphs(df)

    print("Analysis Completed Successfully")


if __name__ == "__main__":
    main()
