import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


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


def generate_all_graphs(df):

    # =====================================================
    # 1. Correlation Heatmap
    # =====================================================

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        df[FACTOR_COLUMNS].corr(),
        annot=True,
        cmap='coolwarm'
    )

    plt.title('Correlation Heatmap')

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/correlation_heatmap.png'
    )

    plt.close()

    # =====================================================
    # 2. Q-Q Plot
    # =====================================================

    plt.figure(figsize=(8, 6))

    stats.probplot(
        df['Math_Anxiety'],
        dist="norm",
        plot=plt
    )

    plt.title(
        'Q-Q Plot for Mathematics Anxiety'
    )

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/qq_plot_math_anxiety.png'
    )

    plt.close()

    # =====================================================
    # 3. 3D Scatter Plot
    # =====================================================

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    ax.scatter(
        df['Math_Anxiety'],
        df['Academic_Pressure'],
        df['Enrollment_Decline']
    )

    ax.set_xlabel('Math Anxiety')
    ax.set_ylabel('Academic Pressure')
    ax.set_zlabel('Enrollment Decline')

    plt.title('3D Factor Interaction')

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/3d_scatter_plot.png'
    )

    plt.close()

    # =====================================================
    # 4. KDE Density Plot
    # =====================================================

    plt.figure(figsize=(8, 6))

    sns.kdeplot(
        df['Entropy'],
        fill=True
    )

    plt.title('Entropy Density Distribution')

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/entropy_density_plot.png'
    )

    plt.close()

    # =====================================================
    # 5. Pair Plot
    # =====================================================

    pairplot = sns.pairplot(
        df[
            [
                'Math_Anxiety',
                'Academic_Pressure',
                'Enrollment_Decline'
            ]
        ]
    )

    pairplot.fig.suptitle(
        'Pair Plot of Key Factors',
        y=1.02
    )

    pairplot.savefig(
        'outputs/graphs/pairplot.png'
    )

    plt.close()

    # =====================================================
    # 6. PCA Visualization
    # =====================================================

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df[FACTOR_COLUMNS]
    )

    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(
        scaled_data
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        principal_components[:, 0],
        principal_components[:, 1]
    )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.title('PCA Visualization')

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/pca_visualization.png'
    )

    plt.close()

    # =====================================================
    # 7. Regression Plot
    # =====================================================

    plt.figure(figsize=(8, 6))

    sns.regplot(
        x=df['Math_Anxiety'],
        y=df['Enrollment_Decline']
    )

    plt.title(
        'Regression: Anxiety vs Enrollment Decline'
    )

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/regression_plot.png'
    )

    plt.close()

    # =====================================================
    # 8. Histogram
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.hist(
        df['DIM2_Output'],
        bins=20
    )

    plt.title('DIM² Output Distribution')

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/dim2_histogram.png'
    )

    plt.close()

    # =====================================================
    # 9. Boxplot
    # =====================================================

    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df[FACTOR_COLUMNS]
    )

    plt.xticks(rotation=45)

    plt.title('Factor Distribution Boxplot')

    plt.tight_layout()

    plt.savefig(
        'outputs/graphs/factor_boxplot.png'
    )

    plt.close()

    print("All Advanced Visualizations Generated")