import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(X, y, out_dir='./EDA'):
    os.makedirs(out_dir, exist_ok=True)

    y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, labels=['No Disease', 'Heart Disease'])
    plt.title("Heart Disease Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/target_distribution.png")
    plt.close()

    X['Age'].plot.hist(bins=20, edgecolor='black')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/age_histogram.png")
    plt.close()

    sns.histplot(X['Cholesterol'], kde=True, bins=30)
    plt.title("Cholesterol Distribution with KDE")
    plt.xlabel("Cholesterol")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cholesterol_kde.png")
    plt.close()

    sns.stripplot(data=X, x='Sex', y='Age', jitter=True)
    plt.title("Age Distribution by Sex")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/age_by_sex.png")
    plt.close()

    sns.scatterplot(data=X, x='Age', y='Cholesterol', hue=y)
    plt.title("Age vs Cholesterol (colored by HeartDisease)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/age_vs_cholesterol.png")
    plt.close()

    print("Saved EDA plots to", out_dir)
    print("-" * 60)
