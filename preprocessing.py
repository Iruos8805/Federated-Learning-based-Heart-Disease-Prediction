import matplotlib.pyplot as plt
import seaborn as sns
import os

def basic_cleaning(df, out_dir='./EDA'):
    print(df.info())
    print("Missing values:\n", df.isnull().sum())
    print("Has Duplicates:", df.duplicated().any())
    print("-" * 60)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df.drop('HeartDisease', axis=1))
    plt.xticks(rotation=45)
    plt.title("Boxplot of Features")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/boxplot.png")
    plt.close()

    print("Saved boxplot to", f"{out_dir}/boxplot.png")
    print("-" * 60)
    return df
