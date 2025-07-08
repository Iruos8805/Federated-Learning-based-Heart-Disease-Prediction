import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import os

def remove_outliers(X, y):
    exclude_cols = ['FastingBS']
    X_numeric = X.select_dtypes(include=['int64', 'float64']).drop(columns=exclude_cols)
    Q1 = X_numeric.quantile(0.25)
    Q3 = X_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (X_numeric < (Q1 - 1.5 * IQR)) | (X_numeric > (Q3 + 1.5 * IQR))
    outlier_rows = outlier_mask.any(axis=1)
    removed = outlier_rows.sum()
    print(f"Removed {removed} outlier rows")
    print("-" * 60)
    return X[~outlier_rows].copy(), y[~outlier_rows].copy()

def feature_engineering(X):
    categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    X = pd.get_dummies(X, columns=categorical)
    X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 40, 55, 70, 100], labels=['Young', 'Middle-aged', 'Senior', 'Very old'])
    X = pd.get_dummies(X, columns=['AgeGroup'], drop_first=True)
    print("Performed feature engineering")
    print("-" * 60)
    return X

def scale_features(X):
    columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = preprocessing.StandardScaler()
    X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    print("Scaled numerical features")
    print("-" * 60)
    return X

def select_features(X, y):
    correlations = X.corrwith(y).abs()
    selected_features = correlations[correlations >= 0.05].index.tolist()
    print(f"Selected {len(selected_features)} features: {selected_features}")
    print("-" * 60)
    return X[selected_features]

def correlation_analysis(X, out_dir='./EDA'):
    os.makedirs(out_dir, exist_ok=True)
    corr_matrix = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png")
    plt.close()
    print("Saved correlation heatmap to", f"{out_dir}/correlation_heatmap.png")
    print("-" * 60)
    return corr_matrix