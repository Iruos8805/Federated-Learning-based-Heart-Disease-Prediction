import pandas as pd

def load_dataset(path='./heart.csv'):
    df = pd.read_csv(path)
    print("Loaded dataset from:", path)
    print("-" * 60)
    return df
