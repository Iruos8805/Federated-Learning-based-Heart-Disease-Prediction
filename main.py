import argparse
from dataset import load_dataset
from preprocessing import basic_cleaning
from eda import run_eda
from modifications import remove_outliers, feature_engineering, scale_features, select_features, correlation_analysis
from optuna_tune import run_optuna
from train import train_model
from test import evaluate_model
from sklearn.model_selection import train_test_split
import sys
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open(f"logs/run_{timestamp}.txt", "w")

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default=None, help='Choose mode: "optuna" for hyperparameter tuning, "fl" for federated learning')
    args = parser.parse_args()

    use_optuna = args.mode == 'optuna'

    df = load_dataset()
    df = basic_cleaning(df)

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    run_eda(X, y)
    X, y = remove_outliers(X, y)
    X = feature_engineering(X)
    X = scale_features(X)
    X = select_features(X, y)
    correlation_analysis(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if use_optuna:
        run_optuna(X_train, y_train)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
