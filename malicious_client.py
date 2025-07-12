import flwr as fl
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from dataset import load_dataset
from preprocessing import basic_cleaning
from modifications import remove_outliers, feature_engineering, scale_features, select_features
from client_augment import augment_client_data

import os
import sys
from datetime import datetime

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open(f"logs/malicious_client_{timestamp}.txt", "w")

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data): [s.write(data) or s.flush() for s in self.streams]
    def flush(self): [s.flush() for s in self.streams]

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ----------------- Client --------------------

class DelayedMaliciousClient(fl.client.NumPyClient):
    def __init__(self, X, y, become_malicious_after=5):
        self.X = X
        self.y_clean = y.copy()
        self.y_malicious = y.copy()
        self.become_malicious_after = become_malicious_after
        self.round_counter = 0

        # Flip 100% of labels in y_malicious
        flip_fraction = 1.0
        flip_indices = np.random.choice(len(self.y_malicious), size=int(len(self.y_malicious) * flip_fraction), replace=False)
        self.y_malicious[flip_indices] = 1 - self.y_malicious[flip_indices]

        self.model = make_pipeline(
            RBFSampler(gamma=0.028092305159489246, n_components=1288, random_state=42),
            SGDClassifier(loss='hinge', alpha=1.0 / 400.7935817191417, max_iter=1000, random_state=42)
        )
        self.model.named_steps['rbfsampler'].fit(self.X)
        self.model.named_steps['sgdclassifier'].partial_fit(
            self.model.named_steps['rbfsampler'].transform(self.X),
            self.y_clean,  # Start clean
            classes=np.unique(self.y_clean)
        )

    def get_parameters(self, config=None):
        clf = self.model.named_steps['sgdclassifier']
        params = []
        if hasattr(clf, 'coef_'): params.append(clf.coef_.flatten())
        if hasattr(clf, 'intercept_'): params.append(clf.intercept_.flatten())
        return params

    def set_parameters(self, parameters):
        clf = self.model.named_steps['sgdclassifier']
        if len(parameters) >= 1 and hasattr(clf, 'coef_'):
            clf.coef_ = parameters[0].reshape(clf.coef_.shape)
        if len(parameters) >= 2 and hasattr(clf, 'intercept_'):
            clf.intercept_ = parameters[1].reshape(clf.intercept_.shape)

    def fit(self, parameters, config):
        self.round_counter += 1
        self.set_parameters(parameters)

        X_transformed = self.model.named_steps['rbfsampler'].transform(self.X)

        if self.round_counter >= self.become_malicious_after:
            print(f"🚨 Round {self.round_counter}: Acting MALICIOUS (flipped labels)")
            y_train = self.y_malicious
        else:
            print(f"✅ Round {self.round_counter}: Acting NORMAL")
            y_train = self.y_clean

        for epoch in range(5):
            self.model.named_steps['sgdclassifier'].partial_fit(X_transformed, y_train)

        return self.get_parameters(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.model.predict(self.X)
        recall = recall_score(self.y_clean, preds)
        print(f"📊 Evaluate | Recall: {recall:.4f}")
        return 0.0, len(self.X), {"recall": recall}


# ----------------- Main -----------------------

if __name__ == "__main__":
    print("🧠 Loading and preprocessing malicious client data...")

    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(0, 1000)
    become_malicious_after = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    df = load_dataset()
    df = basic_cleaning(df)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X, y = remove_outliers(X, y)
    X = feature_engineering(X)
    X = scale_features(X)
    X = select_features(X, y)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=client_id)
    for train_idx, _ in splitter.split(X, y):
        X_part = X.iloc[train_idx].values
        y_part = y.iloc[train_idx].values

    print(f"Client ID: {client_id}")
    print(f"Will act malicious after round: {become_malicious_after}")
    print("Class distribution:", np.bincount(y_part))

    X_aug, y_aug = augment_client_data(X_part, y_part, target_size=2000, method="combined")

    client = DelayedMaliciousClient(X_aug, y_aug, become_malicious_after=become_malicious_after)

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )
