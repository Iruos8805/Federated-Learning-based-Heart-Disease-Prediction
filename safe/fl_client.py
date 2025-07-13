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
log_file = open(f"logs/fl_client_{timestamp}.txt", "w")

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

class HeartClient(fl.client.NumPyClient):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = make_pipeline(
            RBFSampler(gamma=0.028092305159489246, n_components=1288, random_state=42),
            SGDClassifier(loss='hinge', alpha=1.0 / 400.7935817191417, max_iter=1000, random_state=42)
        )
        self.model.named_steps['rbfsampler'].fit(self.X)
        self.model.named_steps['sgdclassifier'].partial_fit(
            self.model.named_steps['rbfsampler'].transform(self.X), 
            self.y, 
            classes=np.unique(self.y)
        )

    def get_parameters(self, config=None):
        """Extract model parameters for federated learning"""
        clf = self.model.named_steps['sgdclassifier']
        params = []
        
        if hasattr(clf, 'coef_') and clf.coef_ is not None:
            params.append(clf.coef_.flatten())
        if hasattr(clf, 'intercept_') and clf.intercept_ is not None:
            params.append(clf.intercept_.flatten())
            
        return params

    def set_parameters(self, parameters):
        """Set model parameters from federated learning"""
        if len(parameters) == 0:
            return
            
        clf = self.model.named_steps['sgdclassifier']
        
        if len(parameters) >= 1 and hasattr(clf, 'coef_'):
            clf.coef_ = parameters[0].reshape(clf.coef_.shape)
        
        if len(parameters) >= 2 and hasattr(clf, 'intercept_'):
            clf.intercept_ = parameters[1].reshape(clf.intercept_.shape)

    def fit(self, parameters, config):
        print("-" * 60)
        print("Fitting model on client data...")
        
        self.set_parameters(parameters)
        
        X_transformed = self.model.named_steps['rbfsampler'].transform(self.X)
        
        local_epochs = 5
        for epoch in range(local_epochs):
            print(f"Epoch {epoch + 1}/{local_epochs}")
            self.model.named_steps['sgdclassifier'].partial_fit(X_transformed, self.y)

        return self.get_parameters(), len(self.X), {}

    def evaluate(self, parameters, config):
        print("-" * 60)
        print("Evaluating model on client data...")
        
        self.set_parameters(parameters)
        
        preds = self.model.predict(self.X)
        score = recall_score(self.y, preds)
        
        print("Recall Score:", score)
        print("Predictions distribution:", np.bincount(preds))
        print("True labels distribution:", np.bincount(self.y))
        print("-" * 60)
        
        return 0.0, len(self.X), {"recall": score}

if __name__ == "__main__":
    print("-" * 60)
    print("Loading and preprocessing client data...")

    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(0, 1000)
    is_malicious = len(sys.argv) > 2 and sys.argv[2].lower() in ["mal", "malicious", "--mal"]   

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
    print(f"Random seed used: {client_id}")
    print(f"Malicious client? {'YES' if is_malicious else 'NO'}")
    print("Class distribution in this client:")
    print(np.bincount(y_part))
    print("Data shape before augmentation:", X_part.shape)
    print("-" * 60)

    if is_malicious:
        print("Malicious behavior: Partially flipping labels")
        flip_fraction = 1.0  
        flip_indices = np.random.choice(len(y_part), size=int(len(y_part) * flip_fraction), replace=False)
        y_part[flip_indices] = 1 - y_part[flip_indices]


    X_aug, y_aug = augment_client_data(X_part, y_part, target_size=2000, method="combined")

    print("Client data augmented")
    print("New shape:", X_aug.shape)
    print("New class distribution:", np.bincount(y_aug))
    print("-" * 60)

    client = HeartClient(X_aug, y_aug)

    print("Starting Federated Learning Client...")
    print("-" * 60)

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()  
    )
