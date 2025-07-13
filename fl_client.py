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
import atexit
from datetime import datetime

#---------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
client_id_from_args = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(0, 1000)
log_file = open(f"logs/fl_client_{client_id_from_args}_{timestamp}.txt", "w")

# Store original streams
original_stdout = sys.stdout
original_stderr = sys.stderr

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                if hasattr(s, 'write') and not s.closed:
                    s.write(data)
                    s.flush()
            except (ValueError, AttributeError):
                # Handle closed file errors silently
                pass

    def flush(self):
        for s in self.streams:
            try:
                if hasattr(s, 'flush') and not s.closed:
                    s.flush()
            except (ValueError, AttributeError):
                # Handle closed file errors silently
                pass

def cleanup_logging():
    """Cleanup function to restore original streams and close log file"""
    try:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file and not log_file.closed:
            log_file.close()
    except:
        pass

# Register cleanup function
atexit.register(cleanup_logging)

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)
#---------------------------------------------------------------

class HeartClient(fl.client.NumPyClient):
    def __init__(self, X, y, client_id, is_malicious=False):
        self.X = X
        self.y = y
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.current_round = 0
        self.warmup_rounds = 15  # ✅ Match server warmup rounds
        self.original_y = y.copy()  # ✅ Keep original labels
        self.attack_active = False
        
        # ✅ Track if client has been detected/blocked
        self.detection_count = 0
        self.is_blocked = False
        
        self.model = make_pipeline(
            RBFSampler(gamma=0.028092305159489246, n_components=1288, random_state=42),
            SGDClassifier(loss='hinge', alpha=1.0 / 400.7935817191417, max_iter=1000, random_state=42)
        )
        # Fit the RBF sampler once with the data
        self.model.named_steps['rbfsampler'].fit(self.X)
        # Initialize the classifier with a dummy fit to create the necessary parameters
        self.model.named_steps['sgdclassifier'].partial_fit(
            self.model.named_steps['rbfsampler'].transform(self.X), 
            self.y, 
            classes=np.unique(self.y)
        )

    def get_parameters(self, config=None):
        """Extract model parameters for federated learning"""
        clf = self.model.named_steps['sgdclassifier']
        params = []
        
        # Get the main parameters that need to be averaged
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
        
        # Set coefficients
        if len(parameters) >= 1 and hasattr(clf, 'coef_'):
            clf.coef_ = parameters[0].reshape(clf.coef_.shape)
        
        # Set intercept
        if len(parameters) >= 2 and hasattr(clf, 'intercept_'):
            clf.intercept_ = parameters[1].reshape(clf.intercept_.shape)

    def fit(self, parameters, config):
        print("-" * 60)
        print(f"Round {self.current_round}: Fitting model on client {self.client_id}")
        
        # ✅ Check if client should be blocked (simulated)
        if self.is_blocked:
            print(f"❌ Client {self.client_id} is blocked - cannot participate")
            return self.get_parameters(), len(self.X), {"client_id": str(self.client_id)}
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # ✅ Implement delayed malicious behavior
        if self.is_malicious:
            if self.current_round < self.warmup_rounds:
                # Behave normally during warmup
                if self.attack_active:
                    print(f"🔄 Client {self.client_id}: Reverting to normal behavior (warmup)")
                    self.y = self.original_y.copy()
                    self.attack_active = False
            else:
                # Start attacking after warmup
                if not self.attack_active:
                    print(f"⚠️  Client {self.client_id}: Starting malicious attack (label flipping)")
                    # Flip ALL labels for maximum impact
                    self.y = 1 - self.original_y
                    self.attack_active = True
                else:
                    print(f"⚠️  Client {self.client_id}: Continuing malicious attack")
        
        # Transform data using RBF sampler
        X_transformed = self.model.named_steps['rbfsampler'].transform(self.X)
        
        # Train for multiple local epochs
        local_epochs = 5
        for epoch in range(local_epochs):
            print(f"Epoch {epoch + 1}/{local_epochs}")
            self.model.named_steps['sgdclassifier'].partial_fit(X_transformed, self.y)

        print(f"✅ Client {self.client_id}: Training completed for round {self.current_round}")
        if self.is_malicious and self.attack_active:
            print(f"🔥 Malicious labels distribution: {np.bincount(self.y)}")
        
        self.current_round += 1
        return self.get_parameters(), len(self.X), {"client_id": str(self.client_id)}

    def evaluate(self, parameters, config):
        print("-" * 60)
        print(f"Evaluating model on client {self.client_id} data...")
        
        # ✅ Always evaluate on original clean labels
        self.set_parameters(parameters)
        preds = self.model.predict(self.X)
        score = recall_score(self.original_y, preds)  # Use original labels
        
        print(f"Client {self.client_id} Recall Score:", score)
        print("Predictions distribution:", np.bincount(preds))
        print("True labels distribution:", np.bincount(self.original_y))
        
        if self.is_malicious:
            print(f"⚠️  Malicious client - Attack active: {self.attack_active}")
        
        print("-" * 60)
        
        return 0.0, len(self.X), {"recall": score, "client_id": str(self.client_id)}

if __name__ == "__main__":
    print("-" * 60)
    print("Loading and preprocessing client data...")

    # Get client ID and malicious flag from command line
    client_id = client_id_from_args
    is_malicious = len(sys.argv) > 2 and sys.argv[2].lower() in ["mal", "malicious", "--mal"]   

    df = load_dataset()
    df = basic_cleaning(df)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X, y = remove_outliers(X, y)
    X = feature_engineering(X)
    X = scale_features(X)
    X = select_features(X, y)
    
    # Stratified split to preserve class balance per client - different seed per client
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=client_id)
    for train_idx, _ in splitter.split(X, y):
        X_part = X.iloc[train_idx].values
        y_part = y.iloc[train_idx].values

    print(f"Client ID: {client_id}")
    print(f"Random seed used: {client_id}")
    print(f"Malicious client? {'✅ YES' if is_malicious else '❌ NO'}")
    print("Class distribution in this client:")
    print(np.bincount(y_part))
    print("Data shape before augmentation:", X_part.shape)
    
    # ✅ Remove immediate label flipping - will be handled in fit() method
    # if is_malicious:
    #     print("⚠️  Malicious behavior: Partially flipping labels")
    #     ...

    print("-" * 60)

    # ✅ Augment the client data before training
    X_aug, y_aug = augment_client_data(X_part, y_part, target_size=2000, method="combined")

    print("✅ Client data augmented")
    print("New shape:", X_aug.shape)
    print("New class distribution:", np.bincount(y_aug))
    print("-" * 60)

    client = HeartClient(X_aug, y_aug, client_id, is_malicious)

    print(f"Starting Federated Learning Client {client_id}...")
    if is_malicious:
        print("⚠️  Malicious behavior will activate after warmup rounds")
    print("-" * 60)

    try:
        fl.client.start_client(
            server_address="localhost:8080",
            client=client.to_client()
        )
    except KeyboardInterrupt:
        print(f"\n🛑 Client {client_id} shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Client {client_id} error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"📝 Client {client_id} logs saved to logs/ directory")
        cleanup_logging()
