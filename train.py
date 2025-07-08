# 📁 modules/train.py
import json
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, config_path='config/best_params.json', best_params=None):
    if best_params is None:
        best_score = -1
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    entry = json.loads(line)
                    if "params" in entry and "recall" in entry:
                        if entry["recall"] > best_score:
                            best_score = entry["recall"]
                            best_params = entry["params"]
                except json.JSONDecodeError:
                    continue
        if best_params is None:
            raise ValueError("No valid best_params entry with recall found in the config file.")

    print("Training with parameters:", best_params)
    print("-" * 60)

    rbf = RBFSampler(gamma=best_params['gamma'], n_components=best_params['n_components'], random_state=42)
    model = make_pipeline(rbf, SGDClassifier(loss='hinge', alpha=1.0 / best_params['C'], max_iter=1000, tol=1e-3))
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    print("Cross-validation scores:", scores)
    print("Mean CV Recall:", scores.mean())
    print("-" * 60)
    model.fit(X_train, y_train)
    return model