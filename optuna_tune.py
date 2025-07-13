import optuna
import json
import os
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler

def run_optuna(X_train, y_train, save_path='config/best_params.json'):
    def objective(trial):
        C = trial.suggest_uniform('C', 200, 500)
        gamma = trial.suggest_loguniform('gamma', 0.001, 0.2)
        n_components = trial.suggest_int('n_components', 1000, 1500)
        rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
        model = make_pipeline(rbf, SGDClassifier(loss='hinge', alpha=1.0/C, max_iter=1000, tol=1e-3))
        return cross_val_score(model, X_train, y_train, cv=5, scoring='recall').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    best_score = study.best_value

    print("Best Parameters:", best_params)
    print(f"Best Cross-Validated Recall: {best_score:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'a') as f:
        json.dump({"params": best_params, "recall": best_score}, f)
        f.write("\n")

    print("Appended best parameters to", save_path)
    print("-" * 60)
