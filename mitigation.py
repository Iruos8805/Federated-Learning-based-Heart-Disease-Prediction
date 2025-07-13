import numpy as np
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from sklearn.model_selection import StratifiedShuffleSplit
from dataset import load_dataset
from preprocessing import basic_cleaning
from modifications import remove_outliers, feature_engineering, scale_features, select_features
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("logs", exist_ok=True)

class GradientSignatureVerifier:
    def __init__(self, signature_window=10, anomaly_threshold=1.8, profile_decay=0.95):
        self.signature_window = signature_window
        self.anomaly_threshold = anomaly_threshold
        self.profile_decay = profile_decay

        self.client_profiles = {}
        self.round_count = 0
        self.adaptive_threshold = anomaly_threshold
        self.false_positive_count = 0
        self.attack_detection_count = 0
        self.total_clients_processed = 0
        self.total_updates_filtered = 0
        
        # ✅ NEW: Track blocked clients and their detection history
        self.blocked_clients = set()
        self.detection_history = {}  # client_id -> list of detection rounds
        self.client_round_mapping = {}  # Maps client proxy to consistent client_id

        self.score_log_path = "logs/client_scores.csv"
        with open(self.score_log_path, "w") as f:
            f.write("round,client_id,score,filtered,blocked\n")

        print("\U0001F50D GSV initialized | window:", signature_window, "| threshold:", anomaly_threshold)

    def compute_gradient_signature(self, parameters):
        if len(parameters) == 0:
            return np.array([])

        try:
            flattened_params = np.concatenate([p.flatten() for p in parameters])
            param_magnitudes = np.abs(flattened_params)

            magnitude_stats = [
                np.mean(param_magnitudes),
                np.std(param_magnitudes),
                np.median(param_magnitudes),
                np.percentile(param_magnitudes, 75) - np.percentile(param_magnitudes, 25),
                np.max(param_magnitudes),
                np.min(param_magnitudes)
            ]

            positive_ratio = np.sum(flattened_params > 0) / len(flattened_params)
            negative_ratio = np.sum(flattened_params < 0) / len(flattened_params)
            zero_ratio = np.sum(flattened_params == 0) / len(flattened_params)
            sign_stats = [positive_ratio, negative_ratio, zero_ratio]

            non_zero_params = flattened_params[flattened_params != 0]
            if len(non_zero_params) > 0:
                mean_val = np.mean(non_zero_params)
                std_val = np.std(non_zero_params)
                skewness = np.mean(((non_zero_params - mean_val) / std_val) ** 3) if std_val > 0 else 0
                kurtosis = np.mean(((non_zero_params - mean_val) / std_val) ** 4) - 3 if std_val > 0 else 0
            else:
                skewness, kurtosis = 0, 0

            distribution_stats = [skewness, kurtosis]

            if len(parameters) > 1:
                layer_magnitudes = [np.mean(np.abs(p)) for p in parameters]
                layer_variance = np.std(layer_magnitudes)
                max_layer_ratio = np.max(layer_magnitudes) / (np.mean(layer_magnitudes) + 1e-8)
            else:
                layer_variance, max_layer_ratio = 0, 1

            layer_stats = [layer_variance, max_layer_ratio]
            sparsity_ratio = np.sum(np.abs(flattened_params) < 1e-6) / len(flattened_params)

            signature = np.array(magnitude_stats + sign_stats + distribution_stats + layer_stats + [sparsity_ratio])
            return np.nan_to_num(signature, nan=0.0, posinf=1.0, neginf=-1.0)

        except Exception as e:
            print(f"\u26A0\uFE0F  Signature generation error: {e}")
            return np.array([0.0] * 14)

    def update_client_profile(self, client_id, signature):
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = {
                'signatures': [],
                'mean_signature': None,
                'signature_cov': None,
                'update_count': 0
            }

        profile = self.client_profiles[client_id]
        profile['signatures'].append(signature)
        profile['update_count'] += 1

        if len(profile['signatures']) > self.signature_window:
            profile['signatures'].pop(0)

        if len(profile['signatures']) > 0:
            signatures_matrix = np.vstack(profile['signatures'])
            profile['mean_signature'] = np.mean(signatures_matrix, axis=0)
            profile['signature_cov'] = (
                np.cov(signatures_matrix.T) if len(profile['signatures']) > 1 else np.eye(len(signature)) * 0.01
            )

    def calculate_signature_distance(self, signature, client_id):
        if client_id not in self.client_profiles:
            return 0.0

        profile = self.client_profiles[client_id]
        if profile['mean_signature'] is None or len(profile['signatures']) < 2:
            return 0.0

        try:
            diff = signature - profile['mean_signature']
            cov_matrix = profile['signature_cov'] + np.eye(len(signature)) * 1e-6
            inv_cov = np.linalg.pinv(cov_matrix)
            mahalanobis_dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
            normalized_distance = mahalanobis_dist / np.sqrt(len(signature))
            normalized_distance = min(normalized_distance, 5.0)

            cosine_distance = 0
            if np.linalg.norm(signature) > 0 and np.linalg.norm(profile['mean_signature']) > 0:
                cosine_sim = np.dot(signature, profile['mean_signature']) / (
                    np.linalg.norm(signature) * np.linalg.norm(profile['mean_signature'])
                )
                cosine_distance = 1 - cosine_sim

            combined_score = 0.7 * normalized_distance + 0.3 * cosine_distance
            return combined_score

        except Exception as e:
            print(f"\u26A0\uFE0F  Distance calculation error: {e}")
            return 0.0

    def _get_client_id(self, client_proxy, fit_res):
        """Get client ID from fit results or fallback to proxy-based ID"""
        # Try to get client ID from fit results first
        if hasattr(fit_res, 'metrics') and fit_res.metrics and 'client_id' in fit_res.metrics:
            return fit_res.metrics['client_id']
        
        # Fallback to proxy-based ID assignment
        client_key = str(client_proxy)
        if client_key not in self.client_round_mapping:
            client_id = f"client_{len(self.client_round_mapping)}"
            self.client_round_mapping[client_key] = client_id
        return self.client_round_mapping[client_key]

    def verify_gradients(self, client_updates):
        print(f"\n\U0001F50D Round {self.round_count}: GSV Verification")
        print("=" * 55)

        if len(client_updates) == 0:
            return []

        filtered_updates = []
        client_scores = []

        warmup_rounds = 15
        min_updates_required = 3
        confidence_margin = 0.2

        for i, (client_proxy, fit_res) in enumerate(client_updates):
            client_id = self._get_client_id(client_proxy, fit_res)
            
            # ✅ Check if client is already blocked
            if client_id in self.blocked_clients:
                print(f"\U0001F6AB Client {client_id}: BLOCKED (previously detected)")
                with open(self.score_log_path, "a") as f:
                    f.write(f"{self.round_count},{client_id},0.0,1,1\n")
                continue

            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            signature = self.compute_gradient_signature(parameters)
            score = self.calculate_signature_distance(signature, client_id)
            client_scores.append(score)

            profile = self.client_profiles.get(client_id, {'update_count': 0})
            is_filtered = False
            is_blocked = False

            if self.round_count < warmup_rounds or profile['update_count'] < min_updates_required:
                print(f"\U0001F501 Client {client_id}: Warming up (score: {score:.4f})")
                self.update_client_profile(client_id, signature)
                filtered_updates.append((client_proxy, fit_res))
            elif score > self.adaptive_threshold + confidence_margin:
                print(f"\U0001F6A8 Client {client_id}: MALICIOUS DETECTED (score: {score:.4f})")
                
                # ✅ Track detection history
                if client_id not in self.detection_history:
                    self.detection_history[client_id] = []
                self.detection_history[client_id].append(self.round_count)
                
                # ✅ Block client after 2 consecutive detections
                if len(self.detection_history[client_id]) >= 2:
                    self.blocked_clients.add(client_id)
                    is_blocked = True
                    print(f"\U0001F6AB Client {client_id}: PERMANENTLY BLOCKED")
                
                self.attack_detection_count += 1
                self.total_updates_filtered += 1
                is_filtered = True
                
                # Don't add to filtered_updates - this client is rejected
            else:
                print(f"\u2705 Client {client_id}: Accepted (score: {score:.4f})")
                self.update_client_profile(client_id, signature)
                filtered_updates.append((client_proxy, fit_res))
                
                # ✅ Reset detection history on good behavior
                if client_id in self.detection_history:
                    self.detection_history[client_id] = []

            # ✅ Log detailed information
            with open(self.score_log_path, "a") as f:
                f.write(f"{self.round_count},{client_id},{score:.4f},{int(is_filtered)},{int(is_blocked)}\n")

        self._adjust_threshold(client_scores)

        print(f"\U0001F4CA Acceptance: {len(filtered_updates)}/{len(client_updates)} "
              f"({100.0 * len(filtered_updates)/len(client_updates):.2f}%)")
        print(f"\U0001F3AF Threshold: {self.adaptive_threshold:.4f}")
        print(f"\U0001F6AB Blocked clients: {len(self.blocked_clients)}")

        self.round_count += 1
        self.total_clients_processed += len(client_updates)

        return filtered_updates

    def _adjust_threshold(self, scores):
        if len(scores) == 0:
            return
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)

        if std_score > 0:
            if max_score > mean_score + std_score:
                self.adaptive_threshold *= 0.995
            elif mean_score < self.adaptive_threshold * 0.6:
                self.adaptive_threshold *= 1.005

        self.adaptive_threshold = max(0.3, min(3.0, self.adaptive_threshold))

    def get_verification_status(self):
        return {
            'round': self.round_count,
            'threshold': self.adaptive_threshold,
            'clients_tracked': len(self.client_profiles),
            'total_processed': self.total_clients_processed,
            'total_filtered': self.total_updates_filtered,
            'attacks_detected': self.attack_detection_count,
            'false_positives': self.false_positive_count,
            'filter_rate': self.total_updates_filtered / max(1, self.total_clients_processed),
            'blocked_clients': len(self.blocked_clients),
            'blocked_client_ids': list(self.blocked_clients)
        }


class GSVStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gsv_verifier = GradientSignatureVerifier()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"\n\U0001F50D Round {server_round}: GSV Aggregation")
        print("=" * 60)
        filtered_results = self.gsv_verifier.verify_gradients(results)
        return super().aggregate_fit(server_round, filtered_results, failures)

    def get_verification_status(self):
        return self.gsv_verifier.get_verification_status()


def load_validation_data():
    print("\U0001F4CA Loading validation data...")
    df = load_dataset()
    df = basic_cleaning(df)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X, y = remove_outliers(X, y)
    X = feature_engineering(X)
    X = scale_features(X)
    X = select_features(X, y)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for _, val_idx in splitter.split(X, y):
        X_val = X.iloc[val_idx].values
        y_val = y.iloc[val_idx].values

    print(f"\u2705 Validation set: {len(X_val)} samples")
    return X_val, y_val
