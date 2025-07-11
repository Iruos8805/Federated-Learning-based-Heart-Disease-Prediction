"""
Biological Anomaly Filtering (BAF) - Immune System Inspired Defense for Federated Learning
Author: [Your Name]
Description: Novel adversarial defense using functional fingerprinting and immune system principles
"""

import numpy as np
import flwr as fl
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from dataset import load_dataset
from preprocessing import basic_cleaning
from modifications import remove_outliers, feature_engineering, scale_features, select_features
import warnings
warnings.filterwarnings("ignore")


class BiologicalAnomalyFilter:
    """
    Biological Anomaly Filtering (BAF) - Immune system inspired defense
    for federated learning against adversarial attacks.
    
    Key Innovations:
    1. Functional fingerprinting instead of parameter-space analysis
    2. Adaptive immune memory with self-profile evolution
    3. Multi-layered anomaly detection (distance + similarity)
    4. Affinity maturation for dynamic threshold adjustment
    """
    
    def __init__(self, validation_data, sensitivity_threshold=0.5, memory_decay=0.95):
        """
        Initialize the Biological Anomaly Filter
        
        Args:
            validation_data: (X, y) tuple for functional fingerprinting
            sensitivity_threshold: Initial detection threshold
            memory_decay: Memory decay rate for immune senescence
        """
        self.validation_X, self.validation_y = validation_data
        self.sensitivity_threshold = sensitivity_threshold
        self.memory_decay = memory_decay
        
        # Immune system components
        self.self_profile = None  # Memory of "self" - normal client behavior
        self.memory_cells = []    # Memory of past legitimate updates
        self.attack_signatures = []  # Memory of detected attacks
        self.round_count = 0
        
        # Adaptive threshold mechanism (affinity maturation)
        self.adaptive_threshold = sensitivity_threshold
        self.false_positive_count = 0
        self.attack_detection_count = 0
        
        # Initialize reference model for functional fingerprinting
        self.reference_model = self._create_reference_model()
        
        print("🦠 Biological Anomaly Filter (BAF) initialized")
        print(f"   Validation set size: {len(self.validation_X)}")
        print(f"   Initial sensitivity: {sensitivity_threshold}")
        
    def _create_reference_model(self):
        """Create reference model for functional fingerprinting"""
        model = make_pipeline(
            RBFSampler(gamma=0.028092305159489246, n_components=1288, random_state=42),
            SGDClassifier(loss='hinge', alpha=1.0 / 400.7935817191417, max_iter=1000, random_state=42)
        )
        
        # Fit RBF sampler
        model.named_steps['rbfsampler'].fit(self.validation_X)
        
        # Initialize classifier
        X_transformed = model.named_steps['rbfsampler'].transform(self.validation_X)
        model.named_steps['sgdclassifier'].partial_fit(
            X_transformed, 
            self.validation_y, 
            classes=np.unique(self.validation_y)
        )
        
        return model
    
    def compute_functional_fingerprint(self, parameters):
        """
        Compute functional fingerprint of model parameters.
        
        CORE INNOVATION: Instead of analyzing parameters directly,
        we analyze what the parameters DO (functional behavior).
        This represents 'antigen presentation' in immune system terms.
        
        Args:
            parameters: Model parameters from client
            
        Returns:
            numpy.ndarray: Functional fingerprint vector
        """
        if len(parameters) == 0:
            return np.array([])
            
        # Apply parameters to reference model
        temp_model = self._apply_parameters_to_model(parameters)
        
        # Compute behavioral signatures
        try:
            # 1. Decision function outputs (for SVM with hinge loss)
            decision_outputs = temp_model.decision_function(self.validation_X)
            
            # 2. Prediction labels and confidence
            pred_labels = temp_model.predict(self.validation_X)
            
            # For SVM, we can derive confidence from decision function
            # Distance from decision boundary indicates confidence
            pred_confidence = np.abs(decision_outputs)
            
            # 3. Class distribution
            class_distribution = np.bincount(pred_labels, minlength=2) / len(pred_labels)
            
            # 4. Decision boundary analysis
            # Sample subset for efficiency
            sample_size = min(50, len(self.validation_X))
            sample_indices = np.random.choice(len(self.validation_X), sample_size, replace=False)
            
            decision_sample = decision_outputs[sample_indices]
            
            # 5. Margin analysis (distance from decision boundary)
            margin_mean = np.mean(np.abs(decision_sample))
            margin_std = np.std(np.abs(decision_sample))
            
            # 6. Support vector analysis (points close to decision boundary)
            support_vector_ratio = np.sum(np.abs(decision_outputs) < 1.0) / len(decision_outputs)

            pred_labels = temp_model.predict(self.validation_X)
            # 2.5 Label flip ratio — how much the model disagrees with ground truth
            label_flip_ratio = np.mean(pred_labels != self.validation_y)
            label_flip_ratio = np.mean(pred_labels != self.validation_y)
            print(f"🔍 Label flip ratio: {label_flip_ratio:.4f}")


            
            # 7. Combine into functional fingerprint
            fingerprint = np.concatenate([
                [pred_confidence.mean()],           # Average prediction confidence
                [pred_confidence.std()],            # Confidence variance
                class_distribution,                 # Class distribution (2 values)
                [margin_mean],                      # Average margin
                [margin_std],                       # Margin variance
                [support_vector_ratio],             # Support vector ratio
                [decision_outputs.mean()],          # Mean decision output
                [decision_outputs.std()],            # Decision output variance
                [label_flip_ratio]
            ])
            
            return fingerprint
            
        except Exception as e:
            print(f"⚠️  Error computing functional fingerprint: {e}")
            # Return default fingerprint with correct dimensions (9 features)
            return np.array([0.0] * 9)
    
    
    def _apply_parameters_to_model(self, parameters):
        """Apply parameters to a copy of reference model"""
        # Create a copy of the reference model
        temp_model = make_pipeline(
            RBFSampler(gamma=0.028092305159489246, n_components=1288, random_state=42),
            SGDClassifier(loss='hinge', alpha=1.0 / 400.7935817191417, max_iter=1000, random_state=42)
        )
        
        # Fit RBF sampler (same as reference)
        temp_model.named_steps['rbfsampler'].fit(self.validation_X)
        
        # Initialize classifier
        X_transformed = temp_model.named_steps['rbfsampler'].transform(self.validation_X)
        temp_model.named_steps['sgdclassifier'].partial_fit(
            X_transformed, 
            self.validation_y, 
            classes=np.unique(self.validation_y)
        )
        
        # Apply new parameters
        clf = temp_model.named_steps['sgdclassifier']
        
        if len(parameters) >= 1 and hasattr(clf, 'coef_'):
            clf.coef_ = parameters[0].reshape(clf.coef_.shape)
        
        if len(parameters) >= 2 and hasattr(clf, 'intercept_'):
            clf.intercept_ = parameters[1].reshape(clf.intercept_.shape)
            
        return temp_model
    
    def calculate_mahalanobis_distance(self, fingerprint, profile_mean, profile_cov):
        """
        Calculate Mahalanobis distance for anomaly detection.
        This represents the 'distance-to-self' metric in immune system terms.
        """
        try:
            if profile_cov.shape[0] == 0 or fingerprint.shape[0] == 0:
                return float('inf')
                
            diff = fingerprint - profile_mean
            
            # Add small regularization to avoid singular matrix
            regularized_cov = profile_cov + np.eye(profile_cov.shape[0]) * 1e-6
            
            # Compute inverse covariance
            try:
                inv_cov = np.linalg.inv(regularized_cov)
            except np.linalg.LinAlgError:
                # If still singular, use pseudo-inverse
                inv_cov = np.linalg.pinv(regularized_cov)
            
            distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
            return distance
            
        except Exception as e:
            print(f"⚠️  Error calculating Mahalanobis distance: {e}")
            return float('inf')
    
    def immune_response(self, client_updates):
        """
        Main immune response function - filters client updates using
        biological anomaly detection principles.
        
        FIXED: Added proper return statement and complete implementation
        """
        print(f"\n🔬 Round {self.round_count}: Immune System Analysis")
        print("=" * 50)
        
        filtered_updates = []
        client_risks = []
        
        # Extract functional fingerprints for all clients
        fingerprints = []
        for client_proxy, fit_res in client_updates:
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            fingerprint = self.compute_functional_fingerprint(parameters)
            fingerprints.append(fingerprint)
        
        # EXTENDED self-profile building period (rounds 0-4 instead of 0-1)
        if self.round_count <= 6:  # Changed from 1 to 4
            # Initial rounds: establish "self" profile (immune tolerance phase)
            self._build_initial_self_profile(fingerprints)
            # Accept all updates in early rounds
            filtered_updates = client_updates
            client_risks = [0.0] * len(client_updates)
            print(f"🔄 Building self-profile (immune tolerance phase) - Round {self.round_count}/4")
            
        else:
            # Subsequent rounds: detect anomalies
            legitimate_fingerprints = []
            
            for i, ((client_proxy, fit_res), fingerprint) in enumerate(zip(client_updates, fingerprints)):
                risk_score = self._assess_client_risk(fingerprint, client_proxy)
                client_risks.append(risk_score)
                
                if risk_score > self.adaptive_threshold:
                    print(f"🚨 Client {i} flagged as anomalous (risk: {risk_score:.4f})")
                    self.attack_detection_count += 1
                    # Store attack signature for future recognition
                    self.attack_signatures.append(fingerprint)
                else:
                    print(f"✅ Client {i} accepted (risk: {risk_score:.4f})")
                    filtered_updates.append((client_proxy, fit_res))
                    # Update memory with legitimate update
                    self._update_immune_memory(fingerprint)
                    legitimate_fingerprints.append(fingerprint)
            
            # Update self-profile with legitimate updates
            if legitimate_fingerprints:
                self._update_self_profile(legitimate_fingerprints)
            
            # Adjust immune sensitivity
            self._adjust_immune_sensitivity()
        
        # FIXED: Increment round count and return filtered updates
        self.round_count += 1
        
        print(f"📊 Immune Summary: {len(filtered_updates)}/{len(client_updates)} clients accepted")
        print(f"🎯 Current sensitivity threshold: {self.adaptive_threshold:.4f}")
        
        return filtered_updates
    
    def _build_initial_self_profile(self, fingerprints):
        """Build initial self-profile from early rounds"""
        if len(fingerprints) == 0:
            return
            
        # Stack fingerprints
        stacked_fingerprints = np.vstack(fingerprints)
        
        # Compute mean and covariance for self-profile
        self.self_profile = {
            'mean': np.mean(stacked_fingerprints, axis=0),
            'cov': np.cov(stacked_fingerprints.T),
            'count': len(fingerprints)
        }
        
        print(f"🧬 Self-profile established with {len(fingerprints)} clients")
    
    def _assess_client_risk(self, fingerprint, client_proxy):
        """
        Assess risk score for a client update using multi-layered detection
        """
        if self.self_profile is None or len(fingerprint) == 0:
            return 0.0
            
        # Primary detection: distance from self-profile
        distance_to_self = self.calculate_mahalanobis_distance(
            fingerprint, 
            self.self_profile['mean'], 
            self.self_profile['cov']
        )
        
        # NORMALIZE distance based on dimensionality and expected evolution
        normalized_distance = distance_to_self / np.sqrt(len(fingerprint))
        
        # ADAPTIVE baseline - expect some evolution
        evolution_allowance = 0.8 * self.round_count  # Allow more change over time
        adjusted_distance = max(0, normalized_distance - evolution_allowance)
        
        # Secondary detection: similarity to known attack signatures
        attack_similarity = 0.0
        if len(self.attack_signatures) > 0:
            for attack_sig in self.attack_signatures:
                if len(attack_sig) == len(fingerprint):
                    similarity = np.dot(fingerprint, attack_sig) / (
                        np.linalg.norm(fingerprint) * np.linalg.norm(attack_sig) + 1e-8
                    )
                    attack_similarity = max(attack_similarity, similarity)
        
        # Combine risk factors with reduced weight
        risk_score = adjusted_distance + attack_similarity * 0.05  # Reduced from 0.3
        
        return risk_score
    
    def _update_immune_memory(self, fingerprint):
        """Update immune memory with legitimate updates (memory B-cells)"""
        self.memory_cells.append(fingerprint)
        
        # Limit memory size (immune senescence)
        if len(self.memory_cells) > 100:
            self.memory_cells.pop(0)
    
    def _update_self_profile(self, legitimate_fingerprints):
        """
        Update self-profile with new legitimate updates
        
        INNOVATION: Adaptive self-definition that evolves with model
        """
        if len(legitimate_fingerprints) == 0 or self.self_profile is None:
            return
            
        # Weighted update (recent updates have more influence)
        stacked_fingerprints = np.vstack(legitimate_fingerprints)
        new_mean = np.mean(stacked_fingerprints, axis=0)
        new_cov = np.cov(stacked_fingerprints.T)
        
        # Exponential moving average update
        alpha = 0.3  # Learning rate for self-profile evolution
        self.self_profile['mean'] = (1 - alpha) * self.self_profile['mean'] + alpha * new_mean
        self.self_profile['cov'] = (1 - alpha) * self.self_profile['cov'] + alpha * new_cov
        self.self_profile['count'] += len(legitimate_fingerprints)
    
    def _adjust_immune_sensitivity(self):
        """
        Adjust immune sensitivity based on performance (affinity maturation)
        """
        total_detections = self.attack_detection_count + self.false_positive_count
        
        if total_detections > 0:
            # Calculate false positive rate
            false_positive_rate = self.false_positive_count / total_detections
            
            if false_positive_rate > 0.1:  # More than 10% false positives (reduced from 20%)
                self.adaptive_threshold *= 1.2  # Increase threshold more (reduce sensitivity)
                print(f"🔧 Reducing immune sensitivity due to false positives")
                
            elif false_positive_rate < 0.02:  # Less than 2% false positives (reduced from 5%)
                self.adaptive_threshold *= 0.98  # Smaller adjustment (was 0.95)
                print(f"🔧 Slightly increasing immune sensitivity")
        
        # More conservative bounds
        self.adaptive_threshold = max(0.1, min(2.0, self.adaptive_threshold))  # Wider range
    
    def get_immune_status(self):
        """Get current immune system status for monitoring"""
        return {
            'round': self.round_count,
            'sensitivity': self.adaptive_threshold,
            'memory_cells': len(self.memory_cells),
            'attack_signatures': len(self.attack_signatures),
            'self_profile_established': self.self_profile is not None,
            'attacks_detected': self.attack_detection_count,
            'false_positives': self.false_positive_count
        }


class BAFStrategy(fl.server.strategy.FedAvg):
    """
    Custom Federated Averaging strategy with Biological Anomaly Filtering
    
    This strategy integrates the BAF immune system into the standard FedAvg
    aggregation process, providing adversarial defense capabilities.
    """
    
    def __init__(self, validation_data, *args, **kwargs):
        """
        Initialize BAF-enhanced FedAvg strategy
        
        Args:
            validation_data: (X, y) tuple for immune system validation
        """
        super().__init__(*args, **kwargs)
        self.baf_filter = BiologicalAnomalyFilter(validation_data)
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results with immune system filtering
        
        CORE INTEGRATION: BAF filtering before FedAvg aggregation
        """
        
        print(f"\n🧠 Round {server_round}: BAF Strategy Aggregation")
        print("=" * 60)
        
        # Apply biological anomaly filtering
        filtered_results = self.baf_filter.immune_response(results)
        
        # Fallback mechanism: if all clients rejected, use original results
        if len(filtered_results) == 0:
            print("⚠️  All clients rejected by immune system - using original results")
            filtered_results = results
        
        # Proceed with standard FedAvg aggregation on filtered results
        return super().aggregate_fit(server_round, filtered_results, failures)
    
    def get_immune_status(self):
        """Get current immune system status"""
        return self.baf_filter.get_immune_status()


"""
Updated load_validation_data function to replace in your BAF system
This connects to your augment.py file for enhanced data loading
"""

def load_validation_data():
    """
    Load validation data for the immune system without augmentation.
    
    Returns:
        X_val, y_val: Clean validation data
    """
    print("📊 Loading validation data for immune system...")

    df = load_dataset()
    df = basic_cleaning(df)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X, y = remove_outliers(X, y)
    X = feature_engineering(X)
    X = scale_features(X)
    X = select_features(X, y)

    # Use a subset for validation (immune system reference)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for _, val_idx in splitter.split(X, y):
        X_val = X.iloc[val_idx].values  # Return as numpy arrays
        y_val = y.iloc[val_idx].values

    print(f"   Validation set size: {len(X_val)}")
    print(f"   Class distribution: {np.bincount(y_val)}")

    return X_val, y_val

