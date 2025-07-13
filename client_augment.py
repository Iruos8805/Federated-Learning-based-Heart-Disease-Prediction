"""
Client-Side Data Augmentation for Federated Learning
Integrates SMOTE, Noise, Interpolation, MixUp, and Dropout
Supports 'light', 'strong', and individual modes
"""

import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


class ClientDataAugmenter:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def augment_with_smote(self, X, y, target_size):
        current_size = len(X)
        if current_size >= target_size:
            return X, y

        unique_classes = np.unique(y)
        class_counts = np.bincount(y)
        target_per_class = target_size // len(unique_classes)

        sampling_strategy = {}
        for i, count in enumerate(class_counts):
            if i < len(unique_classes):
                sampling_strategy[i] = max(target_per_class, count)

        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=min(5, len(X) - 1)
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTE failed: {e}")
            return X, y

    def augment_with_noise(self, X, y, noise_factor=0.05, multiplier=2):
        X_aug = [X]
        y_aug = [y]

        feature_stds = np.std(X, axis=0)

        for i in range(multiplier):
            factor = noise_factor * (1 + i * 0.1)
            noise = np.random.normal(0, factor * feature_stds, X.shape)
            X_noisy = X + noise
            X_noisy = np.clip(X_noisy, X.min(axis=0) - 2*feature_stds, X.max(axis=0) + 2*feature_stds)
            X_aug.append(X_noisy)
            y_aug.append(y)

        return np.vstack(X_aug), np.hstack(y_aug)

    def augment_with_interpolation(self, X, y, num_samples):
        if num_samples <= 0:
            return X, y

        X_synth, y_synth = [], []
        unique_classes = np.unique(y)
        per_class = num_samples // len(unique_classes)

        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) >= 2:
                for _ in range(per_class):
                    i1, i2 = np.random.choice(cls_idx, 2, replace=False)
                    alpha = np.random.uniform(0.1, 0.9)
                    sample = alpha * X[i1] + (1 - alpha) * X[i2]
                    X_synth.append(sample)
                    y_synth.append(cls)

        return np.vstack([X, X_synth]), np.hstack([y, y_synth])

    def augment_with_mixup(self, X, y, num_samples, alpha=0.2):
        X_mix, y_mix = [], []
        for _ in range(num_samples):
            i1, i2 = np.random.choice(len(X), 2, replace=False)
            lam = np.random.beta(alpha, alpha)
            X_new = lam * X[i1] + (1 - lam) * X[i2]
            y_new = y[i1] if lam > 0.5 else y[i2]
            X_mix.append(X_new)
            y_mix.append(y_new)

        return np.vstack([X, X_mix]), np.hstack([y, y_mix])

    def apply_feature_dropout(self, X, rate=0.1):
        X_dropped = X.copy()
        for i in range(len(X)):
            mask = np.random.rand(X.shape[1]) < rate
            X_dropped[i, mask] = 0
        return X_dropped


def augment_client_data(X, y, target_size=2000, method="combined"):
    """
    Client-side augmentation function
    Args:
        X, y: input data (numpy arrays)
        target_size: desired number of samples after augmentation
        method: 'light', 'strong', 'smote', 'noise', 'interpolation', 'combined'
    Returns:
        X_aug, y_aug: augmented data
    """
    augmenter = ClientDataAugmenter()
    X_aug, y_aug = X.copy(), y.copy()

    if method == "smote":
        return augmenter.augment_with_smote(X_aug, y_aug, target_size)

    elif method == "noise":
        multiplier = max(1, target_size // len(X))
        return augmenter.augment_with_noise(X_aug, y_aug, multiplier=multiplier)

    elif method == "interpolation":
        return augmenter.augment_with_interpolation(X_aug, y_aug, target_size - len(X))

    elif method == "light":
        stage1_target = int(target_size * 0.6)
        X_aug, y_aug = augmenter.augment_with_smote(X_aug, y_aug, stage1_target)

        rem = target_size - len(X_aug)
        if rem > 0:
            X_aug, y_aug = augmenter.augment_with_interpolation(X_aug, y_aug, rem)

        X_aug = augmenter.apply_feature_dropout(X_aug, rate=0.02)

        if len(X_aug) > target_size:
            idx = np.random.choice(len(X_aug), target_size, replace=False)
            X_aug, y_aug = X_aug[idx], y_aug[idx]

        return X_aug, y_aug

    elif method in ["strong", "combined"]:
        stage1_target = int(target_size * 0.6)
        X_aug, y_aug = augmenter.augment_with_smote(X_aug, y_aug, stage1_target)

        stage2_target = int(target_size * 0.85)
        rem = stage2_target - len(X_aug)
        if rem > 0:
            X_aug, y_aug = augmenter.augment_with_noise(X_aug, y_aug, multiplier=max(1, rem // len(X_aug)))

        rem = target_size - len(X_aug)
        if rem > 0:
            X_aug, y_aug = augmenter.augment_with_interpolation(X_aug, y_aug, rem)

        X_aug, y_aug = augmenter.augment_with_mixup(X_aug, y_aug, int(0.1 * target_size))
        X_aug = augmenter.apply_feature_dropout(X_aug, rate=0.05)

        if len(X_aug) > target_size:
            idx = np.random.choice(len(X_aug), target_size, replace=False)
            X_aug, y_aug = X_aug[idx], y_aug[idx]

        return X_aug, y_aug

    else:
        raise ValueError(f"Unknown augmentation method: {method}")
