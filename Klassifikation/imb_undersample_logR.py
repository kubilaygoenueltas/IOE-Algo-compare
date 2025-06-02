import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from collections import Counter
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
    TomekLinks,
    AllKNN,
    CondensedNearestNeighbour,
    OneSidedSelection,
    NeighbourhoodCleaningRule
)
import pandas as pd
import os

# Load knee angle data
results_path_knee_angle = "../imt-main/examples/results"
ring_angles_kl = np.load(os.path.join(results_path_knee_angle, "Angles_IMU_RING_Knee_Left.npy"), allow_pickle=True)
ring_angles_kr = np.load(os.path.join(results_path_knee_angle, "Angles_IMU_RING_Knee_Right.npy"), allow_pickle=True)

seq_names = np.load(os.path.join(results_path_knee_angle, "Sequences_names.npy"), allow_pickle=True)

# Load labels
results_path_label = "./results"
labels_extendedLeg_100 = np.load(os.path.join(results_path_label, "labels_extendedLeg_RING_100.npy"), allow_pickle=True)
labels_stiffMovement_100 = np.load(os.path.join(results_path_label, "labels_stiffMovement_RING_100.npy"), allow_pickle=True)

# Load raw IMU data
#imu_data_ml = np.load(os.path.join(results_path_label, "IMU_data_ml.npy"), allow_pickle=True)

y_stiffMovement = [entry['behavior_label'] for entry in labels_stiffMovement_100]
y_extendedLeg = [entry['behavior_label'] for entry in labels_extendedLeg_100]

X = []
y = []
groups = []

for i in range(len(ring_angles_kl)):
    left = ring_angles_kl[i]
    right = ring_angles_kr[i]
    # Choose which label to use
    #label = y_stiffMovement[i]
    label = y_extendedLeg[i]

    if len(left) == len(right) == len(label):
        X.append(np.column_stack([left, right]))  # shape (n_i, 2)
        y.append(label)                           # shape (n_i,)
        group_id = i #// 2                    # every 1 datasets form one group
        groups.extend([group_id] * len(label))
    else:
        print(f"Skipping dataset {i} due to length mismatch.")

# Final stacked arrays
X = np.vstack(X)            # shape (total_samples, 2)
y = np.concatenate(y)       # shape (total_samples,)
groups = np.array(groups)   # shape (total_samples,)

# Check consistency
#print(f"X shape: {X.shape}, y shape: {y.shape}, groups shape: {groups.shape}")

# Initialize classifier
clf = LogisticRegression(
    class_weight='balanced',
    random_state=1,
    max_iter=1000
)

# Define under-sampling methods
samplers = {
    "RandomUnderSampler": RandomUnderSampler(random_state=1),
    "NearMiss1": NearMiss(version=1),
    #"NearMiss2": NearMiss(version=2),
    "NearMiss3": NearMiss(version=3),
    "TomekLinks": TomekLinks(),
    "AllKNN": AllKNN(),
    #"CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=1),
    "OneSidedSelection": OneSidedSelection(random_state=1),
    "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule()
}

# StratifiedGroupKFold CV
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)

# Dictionary to store evaluation metrics
results = {}

# Cross-validation loop for each sampler
for name, sampler in samplers.items():
    balanced_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    #print(f"\nEvaluating {name}...")
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            break

        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test)

        balanced_scores.append(balanced_accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    if len(balanced_scores) == 5:
        results[name] = {
            "Balanced Accuracy": np.mean(balanced_scores),
            "Precision": np.mean(precision_scores),
            "Recall": np.mean(recall_scores),
            "F1 Score": np.mean(f1_scores)
        }

# Show results
df_results = pd.DataFrame(results).T
print("\nComparison of Under-Sampling Methods:\n")
print(df_results.round(3))