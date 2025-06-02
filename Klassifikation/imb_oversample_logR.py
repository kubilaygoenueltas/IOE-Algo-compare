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
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    SVMSMOTE,
    ADASYN
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

# Labels
y_stiffMovement = [entry['behavior_label'] for entry in labels_stiffMovement_100]
y_extendedLeg = [entry['behavior_label'] for entry in labels_extendedLeg_100]

# Create dataset
X = []
y = []
groups = []

for i in range(len(ring_angles_kl)):
    left = ring_angles_kl[i]
    right = ring_angles_kr[i]
    label = y_extendedLeg[i]

    if len(left) == len(right) == len(label):
        X.append(np.column_stack([left, right]))
        y.append(label)
        group_id = i
        groups.extend([group_id] * len(label))
    else:
        print(f"Skipping dataset {i} due to length mismatch.")

X = np.vstack(X)
y = np.concatenate(y)
groups = np.array(groups)

#print(f"X shape: {X.shape}, y shape: {y.shape}, groups shape: {groups.shape}")

# Classifier
clf = LogisticRegression(
    class_weight='balanced',
    random_state=1,
    max_iter=1000
)

# Define over-sampling methods
samplers = {
    "RandomOverSampler": RandomOverSampler(random_state=1, shrinkage=False),
    "RandomOverSampler (smoothed)": RandomOverSampler(random_state=1, shrinkage=True),
    "SMOTE": SMOTE(random_state=1),
    "BorderlineSMOTE1": BorderlineSMOTE(kind='borderline-1', random_state=1),
    "BorderlineSMOTE2": BorderlineSMOTE(kind='borderline-2', random_state=1),
    "KMeansSMOTE": KMeansSMOTE(random_state=1),
    #"SVMSMOTE": SVMSMOTE(random_state=1),
    "ADASYN": ADASYN(random_state=1)
}

# Cross-validation
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
results = {}

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

# Results
df_results = pd.DataFrame(results).T
print("\nComparison of Over-Sampling Methods:\n")
print(df_results.round(3))
