import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import os
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Beispiel: Eingabedaten (X) und Zielvariable (y)
# X: Merkmale [knee_angle_left, knee_angle_right]
# y: Labels (0 = nicht gestreckt/nicht steif, 1 = gestreckt/steif)

# Load knee angle data
results_path_knee_angle = "../imt-main/examples/results"
ring_angles_kl = np.load(os.path.join(results_path_knee_angle, "Angles_IMU_RING_Knee_Left.npy"), allow_pickle=True)
#ring_angles_kl = ring_angles_kl[:52]    # Korrektur, später entfernen
ring_angles_kr = np.load(os.path.join(results_path_knee_angle, "Angles_IMU_RING_Knee_Right.npy"), allow_pickle=True)
#ring_angles_kr = ring_angles_kr[:52]

seq_names = np.load(os.path.join(results_path_knee_angle, "Sequences_names.npy"), allow_pickle=True)

# Load labels
results_path_label = "./results"
labels_extendedLeg_100 = np.load(os.path.join(results_path_label, "labels_extendedLeg_RING_100.npy"), allow_pickle=True)
labels_stiffMovement_100 = np.load(os.path.join(results_path_label, "labels_stiffMovement_RING_100.npy"), allow_pickle=True)

# Load raw IMU data
imu_data_ml = np.load(os.path.join(results_path_label, "IMU_data_ml.npy"), allow_pickle=True)

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

#### Initialize classifier
clf = LogisticRegression(
    class_weight='balanced',
    random_state=1,
    max_iter=1000
)
"""clf = LogisticRegression(
    class_weight={0: 1, 1: 25},
    random_state=1,
    max_iter=1000,
    solver='liblinear'
)"""
"""clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', random_state=1, max_iter=1000)
)"""
"""clf = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    max_depth=5,
    random_state=1
)"""

# StratifiedGroupKFold CV
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)

# Store metrics
balanced_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Cross-validation loop
for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    balanced_scores.append(balanced_accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

#### OUTPUT PRINT
"""print("Evaluation Metrics per Fold:\n")

for i in range(len(balanced_scores)):
    print(f"Fold {i+1}:")
    print(f"  Balanced Accuracy: {balanced_scores[i]:.3f}")
    print(f"  Precision        : {precision_scores[i]:.3f}")
    print(f"  Recall           : {recall_scores[i]:.3f}")
    print(f"  F1 Score         : {f1_scores[i]:.3f}")

    # Print detailed classification report
    print(f"\nFold {i+1} Classification Report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    # Print confusion matrix
    print(f"Fold {i+1} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()"""

print("Mean Metrics Across Folds:")
print(f"Mean Balanced Accuracy: {np.mean(balanced_scores):.3f}")
print(f"Mean Precision        : {np.mean(precision_scores):.3f}")
print(f"Mean Recall           : {np.mean(recall_scores):.3f}")
print(f"Mean F1 Score         : {np.mean(f1_scores):.3f}")