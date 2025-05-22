import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import os

# Beispiel: Eingabedaten (X) und Zielvariable (y)
# X: Merkmale [knee_angle_left, knee_angle_right]
# y: Labels (0 = nicht gestreckt/nicht steif, 1 = gestreckt/steif)

results_path_knee_angle = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/imt-main/examples/results"
ring_angles_kl = np.load(os.path.join(results_path_knee_angle, "Angles_IMU_RING_Knee_Left.npy"), allow_pickle=True)
ring_angles_kr = np.load(os.path.join(results_path_knee_angle, "Angles_IMU_RING_Knee_Right.npy"), allow_pickle=True)
results_path_label = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Klassifikation/results"
labels_extendedLeg_100 = np.load(os.path.join(results_path_label, "labels_extendedLeg_RING_100.npy"), allow_pickle=True)
labels_stiffMovement_100 = np.load(os.path.join(results_path_label, "labels_stiffMovement_RING_100.npy"), allow_pickle=True)

X = np.column_stack([ring_angles_kl, ring_angles_kr])
y_extendedLeg = [entry['behavior_label_5s'] for entry in labels_extendedLeg_100]
y_stiffMovement = [entry['behavior_label'] for entry in labels_stiffMovement_100]
y = y_stiffMovement

groups = groups = np.concatenate([[i] * len(y[i]) for i in range(len(y))])

# Initialisierung des Klassifikators
clf = LogisticRegression(
    class_weight='balanced',
    random_state=1,
    max_iter=1000
)

# Stratified Group K-Fold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)

scores = []
for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    score = balanced_accuracy_score(y_test, y_pred)
    scores.append(score)

print("Balanced Accuracy Scores:", scores)
print("Mean Balanced Accuracy:", np.mean(scores))
