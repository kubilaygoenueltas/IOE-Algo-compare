import numpy as np
import pandas as pd
from pathlib import Path
import os
import my_prediction_functions as my

# LOAD DATA
results_path = "../imt-main/examples/results" 
ring_angles_kl = np.load(os.path.join(results_path, "Angles_CAM_Knee_Left.npy"), allow_pickle=True)
seq_filter = np.load(os.path.join(results_path, "Sequences_filter.npy"), allow_pickle=True)
seq_names = np.load(os.path.join(results_path, "Sequences_names.npy"), allow_pickle=True)


path_stiffMovement = "../Daten/Label/Stiff/"
path_extendedLeg = "../Daten/Label/Extended_Legs/"
behavior_extendedLeg = 'Extended Leg'
behavior_stiffMovement= 'Stiff Movement'
labels_extendedLeg = []
labels_stiffMovement = []
#### EXTENDED LEG
for i, file in enumerate(os.listdir(path_extendedLeg)):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        try:
            # Read label file
            all_labels = pd.read_excel(os.path.join(path_extendedLeg, file))

            interval = np.arange(0, len(ring_angles_kl[i]))

            # call label_extract function
            behavior_label, behavior_label_5s, start_frames_single = my.label_extract(
                all_labels, behavior_extendedLeg, interval, 30)

            # Store results
            labels_extendedLeg.append({
                'behavior_label': behavior_label,
                'behavior_label_5s': behavior_label_5s,
                'start_frames_single': start_frames_single
            })

        except Exception as e:
            print(f"Error processing {file}: {e}")

#### STIFF MOVEMENT
for i, file in enumerate(os.listdir(path_stiffMovement)):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        try:
            # Read label file
            all_labels = pd.read_excel(os.path.join(path_stiffMovement, file))

            interval = np.arange(0, len(ring_angles_kl[i]))

            # call label_extract_legrise function
            behavior_label, behavior_label_5s = my.label_extract_legrise(
                all_labels, behavior_stiffMovement, interval, labels_extendedLeg[i]['start_frames_single'], 30)

            # Store results
            labels_stiffMovement.append({
                'behavior_label': behavior_label,
                'behavior_label_5s': behavior_label_5s
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

#### SAVE RESULTS
path_results = "./results" 
os.makedirs(path_results, exist_ok=True) # check if directory exists
np.save(os.path.join(path_results, "labels_extendedLeg_CAM.npy"), np.array(labels_extendedLeg, dtype=object))  # Save files to results path
np.save(os.path.join(path_results, "labels_stiffMovement_CAM.npy"), np.array(labels_stiffMovement, dtype=object))
np.savetxt(
    os.path.join(path_results, "labels_extendedLeg_CAM.txt"),
    np.array(labels_extendedLeg, dtype=object),
    fmt="%s",  # All elements formatted as strings
    delimiter="\t",  # Tab-separated
    header="behavior_label\tbehavior_label_5s\tstart_frames_single",
    comments=""
)
np.savetxt(
    os.path.join(path_results, "labels_stiffMovement_CAM.txt"),
    np.array(labels_stiffMovement, dtype=object),
    fmt="%s",  # All elements formatted as strings
    delimiter="\t",  # Tab-separated
    header="behavior_label\tbehavior_label_5s",
    comments=""
)
print(f"Saved successfully to: {path_results}")