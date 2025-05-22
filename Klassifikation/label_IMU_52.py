import numpy as np
import pandas as pd
from pathlib import Path
import os
from label_IMU import label_extract, label_extract_legrise # 'Extended Leg', 'Stiff Movement'

# LOAD DATA
path_data = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/"
results_path = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/imt-main/examples/results"
folder_data_imu = path_data + "Dict_Frames/"
folder_data_cam = path_data + "Knee_angle2d/"
filenames_imu = [str(f) for f in Path(folder_data_imu).rglob('*') if f.is_file()]  # all filnames in array
filenames_cam = [str(f) for f in Path(folder_data_cam).rglob('*') if f.is_file()]
sensors = ["S0333", "S1094", "S0593", "S0994", "S0477"] 
sequences = pd.read_csv(path_data + "sequences.txt", delimiter="\t", header=None, index_col=0)
# 1 und 41 ohne Sternum,25 rot, 26 S0194 fehlt, 53 keine Kamera datei, 35 S0994 fehlt, 18 fehlt, ab 54]
seq_cam_start = sequences.iloc[:, 0:1].values.ravel()  # CAM start frame
seq_imu_start = sequences.iloc[:, 1:2].values.ravel()   # IMU start frame
seq_filter = np.load(os.path.join(results_path, "Sequences_filter.npy"), allow_pickle=True)
seq_names = np.load(os.path.join(results_path, "Sequences_names.npy"), allow_pickle=True)

imu_data_all = []

for i in range(len(filenames_imu)):
    if seq_names[i] not in seq_filter:
        # LOAD DATA
        data_imu_seq = np.load(filenames_imu[i], allow_pickle=True).item()
        data_cam_both_seq = np.loadtxt(filenames_cam[i], delimiter=",") # Kniewinkel aus Videodaten beide Beine
        data_cam_kr_seq = data_cam_both_seq[:,1][seq_cam_start[i]:]      # [:,1] nur Knie rechts, [:,0] nur Knie links
        data_cam_kl_seq = data_cam_both_seq[:,0][seq_cam_start[i]:]      # [i:] alles vor index i entfernen

        # PREPARE DATA
        imu_data = {
            i: dict(acc=data_imu_seq[sensors[i]]["acc"][seq_imu_start[i]:], gyr=data_imu_seq[sensors[i]]["gyr_rad"][seq_imu_start[i]:])
            for i in range(len(sensors))  
            }
        
        imu_data_all.append(imu_data)
    else:
        imu_data = {
            i: dict(acc=np.zeros(3), gyr=np.zeros(3))
            for i in range(len(sensors))  
            }
        
        imu_data_all.append(imu_data)

imu_data_lengths = [len(seq[1]['acc']) if 1 in seq and 'acc' in seq[1] else None for seq in imu_data_all]

path_stiffMovement = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/Label/Stiff/"
path_extendedLeg = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/Label/Extended_Legs/"
behavior_extendedLeg = 'Extended Leg'
behavior_stiffMovement= 'Stiff Movement'
labels_extendedLeg = []
labels_stiffMovement = []

#### EXTENDED LEG
for i, file in enumerate(os.listdir(path_extendedLeg)):
    if seq_names[i] not in seq_filter:
        if file.endswith('.xlsx') or file.endswith('.xls'):
            try:
                # Read label file
                all_labels = pd.read_excel(os.path.join(path_extendedLeg, file))

                interval = np.arange(0, imu_data_lengths[i])

                # call label_extract function
                behavior_label, behavior_label_5s, start_frames_single = label_extract(
                    all_labels, behavior_extendedLeg, interval)

                # Store results
                labels_extendedLeg.append({
                    'behavior_label': behavior_label,
                    'behavior_label_5s': behavior_label_5s,
                    'start_frames_single': start_frames_single
                })

            except Exception as e:
                print(f"Error processing {file}: {e}")
    else: 
        labels_extendedLeg.append({
                    'behavior_label': np.zeros(3),
                    'behavior_label_5s': np.zeros(3),
                    'start_frames_single': np.zeros(3)
                })

#### STIFF MOVEMENT
for i, file in enumerate(os.listdir(path_stiffMovement)):
    if seq_names[i] not in seq_filter:
        if file.endswith('.xlsx') or file.endswith('.xls'):
            try:
                # Read label file
                all_labels = pd.read_excel(os.path.join(path_stiffMovement, file))

                interval = np.arange(0, imu_data_lengths[i])

                # call label_extract_legrise function
                behavior_label, behavior_label_5s = label_extract_legrise(
                    all_labels, behavior_stiffMovement, interval, labels_extendedLeg[i]['start_frames_single'])

                # Store results
                labels_stiffMovement.append({
                    'behavior_label': behavior_label,
                    'behavior_label_5s': behavior_label_5s
                })
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
    else: 
        labels_stiffMovement.append({
                    'behavior_label': np.zeros(3),
                    'behavior_label_5s': np.zeros(3)
                })

#### SAVE RESULTS
path_results = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Klassifikation/results" 
os.makedirs(path_results, exist_ok=True) # check if directory exists
np.save(os.path.join(path_results, "labels_extendedLeg_52.npy"), np.array(labels_extendedLeg, dtype=object))  # Save files to results path
np.save(os.path.join(path_results, "labels_stiffMovement_52.npy"), np.array(labels_stiffMovement, dtype=object))
np.save(os.path.join(path_results, "imu_data_lengths.npy"), np.array(imu_data_lengths, dtype=object))
np.savetxt(
    os.path.join(path_results, "labels_extendedLeg_52.txt"),
    np.array(labels_extendedLeg, dtype=object),
    fmt="%s",  # All elements formatted as strings
    delimiter="\t",  # Tab-separated
    header="behavior_label\tbehavior_label_5s\tstart_frames_single",
    comments=""
)
np.savetxt(
    os.path.join(path_results, "labels_stiffMovement_52.txt"),
    np.array(labels_stiffMovement, dtype=object),
    fmt="%s",  # All elements formatted as strings
    delimiter="\t",  # Tab-separated
    header="behavior_label\tbehavior_label_5s",
    comments=""
)
np.savetxt(
    os.path.join(path_results, "imu_data_lengths.txt"),
    np.array(imu_data_lengths, dtype=object),
    fmt="%s",  # All elements formatted as strings
    delimiter="\t",  # Tab-separated
)
print(f"Saved successfully to: {path_results}")