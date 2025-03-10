import numpy as np
import qmt
import imt
import pandas as pd
import quaternion_angle as qa
from pathlib import Path
import os

# LOAD DATA
path_data = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/"
folder_data_imu = path_data + "Dict_Frames/"
folder_data_cam = path_data + "Knee_angle2d/"
filenames_imu = [str(f) for f in Path(folder_data_imu).rglob('*') if f.is_file()]  # all filnames in array
filenames_cam = [str(f) for f in Path(folder_data_cam).rglob('*') if f.is_file()]
sensors = ["S0333", "S1094", "S0593", "S0994", "S0477"] 
sequences = pd.read_csv(path_data + "sequences.txt", delimiter="\t", header=None, index_col=0)
# 1 und 41 ohne Sternum,25 rot, 26 S0194 fehlt, 53 keine Kamera datei, 35 S0994 fehlt, 18 fehlt, ab 54]
seq_cam_start = sequences.iloc[:, 0:1].values.ravel()  # CAM start frame
seq_imu_start = sequences.iloc[:, 1:2].values.ravel()   # IMU start frame
seq_names = sequences.index.values              # Sequenz Namen
seq_filter = np.array(["S0101", "S0125", "S0126", "S0135_01", "S0141_01", "S0153_01"])
seq_filter = np.append(seq_filter, seq_names[54:])

Hz = 52             # IMU sample rate
Hz_resample = 30    # Kamera sample rate
Ts=0.01             # needed sample rate for RING 100 Hz

qhat_all = []
extras_all = []
#angle_kr_all = []
#angle_kl_all = []
qa_kr_all = []
qa_kl_all = []
angle_cam_kr_all = []
angle_cam_kl_all = []
ts_resample_all = []
ts_camera2d_all = []

for i_seq in range(3):#len(filenames_imu)):
    if seq_names[i_seq] not in seq_filter:
        # LOAD DATA
        data_imu_seq = np.load(filenames_imu[i_seq], allow_pickle=True).item()
        data_cam_both_seq = np.loadtxt(filenames_cam[i_seq], delimiter=",") # Kniewinkel aus Videodaten beide Beine
        data_cam_kr_seq = data_cam_both_seq[:,1][seq_cam_start[i_seq]:]      # [:,1] nur Knie rechts, [:,0] nur Knie links
        data_cam_kl_seq = data_cam_both_seq[:,0][seq_cam_start[i_seq]:]      # [i:] alles vor index i entfernen

        # PREPARE DATA
        imu_data = {
            i: dict(acc=data_imu_seq[sensors[i]]["acc"][seq_imu_start[i_seq]:], gyr=data_imu_seq[sensors[i]]["gyr_rad"][seq_imu_start[i_seq]:])
            for i in range(len(sensors))  
            }
        imu_data = imt.utils.resample(imt.utils.crop_tail(imu_data, Hz), Hz, 1/Ts)

        imu_data[0] = dict(
            acc=qmt.rotate(qmt.quatFromAngleAxis(-np.pi, [0, 0, 1]), imu_data[0]["acc"]),
            gyr=qmt.rotate(qmt.quatFromAngleAxis(-np.pi, [0, 0, 1]), imu_data[0]["gyr"]),
        )
        
        # ESTIMATE ORIENTATIONS
        rel_method = imt.methods.RING(axes_directions=np.array([1.0, 0, 0]))
        graph = [-1, 0, 1, 0, 3]
        solver = imt.Solver(graph, [imt.methods.VQF(offline=True)] +
                            #[imt.wrappers.JointTracker1D(rel_method)]*4, 
                            [rel_method]*4, 
                            Ts=Ts)
        qhat, extras = solver.step(imu_data)
        qhat_all.append(qhat)
        extras_all.append(extras)

        q1 = qhat[1]
        q2 = qhat[2] 
        q3 = qhat[3] 
        q4 = qhat[4] 
        # Compute angles
        qa_kr = qa.quaternion_angle(q1, q2)
        qa_kl = qa.quaternion_angle(q3, q4)  

        # Extract timesteps
        T = qhat[0].shape[0]
        ts = np.round(np.arange(T)*Ts, 2)
        # Extract measruement values
        #angle_kr = -np.rad2deg(extras[2]["joint_angle_rad"])
        #angle_kl = -np.rad2deg(extras[4]["joint_angle_rad"])

        ts_camera2d = np.round(np.arange(len(data_cam_kr_seq))*(1/Hz_resample), 3) # timesteps for camera data with camera sample rate
        # DOWNSAMPLE IMU DATA KNEE ANGLE RATE TO CAMERA SAMPLE RATE
        factor = (1/Ts)/Hz_resample
        indices = np.round(np.arange(0, len(ts), factor), 1)  # Select every nth index
        ts_resample = np.round(np.interp(indices, np.arange(len(ts)), ts), 3)
        #angle_kr_resample = np.interp(indices, np.arange(len(ts)), angle_kr)
        #angle_kl_resample = np.interp(indices, np.arange(len(ts)), angle_kl)
        qa_kr_resample = np.interp(indices, np.arange(len(ts)), qa_kr)  
        qa_kl_resample = np.interp(indices, np.arange(len(ts)), qa_kl)  

        #angle_kr_all.append(angle_kr_resample)
        #angle_kl_all.append(angle_kl_resample)
        qa_kr_all.append(qa_kr_resample)
        qa_kl_all.append(qa_kl_resample)
        angle_cam_kr_all.append(data_cam_kr_seq)
        angle_cam_kl_all.append(data_cam_kl_seq)
        ts_resample_all.append(ts_resample)
        ts_camera2d_all.append(ts_camera2d)

# SAVE RESULTS
path_results = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/imt-main/examples/results" 
os.makedirs(path_results, exist_ok=True) # check if directory exists
np.save(os.path.join(path_results, "Angles_IMU_RING_Knee_Left.npy"), np.array(qa_kl_all, dtype=object))  # Save files to results path
np.save(os.path.join(path_results, "Angles_IMU_RING_Knee_Right.npy"), np.array(qa_kr_all, dtype=object))
np.save(os.path.join(path_results, "Angles_CAM_Knee_Left.npy"), np.array(angle_cam_kl_all, dtype=object))
np.save(os.path.join(path_results, "Angles_CAM_Knee_Right.npy"), np.array(angle_cam_kr_all, dtype=object))
np.save(os.path.join(path_results, "Timesteps_RING.npy"), np.array(ts_resample_all, dtype=object))
np.save(os.path.join(path_results, "Timesteps_CAM.npy"), np.array(ts_camera2d_all, dtype=object))
np.save(os.path.join(path_results, "Sequences_filter.npy"), np.array(seq_filter, dtype=object))
np.save(os.path.join(path_results, "Sequences_names.npy"), np.array(seq_names, dtype=object))
print(f"Saved successfully to: {path_results}")