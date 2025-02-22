import numpy as np
import qmt
import imt
import ring
from imt.utils.view import view
from imt.utils.view import VisOptions
import matplotlib.pyplot as plt
import tree
import pandas as pd

# LOAD DATA
file = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/IMU/Dict_Frames/S0133_dict_frame.npy"  # noqa: E501
file_camera2d = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/IMU/Dict_Frames/S0133_Knee_angle_camera2d.txt"
sensors = ["S0333", "S1094", "S0593", "S0994", "S0477"] # 5 Sensoren
#sensors = ["S1094", "S0593", "S0994", "S0477"] # ohne Sternum, beide Knie
#sensors = ["S1094", "S0593"]   # 2 Sensoren, nur Knie rechts
data = np.load(file, allow_pickle=True).item()
data_camera2d_both =np.loadtxt(file_camera2d, delimiter=",") # Kniewinkel aus Videodaten beide Beine
data_camera2d_kr = data_camera2d_both[:,1][17:]      # [:,1] nur Knie rechts, [:,0] nur Knie links
data_camera2d_kl = data_camera2d_both[:,0][17:]      # [17:] alles vor index 17 entfernen
Hz = 52             # IMU sample rate
Hz_resample = 30    # Kamera sample rate
Ts=0.01             # needed sample rate for RING 100 Hz

# CROP DATASET TO GET ONLY N Seconds
#n = 50   # Wie viele Sekunden des Datensatz betrachtet
#ts_crop = int(len(data[sensors[0]]["acc"]) - n/(1/Hz)) 
    
# PREPARE DATA
imu_data = {
    i: dict(acc=data[sensors[i]]["acc"], gyr=data[sensors[i]]["gyr_rad"])
    #i: dict(acc=data[sensors[i]]["acc"][:-ts_crop], gyr=data[sensors[i]]["gyr_rad"][:-ts_crop])    #cropped Data
    for i in range(len(sensors))  
}
imu_data = imt.utils.resample(imt.utils.crop_tail(imu_data, Hz), Hz, 1/Ts)      # resample for RING algorithm to 100 Hz

imu_data[0] = dict(
    acc=qmt.rotate(qmt.quatFromAngleAxis(-np.pi, [0, 0, 1]), imu_data[0]["acc"]),
    gyr=qmt.rotate(qmt.quatFromAngleAxis(-np.pi, [0, 0, 1]), imu_data[0]["gyr"]),
)       # nur bei 5 Sensoren für Sternum

# ESTIMATE ORIENTATIONS
rel_method = imt.methods.RING(axes_directions=np.array([1.0, 0, 0]))
graph = [-1, 0, 1, 0, 3]
#graph = [-1, 0, -1, 2] # ohne Sternum, beide Knie
#graph = [-1, 0]    # nur Knie rechts
qhat, extras = imt.Solver(graph,
                          [imt.methods.VQF(offline=True)] + 
                          [imt.wrappers.JointTracker1D(rel_method)]*4,
                          Ts=Ts).step(
                              imu_data
                              )

# Extract timesteps
T = qhat[0].shape[0]
ts = np.round(np.arange(T)*Ts, 2)

# Extract measruement values
angle_kr = -np.rad2deg(extras[2]["joint_angle_rad"])
#angle_kr = np.clip(angle_kr, 0, 180)   # clip values below 0 and above 180, anatomically not possible
angle_kl = -np.rad2deg(extras[4]["joint_angle_rad"])
#angle_kl = np.clip(angle_kl, 0, 180)   #

# DOWNSAMPLE IMU DATA KNEE ANGLE RATE TO CAMERA SAMPLE RATE
factor = (1/Ts)/Hz_resample
indices = np.round(np.arange(0, len(ts), factor), 1)  # Select every nth index
ts_resample = np.interp(indices, np.arange(len(ts)), ts)
angle_kr_resample = np.interp(indices, np.arange(len(ts)), angle_kr)
angle_kl_resample = np.interp(indices, np.arange(len(ts)), angle_kl)

# EXTRACT TIMESTEPS FOR CAMERA DATA
#ts_camera2d = np.arange(len(data_camera2d_kr))*(T/len(data_camera2d_kr))*Ts                                # timesteps for camera data with 100 Hz sample rate
#ts_camera2d = np.arange(len(data_camera2d_kr))*(len(ts_resample)/len(data_camera2d_kr))*(1/Hz_resample)    # ?
ts_camera2d = np.arange(len(data_camera2d_kr))*(1/Hz_resample)                                              # timesteps for camera data with camera sample rate
# Interpolate the values of camera to match the new timesteps ts
#data_camera2d_new = np.interp(ts, ts_camera2d, data_camera2d)
"""
# PLOT KNEE ANGLES OVER TIME
plt.plot(ts_resample, angle_kr_resample, label="Knie rechts") 
plt.plot(ts_camera2d, data_camera2d_kr, label="Knie rechts camera2d")
plt.grid()
plt.legend()
plt.ylabel("Knee Angle [deg]")
plt.xlabel("Time [s]")
plt.show()

plt.plot(ts_resample, angle_kl_resample, label="Knie links")
plt.plot(ts_camera2d, data_camera2d_kl, label="Knie links camera2d")
plt.grid()
plt.legend()
plt.ylabel("Knee Angle [deg]")
plt.xlabel("Time [s]")
plt.show()"""

# PLOT IMU TO CAMERA KNEE ANGLE DIFFERENCE
smoothed_arr = pd.Series(angle_kr_resample).rolling(window=100, center=True).mean().to_numpy()
smoothed_arr2 = pd.Series(data_camera2d_kr[:len(angle_kr_resample)]).rolling(window=100, center=True, min_periods=1).mean().to_numpy()
imu_to_camera2d_diff = smoothed_arr - smoothed_arr2
#for i in range(len(smoothed_arr)):
#    if smoothed_arr[i]<smoothed_arr2[i]:
#        imu_to_camera2d_diff[i] = smoothed_arr2[i] - smoothed_arr[i]

#imu_to_camera2d_diff = angle_kr_resample - data_camera2d_kr[:len(angle_kr_resample)]
#for i in range(len(angle_kr_resample)):
#    if angle_kr_resample[i] < data_camera2d_kr[:len(angle_kr_resample)][i]:
#        imu_to_camera2d_diff[i] = data_camera2d_kr[:len(angle_kr_resample)][i] - angle_kr_resample[i]
#smoothed_arr_itcd = pd.Series(imu_to_camera2d_diff).rolling(window=100, center=True).mean().to_numpy()

smoothed_arr_kl = pd.Series(angle_kl_resample).rolling(window=100, center=True).mean().to_numpy()
smoothed_arr2_kl = pd.Series(data_camera2d_kl[:len(angle_kl_resample)]).rolling(window=100, center=True, min_periods=1).mean().to_numpy()
imu_to_camera2d_diff_kl = smoothed_arr_kl - smoothed_arr2_kl
"""
plt.plot(ts_resample, imu_to_camera2d_diff, label="Difference IMU to Camera [Knie rechts]")
plt.plot(ts_resample, imu_to_camera2d_diff_kl, label="Difference IMU to Camera [Knie links]")
plt.grid()
plt.legend()
plt.ylabel("Knee Angle [deg]")
plt.xlabel("Time [s]")
plt.show()
import matplotlib.pyplot as plt
import numpy as np"""

# Beispieldaten
x13 = ts_resample
x24 = ts_camera2d
y1 = angle_kr_resample
y2 = data_camera2d_kr
y3 = angle_kl_resample
y4 = data_camera2d_kl
y5 = imu_to_camera2d_diff
y6 = imu_to_camera2d_diff_kl

# Erstellen der Figur mit 2x2 Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Erstes Diagramm (oben links)
axes[0, 0].plot(x13, y1, label="Knie rechts [IMU]")
axes[0, 0].plot(x24, y2, label="Knie rechts [camera2d]")
axes[0, 0].set_title("Kniewinkelverlauf rechts")
axes[0, 0].legend()
axes[0, 0].grid()
axes[0, 0].set_ylabel("Knee Angle [deg]")
axes[0, 0].set_xlabel("Time [s]")

# Zweites Diagramm (oben rechts)
axes[1, 0].plot(x13, y3, label="Knie links [IMU]", color="m")
axes[1, 0].plot(x24, y4, label="Knie links [camera2d]", color="g")
axes[1, 0].set_title("Kniewinkelverlauf links")
axes[1, 0].legend()
axes[1, 0].grid()
axes[1, 0].set_ylabel("Knee Angle [deg]")
axes[1, 0].set_xlabel("Time [s]")

# Drittes Diagramm (unten links)
axes[0, 1].plot(x13, y5, color="r")
axes[0, 1].set_title("Difference IMU to Camera [Knie rechts]")
axes[0, 1].legend()
axes[0, 1].grid()
axes[0, 1].set_ylabel("Knee Angle [deg]")
axes[0, 1].set_xlabel("Time [s]")

# Viertes Diagramm (unten rechts)
axes[1, 1].plot(x13, y6, color="r")
axes[1, 1].set_title("Difference IMU to Camera [Knie links]")
axes[1, 1].legend()
axes[1, 1].grid()
axes[1, 1].set_ylabel("Knee Angle [deg]")
axes[1, 1].set_xlabel("Time [s]")

# Automatische Anpassung der Abstände
plt.tight_layout()
plt.show()