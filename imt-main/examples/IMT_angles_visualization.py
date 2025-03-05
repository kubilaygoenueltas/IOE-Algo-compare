import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

results_path = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/imt-main/examples/results" 
ring_angles_kl = np.load(os.path.join(results_path, "Angles_IMU_RING_Knee_Left.npy"), allow_pickle=True)
ring_angles_kr = np.load(os.path.join(results_path, "Angles_IMU_RING_Knee_Right.npy"), allow_pickle=True)
cam_angles_kl = np.load(os.path.join(results_path, "Angles_CAM_Knee_Left.npy"), allow_pickle=True)
cam_angles_kr = np.load(os.path.join(results_path, "Angles_CAM_Knee_Right.npy"), allow_pickle=True)
ts_ring = np.load(os.path.join(results_path, "Timesteps_RING.npy"), allow_pickle=True)
ts_cam = np.load(os.path.join(results_path, "Timesteps_CAM.npy"), allow_pickle=True)

# Create the main Tkinter window
root = tk.Tk()
root.title("Scrollable Matplotlib Plot")

# Create a canvas with a scrollbar
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas)

# Attach the frame to the canvas
frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)

# Create the Matplotlib figure
fig = Figure(figsize=(15, 80))  # Adjust the size to make it scrollable
axes = fig.subplots(len(ring_angles_kl), 3)  # Create subplots dynamically

for i in range(len(ring_angles_kl)):
#for i in range(2):
    # Diagramm links
    axes[i, 0].plot(ts_ring[i], ring_angles_kl[i], label="[RING]")
    axes[i, 0].plot(ts_cam[i], cam_angles_kl[i], label="[CAM]")
    axes[i, 0].set_title("Kniewinkelverlauf links")
    axes[i, 0].legend()
    axes[i, 0].grid()
    axes[i, 0].set_ylabel("Knee Angle [deg]")
    axes[i, 0].set_xlabel("Time [s]")

    # Diagramm mitte
    axes[i, 1].plot(ts_ring[i], ring_angles_kr[i], label="[RING]")
    axes[i, 1].plot(ts_cam[i], cam_angles_kr[i], label="[CAM]")
    axes[i, 1].set_title("Kniewinkelverlauf rechts")
    axes[i, 1].legend()
    axes[i, 1].grid()
    axes[i, 1].set_ylabel("Knee Angle [deg]")
    axes[i, 1].set_xlabel("Time [s]")



    smoothed_arr = pd.Series(ring_angles_kl[i]).rolling(window=100, center=True).mean().to_numpy()
    smoothed_arr2 = pd.Series(cam_angles_kl[i][:len(ring_angles_kl[i])]).rolling(window=100, center=True, min_periods=1).mean().to_numpy()
    try:
        imu_to_camera2d_diff = smoothed_arr - smoothed_arr2
    except ValueError as e:
        imu_to_camera2d_diff = np.zeros(len(smoothed_arr))  # Assign None or another default value

    smoothed_arr3 = pd.Series(ring_angles_kr[i]).rolling(window=100, center=True).mean().to_numpy()
    smoothed_arr4 = pd.Series(cam_angles_kr[i][:len(ring_angles_kr[i])]).rolling(window=100, center=True, min_periods=1).mean().to_numpy()
    try:
        imu_to_camera2d_diff2 = smoothed_arr3 - smoothed_arr4
    except ValueError as e:
        imu_to_camera2d_diff2 = np.zeros(len(smoothed_arr3))  # Assign None or another default value

    # Diagramm rechts
    axes[i, 2].plot(ts_ring[i], imu_to_camera2d_diff, label="Knie links", color="m")
    axes[i, 2].plot(ts_ring[i], imu_to_camera2d_diff2, label="Knie rechts", color="r")
    axes[i, 2].set_title("Difference RING to CAM")
    axes[i, 2].legend()
    axes[i, 2].grid()
    axes[i, 2].set_ylabel("Knee Angle [deg]")
    axes[i, 2].set_xlabel("Time [s]")

# Embed Matplotlib figure into Tkinter canvas
canvas_widget = FigureCanvasTkAgg(fig, master=frame)
canvas_widget.get_tk_widget().pack()

# Pack canvas and scrollbar into the Tkinter window
canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

# Run the Tkinter event loop
root.mainloop()