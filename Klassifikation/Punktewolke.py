import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import math

# Load knee angle data
results_path_knee_angle = "../imt-main/examples/results"
ring_angles_kl = np.load(os.path.join(results_path_knee_angle, "Angles_CAM_Knee_Left.npy"), allow_pickle=True)
ring_angles_kr = np.load(os.path.join(results_path_knee_angle, "Angles_CAM_Knee_Right.npy"), allow_pickle=True)
seq_names = np.load(os.path.join(results_path_knee_angle, "Sequences_names.npy"), allow_pickle=True)

# Load labels
results_path_label = "./results"
labels_extendedLeg_100 = np.load(os.path.join(results_path_label, "labels_extendedLeg_CAM.npy"), allow_pickle=True)
import numpy as np
import os
import plotly.graph_objs as go
from IPython.display import display

# === Daten laden ===
results_path_knee_angle = "../imt-main/examples/results"
ring_angles_kl = np.load(os.path.join(results_path_knee_angle, "Angles_CAM_Knee_Left.npy"), allow_pickle=True)
ring_angles_kr = np.load(os.path.join(results_path_knee_angle, "Angles_CAM_Knee_Right.npy"), allow_pickle=True)
seq_names = np.load(os.path.join(results_path_knee_angle, "Sequences_names.npy"), allow_pickle=True)

results_path_label = "./results"
labels_extendedLeg_100 = np.load(os.path.join(results_path_label, "labels_extendedLeg_CAM.npy"), allow_pickle=True)

# === Sammellisten für Gesamtplot ===
all_kr, all_kl = [], []

# === Einzelplots ===
for i in range(len(ring_angles_kl)):
    knie_links = ring_angles_kl[i]
    knie_rechts = ring_angles_kr[i]
    labels = labels_extendedLeg_100[i]['behavior_label']

    # Nur Label-1-Punkte extrahieren
    kr_1 = [kr for kr, l in zip(knie_rechts, labels) if l == 1]
    kl_1 = [kl for kl, l in zip(knie_links, labels) if l == 1]

    # Für Gesamtplot sammeln
    all_kr.extend(kr_1)
    all_kl.extend(kl_1)

    # Einzelplot erstellen
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kr_1,
        y=kl_1,
        mode='markers',
        marker=dict(color='green'),
        name='Label 1'
    ))
    fig.update_layout(
        title=f"Punktewolke: {seq_names[i]}",
        xaxis_title="Kniewinkel rechts [°]",
        yaxis_title="Kniewinkel links [°]",
        xaxis=dict(range=[0, 180]),
        yaxis=dict(range=[0, 180]),
        height=400,
        width=800
    )
    display(fig)

# === Gesamtplot ===
fig_total = go.Figure_
