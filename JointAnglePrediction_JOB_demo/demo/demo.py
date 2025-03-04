# Import libraries
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import calculate_angles as ca

# Get the current working directory
current_workspace = os.getcwd()

# Set it as the default path
os.chdir(current_workspace)



########################################################################
# Get the arguments                                                    #
########################################################################
# Get the argument from command line
#joint, activity = sys.argv[1:3]
joint = 'Hip'
activity = 'Walking'

# Check if the arguments are valid
assert activity in ['Walking', 'Running']
assert joint in ['Hip', 'Knee', 'Ankle']

# Print the arguments
print('#' * 10 + ' Tutorial Setup ' + '#' * 10)
print(f'Activity : {activity}')
print(f'Joint    : {joint}')
print('#' * 36)




########################################################################
# Check if CUDA is available                                           #
########################################################################
print(f"\n\n Is CUDA available?: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




########################################################################
# Load model                                                           #
########################################################################
# Load model kwargs
with open(f'models/checkpoints/{activity}/{joint}_model_kwargs.pkl', 'rb') as file:
    kwargs = pickle.load(file)

# Print model kwargs
print('#' * 10 + ' Model Configuration ' + '#' * 10)
print(f'Type of the network : {kwargs["model_type"].replace("Custom", "")}')
if kwargs["model_type"] == "CustomLSTM":
    print(f'Bidirectional       : {kwargs["bidir"]}')
print(f'Number of layers    : {len(kwargs["layers"])}')
print(f'Channels of layers  : {kwargs["layers"]}')
print(f'Dropout layers      : {kwargs["dropout"]}')
print(f'Input channel       : {kwargs["inp_size"]}')
print(f'Output channel      : {kwargs["outp_size"]}')
print('#' * 41 + '\n')

# Load model
from models.pure_conv import CustomConv1D
from models.pure_lstm import CustomLSTM
model = globals()[kwargs['model_type']](**kwargs)

# Load pretrained model parameters
state_dict = torch.load(f'models/checkpoints/{activity}/{joint}_model.pt', map_location=device)
model.load_state_dict(state_dict)
model.to(device=device)
print('\n Pretrained model loaded!')




########################################################################
# Visualize the data                                                   #
########################################################################
# IMU-Joint mapping
SegJointDict = {'Hip': ['pelv', 'thigh'], 'Knee': ['thigh', 'shank'], 'Ankle': ['shank', 'foot']}

# Load IMU data
#acc1 = np.load(f'data/{activity}/{SegJointDict[joint][0]}_acc.npy')
#gyr1 = np.load(f'data/{activity}/{SegJointDict[joint][0]}_gyr.npy')
#acc2 = np.load(f'data/{activity}/{SegJointDict[joint][1]}_acc.npy')
#gyr2 = np.load(f'data/{activity}/{SegJointDict[joint][1]}_gyr.npy')

file = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/IMU/Dict_Frames/S0133_dict_frame.npy"
data33 = np.load(file, allow_pickle=True).item()
data33 = data33
data33_thigh_right = data33['S1094']
data33_shank_right = data33['S0593']
#data33_thigh_right_acc = data33_thigh_right['acc'].reshape(1, len(data33_thigh_right['acc']), 3)
#data33_thigh_right_gyr = data33_thigh_right['gyr_rad'].reshape(1, len(data33_thigh_right['gyr_rad']), 3)
#data33_shank_right_acc = data33_shank_right['acc'].reshape(1, len(data33_shank_right['acc']), 3)
#data33_shank_right_gyr = data33_shank_right['gyr_rad'].reshape(1, len(data33_shank_right['gyr_rad']), 3)
data33_thigh_right_acc = ca.rotate_vectors(ca.rotate_vectors(data33_thigh_right['acc'], theta = np.radians(270), axis = 'x'), theta = np.radians(0), axis = 'z').reshape(1, len(data33_thigh_right['acc']), 3)
data33_thigh_right_gyr = ca.rotate_vectors(ca.rotate_vectors(data33_thigh_right['gyr_rad'], theta = np.radians(270), axis = 'x'), theta = np.radians(0), axis = 'z').reshape(1, len(data33_thigh_right['gyr_rad']), 3)
data33_shank_right_acc = ca.rotate_vectors(ca.rotate_vectors(data33_shank_right['acc'], theta = np.radians(270), axis = 'x'), theta = np.radians(0), axis = 'z').reshape(1, len(data33_shank_right['acc']), 3)
data33_shank_right_gyr = ca.rotate_vectors(ca.rotate_vectors(data33_shank_right['gyr_rad'], theta = np.radians(270), axis = 'x'), theta = np.radians(0), axis = 'z').reshape(1, len(data33_shank_right['gyr_rad']), 3)

acc1 = data33_thigh_right_acc
gyr1 = data33_thigh_right_gyr
acc2 = data33_shank_right_acc
gyr2 = data33_shank_right_gyr

"""
# Visualize IMU data
fig = plt.figure(figsize=(15, 10))
for i, (acc, gyr) in enumerate(zip([acc1, acc2], [gyr1, gyr2])):
    ax_acc = fig.add_subplot(2, 2, i+1)
    ax_gyr = fig.add_subplot(2, 2, i+3)
    
    for j, axis in enumerate(['X', 'Y', 'Z']):
        ax_acc.plot(acc[0, 1000:9363, j], label=f'{SegJointDict[joint][i]} Acc {axis}')
        ax_gyr.plot(gyr[0, 1000:9363, j], label=f'{SegJointDict[joint][i]} Gyr {axis}')
    ax_acc.legend()
    ax_gyr.legend()

plt.show()
"""



########################################################################
# Preprocess IMU data                                                  #
########################################################################
# Add magnitude and concatenate all features
inputs = []
for data in [acc1, gyr1, acc2, gyr2]:
    mag = np.linalg.norm(data, axis=-1, keepdims=True)
    _data = np.concatenate((data, mag), axis=-1)
    inputs += [_data]

inputs = np.concatenate(inputs, axis=-1)
inputs = torch.from_numpy(inputs).to(device=device).float()
print(f'Input data shape: {inputs.shape}')

# Normalize input data
norm_dict = torch.load(f'models/checkpoints/{activity}/{joint}_norm_dict.pt', map_location=device)['params']
inputs = (inputs - norm_dict['x_mean']) / norm_dict['x_std']



########################################################################
# Predict joint angle                                                  #
########################################################################
from time import time

# Run inference
model.eval()
t1 = time()
pred = model(inputs)
print(pred.size())
t2 = time()

# Unnormalize prediction
pred = pred * norm_dict['y_std'] + norm_dict['y_mean']
pred = pred.detach().cpu().numpy()

print('\n\nPrediction completed! %.2f seconds taken\n'%(t2-t1))
#pred_resultant = ca.compute_resultant_angle(pred[0])
pred_resultant = ca.resultant_overall_angle(pred[0])

T = len(pred_resultant)
Ts = 1/52
ts = np.round(np.arange(T)*Ts,2)

plt.plot(ts, pred_resultant, label="Knie rechts")
plt.grid()
plt.legend()
plt.ylabel("Knee Angle [deg]")
plt.xlabel("Time [s]")
plt.show()




########################################################################
# Analyze the results                                                  #
########################################################################
# Load groundtruth
#label = np.load(f'data/{activity}/{joint}_angle.npy')
#file_camera2d = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/IMU/Knee_angle2d/S0133_Knee_angle_camera2d.txt"
#data_camera2d_both =np.loadtxt(file_camera2d, delimiter=",") # Kniewinkel aus Videodaten beide Beine
#data_camera2d_kr = data_camera2d_both[:,1][17:]      # [:,1] nur Knie rechts, [:,0] nur Knie links
#data_camera2d_kl = data_camera2d_both[:,0][17:]      # [17:] alles vor index 17 entfernen
#label = data_camera2d_kr

label = pred *0

# Berechnung des Winkels für jede Spalte (xyz)
#angles_x = ca.calculate_angle1D(array1[:, [0]], array2[:, [0]])
#angles_y = ca.calculate_angle1D(array1[:, [1]], array2[:, [1]])
#angles_z = ca.calculate_angle1D(array1[:, [2]], array2[:, [2]])

# Zusammenführen der Winkel in ein neues Array
#angles_array = np.vstack((angles_x, angles_y, angles_z)).T

# Align neutral angle (passive-pseudo calibration)
pred = pred - pred.mean(axis=1, keepdims=True) + label.mean(axis=1, keepdims=True)
"""
# Calculate RMSE
#rmse = np.sqrt(np.square(pred - label).mean(axis=1)).mean(axis=0)
rmse = np.sqrt(np.square(pred - (pred*0.9)).mean(axis=1)).mean(axis=0)

print('#'*10 + ' Root-Mean-Square-Error ' + '#'*10)
print(f'Activity          : {activity}')
print(f'Joint             : {joint}')
print('RMSE-Flex/Ext     : %.2f deg'%rmse[0])
print('RMSE-Add/Abd      : %.2f deg'%rmse[1])
print('RMSE-Int/Ext Rot  : %.2f deg'%rmse[2])
print('#'*44 + '\n\n\n')

# Visualize joint angle curve
plt.close('all')
fig = plt.figure(figsize=(7, 14))

for i, axis in enumerate(['Flex/Ext', 'Add/Abd', 'Int/Ext Rot']):
    _ax = fig.add_subplot(3, 1, i+1)
    _ax.plot(pred[0, :9363, i], label='Prediction')
    _ax.plot(label[0, :9363, i], label='Ground-Truth')
    _ax.set_ylabel(f'RMSE-{axis} (deg)', fontsize=10)
    _ax.legend()
_ax.set_xlabel('Frames (FPS:200)', fontsize=10)
plt.show()
"""