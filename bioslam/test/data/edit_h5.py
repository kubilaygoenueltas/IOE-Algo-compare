import h5py
import numpy as np

# Paths
h5_file_path = r"C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare//bioslam/test/data/20170411-154746-Y1_TUG_6.h5"
txt_file_path = r"C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/bioslam/test/data/S0133_dict_frame.txt"

# Step 1: Load new accelerometer + gyroscope data
new_data = np.loadtxt(txt_file_path)  # Assumes shape (9212, 30)

# Step 2: Define sensor-to-column mapping
sensor_mapping = {
    367: ([0, 1, 2], [3, 4, 5]),       # Sternum
    91:  ([0, 1, 2], [3, 4, 5]),       # Sternum
    379: ([6, 7, 8], [9, 10, 11]),     # Right thigh
    456: ([12, 13, 14], [15, 16, 17]), # Right tibia
    482: ([12, 13, 14], [15, 16, 17]), # Right tibia
    617: ([18, 19, 20], [21, 22, 23]), # Left tibia
    640: ([18, 19, 20], [21, 22, 23]), # Left tibia
    785: ([24, 25, 26], [27, 28, 29])  # Left thigh
}

# Step 3: Write to HDF5 using original dataset sizes
with h5py.File(h5_file_path, 'r+') as h5file:
    for sensor_id, (acc_cols, gyr_cols) in sensor_mapping.items():
        acc_path = f'/Sensors/{sensor_id}/Accelerometer'
        gyr_path = f'/Sensors/{sensor_id}/Gyroscope'

        # Process Accelerometer
        if acc_path in h5file:
            original_size = h5file[acc_path].shape[0]
            acc_data = new_data[:original_size, acc_cols]  # Take only first N rows
            if h5file[acc_path].shape == acc_data.shape:
                h5file[acc_path][...] = acc_data
                print(f"Accelerometer for sensor {sensor_id} updated with {original_size} samples.")
            else:
                print(f"Shape mismatch for {acc_path}. Expected {h5file[acc_path].shape}, got {acc_data.shape}")
        else:
            print(f"{acc_path} not found.")

        # Process Gyroscope
        if gyr_path in h5file:
            original_size = h5file[gyr_path].shape[0]
            gyr_data = new_data[:original_size, gyr_cols]  # Take only first N rows
            if h5file[gyr_path].shape == gyr_data.shape:
                h5file[gyr_path][...] = gyr_data
                print(f"Gyroscope for sensor {sensor_id} updated with {original_size} samples.")
            else:
                print(f"Shape mismatch for {gyr_path}. Expected {h5file[gyr_path].shape}, got {gyr_data.shape}")
        else:
            print(f"{gyr_path} not found.")