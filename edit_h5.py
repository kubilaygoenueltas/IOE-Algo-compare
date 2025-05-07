import h5py
import numpy as np

# Paths
h5_file_path = r'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/bioslam/bioslam/test/data/20170411-154746-Y1_TUG_6 _edit.h5'
txt_file_path = r'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/TXT_bioslam/S0133_dict_frame.txt'

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

# Step 3: Write to HDF5 with resizing
with h5py.File(h5_file_path, 'r+') as h5file:
    for sensor_id, (acc_cols, gyr_cols) in sensor_mapping.items():
        acc_data = new_data[:, acc_cols]
        gyr_data = new_data[:, gyr_cols]

        acc_path = f'/Sensors/{sensor_id}/Accelerometer'
        gyr_path = f'/Sensors/{sensor_id}/Gyroscope'

        # Update Accelerometer
        if acc_path in h5file:
            # Resize if shape mismatch
            if h5file[acc_path].shape != acc_data.shape:
                if h5file[acc_path].maxshape[0] is None:
                    h5file[acc_path].resize(acc_data.shape)
                    print(f"Resized {acc_path} to {acc_data.shape}")
                else:
                    print(f"Cannot resize {acc_path} — maxshape is fixed.")
                    continue
            h5file[acc_path][...] = acc_data
            print(f"Accelerometer for sensor {sensor_id} updated.")
        else:
            print(f"{acc_path} not found.")

        # Update Gyroscope
        if gyr_path in h5file:
            if h5file[gyr_path].shape != gyr_data.shape:
                if h5file[gyr_path].maxshape[0] is None:
                    h5file[gyr_path].resize(gyr_data.shape)
                    print(f"Resized {gyr_path} to {gyr_data.shape}")
                else:
                    print(f"Cannot resize {gyr_path} — maxshape is fixed.")
                    continue
            h5file[gyr_path][...] = gyr_data
            print(f"Gyroscope for sensor {sensor_id} updated.")
        else:
            print(f"{gyr_path} not found.")
