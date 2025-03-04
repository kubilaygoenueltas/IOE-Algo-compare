import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
from pathlib import Path

def quaternion_angle(q1_array, q2_array):
    """
    Computes the angle between two arrays of quaternions.
    
    Parameters:
    - q1_array: (N, 4) NumPy array of quaternions (w, x, y, z)
    - q2_array: (N, 4) NumPy array of quaternions (w, x, y, z)
    
    Returns:
    - angles: (N,) array of angles in degrees
    """
    # Ensure they are unit quaternions
    q1_array = q1_array / np.linalg.norm(q1_array, axis=1, keepdims=True)
    q2_array = q2_array / np.linalg.norm(q2_array, axis=1, keepdims=True)
    
    # Compute the dot product of corresponding quaternions
    dot_products = np.einsum('ij,ij->i', q1_array, q2_array)
    
    # Clamp values to the valid range to avoid numerical errors
    #dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Compute the angle (in radians) using the arccos function
    angles_rad = 2 * np.arccos(dot_products)
    
    # Convert to degrees
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg

def count_elements(folder_path):
    folder = Path(folder_path)
    num_files = sum(1 for f in folder.rglob('*') if f.is_file())
    num_folders = sum(1 for f in folder.rglob('*') if f.is_dir())
    return num_files, num_folders

def load_files_by_type(folder_path, file_extension):
    loaded_data = {}
    for file_path in Path(folder_path).rglob(f"*.{file_extension}"):
        if file_extension == "npy":
            loaded_data[file_path.name] = np.load(file_path, allow_pickle=True).item()
        elif file_extension == "csv":
            loaded_data[file_path.name] = pd.read_csv(file_path)
        elif file_extension == "txt":
            loaded_data[file_path.name] = np.loadtxt(file_path, delimiter=",")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_data[file_path.name] = f.read()
    return loaded_data
