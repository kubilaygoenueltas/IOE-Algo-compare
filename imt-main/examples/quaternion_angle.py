import numpy as np
from scipy.spatial.transform import Rotation as R

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