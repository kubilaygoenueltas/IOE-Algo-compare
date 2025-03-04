import numpy as np
from scipy.spatial.transform import Rotation as R

# Funktion zur Berechnung des Winkels zwischen zwei Vektoren
def calculate_angle1D(v1, v2):
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)  # Clipping für numerische Stabilität
    angles = np.arccos(cos_theta)  # Winkel in Radiant
    return np.degrees(angles)  # Umrechnung in Grad

def compute_resultant_angle(joint_angles):
    """
    Compute the resultant knee joint angle from flexion/extension,
    adduction/abduction, and internal/external rotation angles.
    
    Parameters:
        joint_angles (numpy.ndarray): A 3D joint angle array with shape (N, 3) where columns are:
                                      [flexion/extension, adduction/abduction, internal/external rotation]
    
    Returns:
        numpy.ndarray: An array of resultant angles for each row in input.
    """
    return np.linalg.norm(joint_angles, axis=1)

def resultant_overall_angle(angles, degrees=True, seq = 'zyx'):
    """
    Computes the resultant overall rotation angle from a 3D array of Euler angles.
    
    Parameters:
    angles (np.ndarray): A Nx3 array where each row represents [yaw, pitch, roll]
    degrees (bool): If True, input angles are in degrees (default: True)
    
    Returns:
    np.ndarray: The resultant overall rotation angles in degrees.
    """
    # Ensure input is a numpy array
    angles = np.asarray(angles)
    
    # Create the rotation objects
    rotation = R.from_euler(seq, angles, degrees=degrees)
    
    # Get the equivalent rotation matrices
    R_mats = rotation.as_matrix()
    
    # Compute the resultant rotation angles using the trace method
    theta_total = np.arccos((np.trace(R_mats, axis1=1, axis2=2) - 1) / 2)
    
    # Convert to degrees if required
    if degrees:
        theta_total = np.degrees(theta_total)
    
    return theta_total

def rotate_vectors(vectors, theta, axis):
    """
    Rotiert ein Array von 3D-Vektoren um eine angegebene Achse (X, Y oder Z).
    :param vectors: NumPy-Array der Form (n, 3), wobei jede Zeile ein 3D-Vektor ist
    :param theta: Rotationswinkel in Radiant
    :param axis: Achse um die rotiert wird ('x', 'y' oder 'z')
    :return: Rotierte Vektoren als NumPy-Array
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Achse muss 'x', 'y' oder 'z' sein")
    
    return np.round(np.dot(vectors, rotation_matrix.T), 5)  # Transponierte für richtige Multiplikation