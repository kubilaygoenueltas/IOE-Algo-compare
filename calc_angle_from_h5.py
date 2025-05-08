import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the HDF5 file
with h5py.File("C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/bioslam/test/data/example_results.h5", "r") as f:
    # Access the right knee angle components (in radians)
    rk_abad = f["Derived/RKneeAngles_AbAd"][:]
    rk_flexex = f["Derived/RKneeAngles_FlexEx"][:]
    rk_intextrot = f["Derived/RKneeAngles_IntExtRot"][:]

# Calculate the magnitude of the angle vector (in radians)
rk_resultant_rad = np.sqrt(rk_abad**2 + rk_flexex**2 + rk_intextrot**2)

# Convert to degrees
rk_resultant_deg = np.degrees(rk_resultant_rad)

# Plot the resulting angle in degrees
plt.figure(figsize=(10, 5))
plt.plot(rk_resultant_deg, color='blue', linewidth=2, label="Resulting RKnee Angle")
plt.xlabel("Time Index")
plt.ylabel("Angle Magnitude [°]")
plt.title("Resulting Right Knee Angle Over Time (Degrees)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
