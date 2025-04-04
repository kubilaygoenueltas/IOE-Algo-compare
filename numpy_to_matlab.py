### Created with ChatGPT model 03-mini-high
import os
import numpy as np
from scipy.io import savemat

def convert_numpy_to_mat(src_dir, dst_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Iterate over files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith('.npy'):
            numpy_path = os.path.join(src_dir, filename)
            # Use allow_pickle=True in case the .npy file stores a dict inside an array
            array_data = np.load(numpy_path, allow_pickle=True)
            
            # If array_data is a 1x1 numpy object array whose single element is a dict,
            # extract that dict so that savemat creates a direct MATLAB struct.
            if isinstance(array_data, np.ndarray) and array_data.size == 1 and isinstance(array_data.item(), dict):
                array_data = array_data.item()
            
            # Use the file name (without extension) as the variable name
            var_name = os.path.splitext(filename)[0]
            mat_dict = {var_name: array_data}
            
            # Save to a .mat file
            output_filename = var_name + '.mat'
            output_path = os.path.join(dst_dir, output_filename)
            savemat(output_path, mat_dict)
            print(f"Converted {filename} to {output_filename}")


if __name__ == '__main__':
    src_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/Dict_Frames'
    dst_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/MAT'
    convert_numpy_to_mat(src_directory, dst_directory)