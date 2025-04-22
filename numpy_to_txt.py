### Created with ChatGPT model 03-mini-high
import os
import numpy as np

def convert_numpy_to_txt(src_dir, dst_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Iterate over files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith('.npy'):
            numpy_path = os.path.join(src_dir, filename)
            # Load the .npy file (allow_pickle=True for dictionaries)
            array_data = np.load(numpy_path, allow_pickle=True)
            
            # Use the file name (without extension) as the variable name
            var_name = os.path.splitext(filename)[0]
            output_filename = var_name + '.txt'
            output_path = os.path.join(dst_dir, output_filename)
            
            # Handle different cases of array_data:
            if isinstance(array_data, np.ndarray):
                if array_data.ndim == 0:  # 0D array (scalar)
                    with open(output_path, 'w') as f:
                        f.write(str(array_data.item()))  # Save as plain text
                elif array_data.size == 1 and isinstance(array_data.item(), dict):  # Dict inside a 1x1 array
                    with open(output_path, 'w') as f:
                        f.write(str(array_data.item()))  # Save dict as string
                else:  # 1D/2D array
                    np.savetxt(output_path, array_data, fmt='%s', delimiter=',')
            else:  # Non-array data (e.g., direct scalar or dict)
                with open(output_path, 'w') as f:
                    f.write(str(array_data))
            
            print(f"Converted {filename} to {output_filename}")


if __name__ == '__main__':
    src_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/Dict_Frames'
    dst_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/TXT'
    convert_numpy_to_txt(src_directory, dst_directory)