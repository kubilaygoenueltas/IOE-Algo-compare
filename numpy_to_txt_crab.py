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
            data = np.load(numpy_path, allow_pickle=True).item()
            
            # Get the required sensor data
            s1094_acc = data['S1094']['acc']
            s1094_gyr = data['S1094']['gyr_rad']
            s0593_acc = data['S0593']['acc']
            s0593_gyr = data['S0593']['gyr_rad']
            
            # Combine the data into the requested format
            combined_data = np.hstack((
                s1094_acc,    # Columns 1-3
                s1094_gyr,    # Columns 4-6
                s0593_acc,    # Columns 7-9
                s0593_gyr     # Columns 10-12
            ))
            
            # Use the file name (without extension) as the output name
            var_name = os.path.splitext(filename)[0]
            output_filename = var_name + '.txt'
            output_path = os.path.join(dst_dir, output_filename)
            
            # Save to text file with scientific notation and 8 decimal places
            np.savetxt(output_path, combined_data, 
                      fmt='%.8e',  # Scientific notation with 8 decimal places
                      delimiter='\t')  # Tab delimiter
            
            print(f"Converted {filename} to {output_filename}")

if __name__ == '__main__':
    src_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/Dict_Frames'
    dst_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/TXT_crab'
    convert_numpy_to_txt(src_directory, dst_directory)