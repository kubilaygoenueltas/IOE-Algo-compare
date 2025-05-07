import os
import numpy as np

def convert_numpy_to_txt_bioslam(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    sensor_order = [
        ('S0333', 'acc'), ('S0333', 'gyr_rad'),         # Sternum                   	column 1-3, 4-6
        ('S1094', 'acc'), ('S1094', 'gyr_rad'),         # rechter Oberschenkel          column 7-9, 10-12
        ('S0593', 'acc'), ('S0593', 'gyr_rad'),         # rechter Unterschenkel         column 13-15, 16-18
        ('S0477', 'acc'), ('S0477', 'gyr_rad'),         # linker Unterschenkel          column 19-21, 22-24
        ('S0994', 'acc'), ('S0994', 'gyr_rad'),         # linker Oberschenkel           column 25-27, 28-30
    ]

    for filename in os.listdir(src_dir):
        if filename.endswith('.npy'):
            numpy_path = os.path.join(src_dir, filename)
            try:
                data = np.load(numpy_path, allow_pickle=True).item()
                collected = []
                n_rows = None
                skip_file = False

                for sensor, signal in sensor_order:
                    if sensor in data and signal in data[sensor]:
                        sensor_data = data[sensor][signal]
                        if not isinstance(sensor_data, np.ndarray) or np.isnan(sensor_data).any():
                            print(f"↪ Skipping sensor {sensor}/{signal} in {filename} (contains NaN)")
                            sensor_data = None
                        else:
                            if n_rows is None:
                                n_rows = sensor_data.shape[0]
                            elif sensor_data.shape[0] != n_rows:
                                print(f"✘ Skipped {filename} — inconsistent row count for {sensor}/{signal}")
                                skip_file = True
                                break
                    else:
                        print(f"↪ Skipping sensor {sensor}/{signal} in {filename} (missing)")
                        sensor_data = None
                    
                    # Add valid data or fill with NaNs (3 columns per sensor part)
                    if sensor_data is not None:
                        collected.append(sensor_data)
                    else:
                        if n_rows is None:
                            skip_file = True  # Can't determine size for placeholders yet
                            break
                        collected.append(np.full((n_rows, 3), np.nan))

                if skip_file:
                    print(f"✘ Skipped {filename} — could not complete due to missing shape info.")
                    continue

                combined_data = np.hstack(collected)

                var_name = os.path.splitext(filename)[0]
                output_filename = var_name + '.txt'
                output_path = os.path.join(dst_dir, output_filename)

                np.savetxt(output_path, combined_data, fmt='%.8e', delimiter='\t')
                print(f"✔ Converted: {filename}")

            except Exception as e:
                print(f"✘ Skipped {filename} — error: {e}")

if __name__ == '__main__':
    src_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/Dict_Frames'
    dst_directory = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/TXT_bioslam'
    convert_numpy_to_txt_bioslam(src_directory, dst_directory)
