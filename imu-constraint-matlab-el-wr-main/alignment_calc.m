% Example script for solving for translation and rotational alignment, and using these
% paramenters to run mekf-acc, rts-acc, mekf-dof, and rts-dof for the elbow joint 
% must specify ua.acc, ua.gyr, fa.acc, fa.gyr (accelerometer and gyro
% measurements for the upper arm and forearm IMU). gyro in rad/s, acc in m/s^2
% Assums right-handed coordinate frame with 
% +x points distally parallel with the bone, +y point forward in the 'I-pose' if
% IMU attached to the right side. 

% Author: Howard Chen, PhD
% Affiliation: University of Alabama in Huntsville

% SLOW MID FAST DIFFERENT INFLUENCE ON ALGORITHM !, calibration fail
% results in wrong joint angles, intital pose (I pose) -> absolute angle
% values wrong
clear all;
clc;

addpath(genpath('./filters')); 
addpath(genpath('./alignment'));
addpath(genpath('./kinematics')); 
addpath(genpath('C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/MAT'));

% Parameters
freq = 52; % Hz
gyr_noise = 0.017;   
con_acc_rts_acc = 0.02;
con_acc_rts_dof = 0.01;
con_dof_rts_dof = 0.02;
T = [1 0 0; 0 1 0; 0 0 1];

% Get list of all MAT files in the directory
data_folder = 'C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/MAT';
mat_files = dir(fullfile(data_folder, '*.mat'));

% Open a text file for writing
output_file = 'alignment_results.txt';
fid = fopen(output_file, 'w');  % 'w' for write mode (overwrites existing file)

% Process each file in a loop
for file_idx = 1:length(mat_files)
    current_file = mat_files(file_idx).name;
    fprintf('Processing file %d of %d: %s\n', file_idx, length(mat_files), current_file);
    
    try
        % Load the data
        IMU = load(fullfile(data_folder, current_file));
        fields = fieldnames(IMU);
        IMU = IMU.(fields{1});
        
        % Extract data
        ua.acc = (T * IMU.S1094.acc')';
        ua.gyr = (T * IMU.S1094.gyr_rad')';
        fa.acc = (T * IMU.S0593.acc')';
        fa.gyr = (T * IMU.S0593.gyr_rad')';
        %ua.acc=(T * IMU.S0994.acc')';
        %ua.gyr=(T * IMU.S1094.gyr_rad')';
        %fa.acc=(T * IMU.S0477.acc')';
        %fa.gyr=(T * IMU.S0477.gyr_rad')';
        
        % Solve for translation alignment
        [rUA2, rFA] = lev_calc(ua.acc, ua.gyr, fa.acc, fa.gyr, 1/freq);
        
        % Solve for rotational alignment
        [~, ~, ua.quat_s, fa.quat_s] = mekf_acc_s(ua, fa, freq, gyr_noise, con_acc_rts_acc, rUA2, rFA); 
        [q1_imu, q2_imu, ~, ~, er] = laidig_knee_align(ua.quat_s, ua.gyr(3:end-2,:), fa.quat_s, fa.gyr(3:end-2,:), 200);
        
        % Write results to the text file (no headers, with parentheses)
        fprintf(fid, '%s\t', current_file);  % Filename
        fprintf(fid, '(%.4f,%.4f,%.4f)\t', rUA2);  % rUA2 in (x,y,z)
        fprintf(fid, '(%.4f,%.4f,%.4f)\t', rFA);   % rFA in (x,y,z)
        fprintf(fid, '(%.4f,%.4f,%.4f,%.4f)\t', q1_imu);  % q1_imu in (w,x,y,z)
        fprintf(fid, '(%.4f,%.4f,%.4f,%.4f)\n', q2_imu);  % q2_imu in (w,x,y,z) %newline
        %fprintf(fid, '(%.4f,%.4f,%.4f,%.4f)\t', q2_imu);  % q2_imu in (w,x,y,z)
        %fprintf(fid, '%.4f\n', er);  % Alignment error (no parentheses)
        
    catch ME
        fprintf('Error processing file %s: %s\n', current_file, ME.message);
        fprintf(fid, '%s\t ERROR: %s\n', current_file, ME.message);  % Log errors
    end
end

% Close the file
fclose(fid);
disp(['Results saved to: ' output_file]);