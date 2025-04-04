% Example script for solving for translation and rotational alignment, and using these
% paramenters to run mekf-acc, rts-acc, mekf-dof, and rts-dof for the elbow joint 
% must specify ua.acc, ua.gyr, fa.acc, fa.gyr (accelerometer and gyro
% measurements for the upper arm and forearm IMU). gyro in rad/s, acc in m/s^2
% Assums right-handed coordinate frame with 
% +x points distally parallel with the bone, +y point forward in the 'I-pose' if
% IMU attached to the right side. 

% Author: Howard Chen, PhD
% Affiliation: University of Alabama in Huntsville
clear all;
clc;

addpath(genpath('./filters')); 
addpath(genpath('./alignment'));
addpath(genpath('./kinematics')); 
addpath(genpath('C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/MAT'))
%% load the data
IMU=load('S0116_dict_frame.mat');
fields = fieldnames(IMU);
IMU = IMU.(fields{1});
% Daten transformieren um an KOS anzupassen
% Transformationsmatrix
T = [0 1 0; 0 0 1; -1 0 0];
%T = [1 0 0; 0 1 0; 0 0 1];
% Transformation: jeder Zeile wird der neue Punkt zugeordnet. um Dimensionen anzupassen, muss transponiert werden.
data = (T * IMU.S1094.acc')';

ua.acc=(T * IMU.S1094.acc')';
ua.gyr=(T * IMU.S1094.gyr_rad')';
fa.acc=(T * IMU.S0593.acc')';
fa.gyr=(T * IMU.S0593.gyr_rad')';

% parameters
freq = 52; % Hz
gyr_noise = 0.005; 
con_acc_rts_acc = 0.02; 
 
%% solve for translation alignment
[rUA2,rFA] = lev_calc(ua.acc,ua.gyr,fa.acc,fa.gyr,1/freq);

%% solve for rotational alignment
[~,~,ua.quat_s,fa.quat_s] = mekf_acc_s(ua,fa,freq,gyr_noise, con_acc_rts_acc, rUA2,rFA); 
[q1_imu, q2_imu,~, ~,er] = laidig_elbow_align(ua.quat_s,ua.gyr(3:end-2,:),fa.quat_s,fa.gyr(3:end-2,:),100);

%% run filter
% multiplicative kalman smoother with linear acceleration constraint
el_rts_acc = mekf_acc_s(ua,fa,freq,gyr_noise, con_acc_rts_acc, rUA2,rFA,q1_imu,q2_imu); 
eul_el_rts_acc = rad2deg(quatToEuler(el_rts_acc));

%% quat angle
norms = sqrt(sum(el_rts_acc.^2, 2));
el_rts_acc_norm = el_rts_acc ./ norms;
angles_rad = 2 * acos(el_rts_acc_norm(:,1));
angles_deg = angles_rad * (180/pi);

%% Plot data
dt = 1/freq;
t = (0:length(angles_deg)-1)*dt;
plot(t, angles_deg);
xlabel('Time (s)');
ylabel('joint angle');
