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
addpath(genpath('C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/MAT'))
%% load the data
IMU=load('S0133_dict_frame.mat');
fields = fieldnames(IMU);
IMU = IMU.(fields{1});
% Daten transformieren um an KOS anzupassen
% Transformationsmatrix
T = [1 0 0; 0 1 0; 0 0 1];
%T = [1 0 0; 0 -1 0; 0 0 1];   % T-Pose
%T = [0 0 -1; 0 -1 0; 1 0 0]; % I-Pose X
%T = [0 -1 0; -1 0 0; 0 0 1];   % x distal, z hoch,  rechts hand regel  
%T = [0 -1 0; 1 0 0; 0 0 -1];   % x distal, z runter,  links hand regel 
%T = [0 -1 0; 0 0 1; 1 0 0];     % X: distal, Y: forwards, Z: lateral
%T = [0 -1 0; 1 0 0; 0 0 1];

ua.acc=(T * IMU.S1094.acc')';
ua.gyr=(T * IMU.S1094.gyr_rad')';
fa.acc=(T * IMU.S0593.acc')';
fa.gyr=(T * IMU.S0593.gyr_rad')';
%ua.acc=(T * IMU.S0994.acc')';              %linkes Knie
%ua.gyr=(T * IMU.S0994.gyr_rad')';
%fa.acc=(T * IMU.S0477.acc')';
%fa.gyr=(T * IMU.S0477.gyr_rad')';

% parameters
freq = 52; % Hz
gyr_noise = 0.017;   
con_acc_rts_acc = 0.02; % Increased from 0.02 (elbow) due to soft tissue artifacts
con_acc_rts_dof = 0.01;
con_dof_rts_dof = 0.02; % Increased from 0.02 (elbow) to account for slight natural varus/valgus


%% solve for translation alignment
[rUA2,rFA] = lev_calc(ua.acc,ua.gyr,fa.acc,fa.gyr,1/freq);

%% solve for rotational alignment
[~,~,ua.quat_s,fa.quat_s] = mekf_acc_s(ua,fa,freq,gyr_noise, con_acc_rts_acc, rUA2,rFA); 
[q1_imu, q2_imu,~, ~,er] = laidig_knee_align(ua.quat_s,ua.gyr(3:end-2,:),fa.quat_s,fa.gyr(3:end-2,:),200);

%% run filter
% multiplicative kalman smoother with linear acceleration and dof constraint
kn_rts_dof = mekf_knee_acc_s(ua,fa,freq,gyr_noise, con_acc_rts_dof, con_dof_rts_dof, rUA2,rFA,q1_imu,q2_imu); 
eul_kn_rts_dof = rad2deg(quatToEuler(kn_rts_dof));

%% quat angle
norms = sqrt(sum(kn_rts_dof.^2, 2));
kn_rts_dof_norm = kn_rts_dof ./ norms;
angles_rad = 2 * acos(kn_rts_dof_norm(:,1));
angles_deg = angles_rad * (180/pi);

%% Plot data
dt = 1/freq;
t = (0:length(angles_deg)-1)*dt;

subplot(2,2,1);
plot(t, eul_kn_rts_dof(:,1)); 
ylabel('rotation');
xticks(0:25:max(t));

subplot(2,2,2);
plot(t, eul_kn_rts_dof(:,2)); 
ylabel('flexion'); 
xticks(0:25:max(t));

subplot(2,2,3);
plot(t, eul_kn_rts_dof(:,3)); 
ylabel('deviation');
xticks(0:25:max(t));

subplot(2,2,4)
plot(t, angles_deg);
xlabel('Time (s)');
ylabel('joint angle');
xticks(0:25:max(t));