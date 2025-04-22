% Example script for solving for translation and rotational alignment, and using these
% paramenters to run mekf-acc, rts-acc, mekf-dof, and rts-dof for the wrist joint 
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
IMU=load('S0133_dict_frame.mat');
fields = fieldnames(IMU);
IMU = IMU.(fields{1});
% Daten transformieren um an KOS anzupassen
% Transformationsmatrix
T = [0 1 0; 1 0 0; 0 0 1]; % I-Pose
%T = [1 0 0; 0 1 0; 0 0 1];
% Transformation: jeder Zeile wird der neue Punkt zugeordnet. um Dimensionen anzupassen, muss transponiert werden.
%data = (T * IMU.S1094.acc')';

fa.acc=(T * IMU.S1094.acc')';
fa.gyr=(T * IMU.S1094.gyr_rad')';
ha.acc=(T * IMU.S0593.acc')';
ha.gyr=(T * IMU.S0593.gyr_rad')';
%fa.acc=(T * IMU.S0994.acc')';
%fa.gyr=(T * IMU.S1094.gyr_rad')';
%ha.acc=(T * IMU.S0477.acc')';
%ha.gyr=(T * IMU.S0477.gyr_rad')';

% parameters
freq = 52; % Hz
gyr_noise = 0.005; 
con_acc_rts_acc = 0.02; 
con_acc_rts_dof = 0.01;  
con_dof_rts_dof = 0.02; 

%% solve for translational alignment parameters
[rFA2,rHA] = lev_calc(fa.acc,fa.gyr,ha.acc,ha.gyr,1/freq);

%% solve for rotational alignment parameters
[~,~,fa.quat_s,ha.quat_s] = mekf_acc_s(fa,ha,freq,gyr_noise, con_acc_rts_acc, rFA2,rHA); 
[q1_imu,q2_imu,er] = wristAlign(fa.quat_s,ha.quat_s,50);

%% run filter
% multiplicative kalman smoother with linear acceleration and dof constraint
wr_rts_dof = mekf_wrist_acc_s(fa,ha,freq,gyr_noise, con_acc_rts_dof, con_dof_rts_dof, rFA2,rHA,q1_imu,q2_imu); 
eul_wr_rts_dof = rad2deg(quatToXYZ(wr_rts_dof));

%% quat angle
norms = sqrt(sum(wr_rts_dof.^2, 2));
wr_rts_dof_norm = wr_rts_dof ./ norms;
angles_rad = 2 * acos(wr_rts_dof_norm(:,1));
angles_deg = angles_rad * (180/pi);

%% Plot data
dt = 1/freq;
t = (0:length(angles_deg)-1)*dt;

subplot(2,2,1);
plot(t, eul_wr_rts_dof(:,1)); 
ylabel('rotation');
xticks(0:25:max(t));

subplot(2,2,2);
plot(t, eul_wr_rts_dof(:,2)); 
ylabel('flexion'); 
xticks(0:25:max(t));

subplot(2,2,3);
plot(t, eul_wr_rts_dof(:,3)); 
ylabel('deviation');
xticks(0:25:max(t));

subplot(2,2,4)
plot(t, angles_deg);
xlabel('Time (s)');
ylabel('joint angle');
xticks(0:25:max(t));