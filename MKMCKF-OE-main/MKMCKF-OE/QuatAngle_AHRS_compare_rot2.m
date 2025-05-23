%%
% This function is used to compare the performance of orientation
% estimation using ESKF, GD, and MKMCKF-OE.
%
% The raw data is sampled at gait frequency f=0.2hz with 
% magnetic disturbance.
%%

clear all
%% add path
addpath('MKMCKF-OE');
addpath('data100hz');
addpath('madgwick_algorithm_matlab/quaternion_library');

%% load the data
load('S0110_01_dict_frame.mat');
IMU=data;

fs=52; %Hz
sample_freq=fs;

Accelerometer=IMU.S1094.acc;
Gyroscope=IMU.S1094.gyr_rad;
Accelerometer2=IMU.S0593.acc;
Gyroscope2=IMU.S0593.gyr_rad;
Magnetic= zeros(10392, 3);
Magnetic2=Magnetic;

%% plot the raw data
t=0:1/fs:1/fs*(length(Accelerometer)-1);
time=[t;t;t];
time=time';
figure
x1=subplot(3,1,1);
plot(time,Accelerometer,time,Accelerometer2)
legend('acc')
set(gca,'FontSize',16)
x2=subplot(3,1,2);
plot(time,Gyroscope,time,Gyroscope2)
legend('gyro')
set(gca,'FontSize',16)
x3=subplot(3,1,3);
plot(time,Magnetic,time,Magnetic2)
legend('mag')
set(gca,'FontSize',16)
linkaxes([x1,x2,x3],'x')


%% cmkmc ahrs
sigma_1=1.6188;
sigma1=2*sigma_1*sigma_1;
xigma2_x=[10^8 10^8 10^8 10^8 10^8 10^8 sigma1 sigma1 sigma1]; 
xigma2_y=[10^8 10^8 10^8];
cmkmc_ahrs=orientation_estimation_ahrs_mkmc_fun_gpt(Accelerometer,Gyroscope,fs,xigma2_x,xigma2_y);
euler_cmkmc_ahrs=eulerd(cmkmc_ahrs.Quat,'XYZ','frame');
cmkmc_ahrs2=orientation_estimation_ahrs_mkmc_fun_gpt(Accelerometer2,Gyroscope2,fs,xigma2_x,xigma2_y);
euler_cmkmc_ahrs2=eulerd(cmkmc_ahrs2.Quat,'XYZ','frame');

%% PLOTS
figure
x1=subplot(3,1,1);
plot(time,euler_cmkmc_ahrs(:,1),'blue',time,euler_cmkmc_ahrs2(:,1),'red')
legend('roll 1','roll')
set(gca,'FontSize',12)
x2=subplot(3,1,2);
plot(time,euler_cmkmc_ahrs(:,2),'blue',time,euler_cmkmc_ahrs2(:,2),'red')
legend('pitch 1','pitch 2')
set(gca,'FontSize',12)
x3=subplot(3,1,3);
plot(time,euler_cmkmc_ahrs(:,3),'blue',time,euler_cmkmc_ahrs2(:,3),'red')
legend('yaw 1','yaw 2')
set(gca,'FontSize',12)
linkaxes([x1,x2,x3],'x')

%% JOINT ANGLE
time=cmkmc_ahrs.t;
q1_data = cmkmc_ahrs.Quat; % Simulated quaternions for sensor 1
q2_data = cmkmc_ahrs2.Quat; % Simulated quaternions for sensor 2

plot_quaternion_angles(time, q1_data, q2_data);

