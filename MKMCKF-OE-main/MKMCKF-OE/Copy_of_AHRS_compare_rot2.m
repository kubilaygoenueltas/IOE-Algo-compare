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
Magnetic= ones(10392, 3);
Magnetic(:, 1)=40;
Magnetic(:, [2,3])=-40;

%% load the data
load('gait_02_100hz.mat');
IMU=gait;

Accelerometer=IMU.Acceleration;
Gyroscope=IMU.Gyroscope;
fs=IMU.fs;
Magnetic=IMU.Magnetic;
len=length(Accelerometer);

%for i=1:len
%Accelerometer_norm(i)= norm(Accelerometer(i,:)); 
%Magnetic_norm(i)= norm(Magnetic(i,:)); 
%end

%% plot the raw data
t=0:1/fs:1/fs*(length(Accelerometer)-1);
time=[t;t;t];
time=time';
figure
x1=subplot(3,1,1);
plot(time,Accelerometer)
legend('acc')
set(gca,'FontSize',16)
x2=subplot(3,1,2);
plot(time,Gyroscope)
legend('gyro')
set(gca,'FontSize',16)
x3=subplot(3,1,3);
plot(time,Magnetic)
legend('mag')
set(gca,'FontSize',16)
linkaxes([x1,x2,x3],'x')


%% mkmc ahrs
sigma_1=1.6188;
sigma_2=0.4234;

sigma1=2*sigma_1*sigma_1;
sigma2=2*sigma_2*sigma_2;
xigma_x=[10^8 10^8 10^8 10^8 10^8 10^8 sigma1 sigma1 sigma1 sigma2 sigma2 sigma2]; 
xigma_y=[10^8 10^8 10^8 10^8 10^8 10^8];
mkmc_ahrs=orientation_estimation_ahrs_mkmc_fun_(Accelerometer,Gyroscope,Magnetic,fs,xigma_x,xigma_y);
euler_mkmc_ahrs=eulerd(mkmc_ahrs.Quat,'ZXY','frame');

%% cmkmc ahrs
cmkmc_ahrs=orientation_estimation_ahrs_mkmc_fun_nomag_(Accelerometer,Gyroscope,Magnetic,fs,xigma_x,xigma_y);
euler_cmkmc_ahrs=eulerd(cmkmc_ahrs.Quat,'ZXY','frame');

%% cmkmc ahrs 2
sigma_1=1.6188;
sigma1=2*sigma_1*sigma_1;
xigma2_x=[10^8 10^8 10^8 10^8 10^8 10^8 sigma1 sigma1 sigma1]; 
xigma2_y=[10^8 10^8 10^8];
cmkmc_ahrs2=orientation_estimation_ahrs_mkmc_fun_gpt(Accelerometer,Gyroscope,fs,xigma2_x,xigma2_y);
euler_cmkmc_ahrs2=eulerd(cmkmc_ahrs2.Quat,'ZXY','frame');

%% cmkmc ahrs 3
Magnetic3=Magnetic;
Magnetic3(:,1)=-100;
Magnetic3(:,[2,3])=-100;
cmkmc_ahrs3=orientation_estimation_ahrs_mkmc_fun_nomag_(Accelerometer,Gyroscope,Magnetic3,fs,xigma_x,xigma_y);
euler_cmkmc_ahrs3=eulerd(cmkmc_ahrs3.Quat,'ZXY','frame');

%% PLOTS
figure
x1=subplot(3,1,1);
plot(time,euler_mkmc_ahrs(:,1),'blue',time,euler_cmkmc_ahrs2(:,1),'green',time,euler_cmkmc_ahrs3(:,1),'black')
legend('MKMC Yaw',  'CMKMC Yaw 2', 'CMKMC Yaw 3')
set(gca,'FontSize',12)
x2=subplot(3,1,2);
plot(time,euler_mkmc_ahrs(:,2),'blue',time,euler_cmkmc_ahrs2(:,2),'green',time,euler_cmkmc_ahrs3(:,2),'black')
legend('MKMC Roll', 'CMKMC Roll 2', 'CMKMC Roll 3')
set(gca,'FontSize',12)
x3=subplot(3,1,3);
plot(time,euler_mkmc_ahrs(:,3),'blue',time,euler_cmkmc_ahrs2(:,3),'green',time,euler_cmkmc_ahrs3(:,3),'black')
legend('MKMC Pitch', 'CMKMC Pitch 2', 'CMKMC Pitch 3')
set(gca,'FontSize',12)
linkaxes([x1,x2,x3],'x')