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
euler_cmkmc_ahrs=eulerd(cmkmc_ahrs.Quat,'ZXY','frame');
cmkmc_ahrs2=orientation_estimation_ahrs_mkmc_fun_gpt(Accelerometer2,Gyroscope2,fs,xigma2_x,xigma2_y);
euler_cmkmc_ahrs2=eulerd(cmkmc_ahrs2.Quat,'ZXY','frame');

%% PLOTS
figure
x1=subplot(3,1,1);
plot(time,euler_cmkmc_ahrs(:,1),'blue',time,euler_cmkmc_ahrs2(:,1),'red')
legend('Yaw 1','Yaw2')
set(gca,'FontSize',12)
x2=subplot(3,1,2);
plot(time,euler_cmkmc_ahrs(:,2),'blue',time,euler_cmkmc_ahrs2(:,2),'red')
legend('Roll 1','Roll 2')
set(gca,'FontSize',12)
x3=subplot(3,1,3);
plot(time,euler_cmkmc_ahrs(:,3),'blue',time,euler_cmkmc_ahrs2(:,3),'red')
legend('Pitch 1','Pitch 2')
set(gca,'FontSize',12)
linkaxes([x1,x2,x3],'x')

%% JOINT ANGLE
time=cmkmc_ahrs.t;
q1_data = cmkmc_ahrs.Quat; % Simulated quaternions for sensor 1
q2_data = cmkmc_ahrs2.Quat; % Simulated quaternions for sensor 2

plot_quaternion_angles(time, q1_data, q2_data);

%% Pitch angle
% Convert quaternions to Euler angles (ZXY or other relevant order)
euler1 = euler_cmkmc_ahrs; % Extract Euler angles
euler2 = euler_cmkmc_ahrs2;

% Extract only the Pitch angle (Y-axis rotation in ZXY convention)
pitch1 = euler1(:,1); % Second column represents pitch
pitch2 = euler2(:,1);

% Compute pitch angle difference
pitch_diff = pitch1 - pitch2; % Absolute difference
 
% Plot the results
figure;
plot(time, pitch_diff, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Pitch Angle Difference (degrees)');
title('Pitch Angle Difference between Two Sensors');
grid on;

%%
X=time';
Y=pitch_diff;
ft = fittype('A*sin(2*pi*f*x + phi) + C', ...
    'independent', 'x', ...
    'coefficients', {'A', 'f', 'phi', 'C'});

% Erste Schätzung für die Parameter
A0 = (max(Y) - min(Y)) / 2;
f0 = 1 / (max(X) - min(X));
phi0 = 0;
C0 = mean(Y);

% Fit ausführen
fit_result = fit(X, Y, ft, 'StartPoint', [A0, f0, phi0, C0]);

% Angepasste Sinuskurve plotten
hold on;
plot(X, Y, 'b'); % Originaldaten
plot(X, fit_result(X), 'r', 'LineWidth', 2); % Angepasste Sinuskurve
legend('Original Signal', 'Angepasste Sinuskurve');
hold off;

%%
% Compute pitch angle difference
pitch_diff = pitch_diff-norm(fit_result(X)); % Absolute difference
 
% Plot the results
figure;
plot(time, pitch_diff, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Pitch Angle Difference (degrees)');
title('Pitch Angle Difference between Two Sensors');
grid on;