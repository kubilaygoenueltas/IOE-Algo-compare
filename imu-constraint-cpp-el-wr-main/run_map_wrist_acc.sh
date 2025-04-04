#!/bin/sh

# compile:
# g++ -O3 -I .\eigen-3.4.0 kinematics.cpp run_map_wrist_acc.cpp map_wrist_acc.cpp -o map_wrist_acc
# run:
# open git bash,  execute ./run_map_wrist_acc.sh 

freq=128.0 
priNoise=1.0
gyrNoise=0.005
conNoise=0.05
dofNoise=0.08
tol=0.01
lam=1e-8
max_iter=25 

file="C:\Users\howar\Documents\cpp-code2\filter-development\data\S01ImuData_faha.txt"
rFA2=(-0.1118253608442759, -0.001458130654477649, 0.03959662408972148)
rHA=(0.05729572003349177, 0.005828477980505356, 0.031778460033004)
q1_imu=(1.003559632632128, -0.06385051082387301, -0.0007813591255495079, -0.08160166810334946)
q2_imu=(1.003100319652145, 0.1047617431168122, 0.02774623247470477, 0.003704146570002234)

./map_wrist_acc ${file} ${freq} ${priNoise} ${gyrNoise} ${conNoise} ${dofNoise} ${rFA2[0]} ${rFA2[1]} ${rFA2[2]} ${rHA[0]} ${rHA[1]} ${rHA[2]} ${q1_imu[0]} ${q1_imu[1]} ${q1_imu[2]} ${q1_imu[3]} ${q2_imu[0]} ${q2_imu[1]} ${q2_imu[2]} ${q2_imu[3]} ${tol} ${lam} ${max_iter}