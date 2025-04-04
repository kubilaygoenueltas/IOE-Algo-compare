#!/bin/sh

# compile:
# g++ -O3 -I .\eigen-3.4.0 kinematics.cpp run_map_elbow_acc.cpp map_elbow_acc.cpp -o map_elbow_acc
# run:
# open git bash,  execute ./run_map_elbow_acc.sh 


freq=128.0 
priNoise=1.0
gyrNoise=0.005
conNoise=0.05
dofNoise=0.04
tol=0.01
lam=1e-8
max_iter=25 

file="C:\Users\howar\Documents\cpp-code2\filter-development\data\S01ImuData_uafa.txt"
rUA2=(-0.111186980425604, -0.0147510320541111, 0.0447595702615915)	
rFA=(0.171656430869623, -0.0245042422329814, 0.0398833178232241)	
q1_imu=(0.139547584090428, 0.785105542601398, 0.60021792796274, 0.0622430534534179)
q2_imu=(0.146049569037773, -0.98469718824436, -0.00445454676991519, -0.0949796181325648)

./map_elbow_acc ${file} ${freq} ${priNoise} ${gyrNoise} ${conNoise} ${dofNoise} ${rUA2[0]} ${rUA2[1]} ${rUA2[2]} ${rFA[0]} ${rFA[1]} ${rFA[2]} ${q1_imu[0]} ${q1_imu[1]} ${q1_imu[2]} ${q1_imu[3]} ${q2_imu[0]} ${q2_imu[1]} ${q2_imu[2]} ${q2_imu[3]} ${tol} ${lam} ${max_iter}
