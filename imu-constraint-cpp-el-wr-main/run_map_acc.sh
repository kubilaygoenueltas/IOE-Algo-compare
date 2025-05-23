#!/bin/sh

# compile:
# g++ -O3 -I .\eigen-3.4.0 kinematics.cpp run_map_acc.cpp map_acc.cpp -o map_acc
# run:
# open git bash,  execute ./run_map_acc.sh 

freq=52 
priNoise=1.0
gyrNoise=0.005
conNoise=0.04
tol=0.01
lam=1e-8
max_iter=25

file="C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/IOE-Algo-compare/Daten/TXT_crab/S0133_dict_frame.txt"
rUA2=(-0.111186980425604, -0.0147510320541111, 0.0447595702615915)	
rFA=(0.171656430869623, -0.0245042422329814, 0.0398833178232241)
q1_imu=(0.139547584090428, 0.785105542601398, 0.60021792796274, 0.0622430534534179)
q2_imu=(0.146049569037773, -0.98469718824436, -0.00445454676991519, -0.0949796181325648)

./map_acc ${file} ${freq} ${priNoise} ${gyrNoise} ${conNoise} ${rProx[0]} ${rProx[1]} ${rProx[2]} ${rDist[0]} ${rDist[1]} ${rDist[2]} ${q1_imu[0]} ${q1_imu[1]} ${q1_imu[2]} ${q1_imu[3]} ${q2_imu[0]} ${q2_imu[1]} ${q2_imu[2]} ${q2_imu[3]} ${tol} ${lam} ${max_iter}