#!/bin/sh

# compile:
# g++ -O3 -I .\eigen-3.4.0 kinematics.cpp run_map_elbow_acc.cpp map_elbow_acc.cpp -o map_elbow_acc

# input files need to be in "input/"
# ouput files get saved into "out/"

# run:
# open git bash
# to run all files execute
# bash ./run_map_elbow_acc.sh
# to only run one file execute
# bash ./run_map_elbow_acc.sh filename.txt  

ARGC=$#

freq=128.0 
priNoise=1.0
gyrNoise=0.005
conNoise=0.05
dofNoise=0.04
tol=0.01
lam=1e-8
max_iter=25 

: << 'END_COMMENT'
file="input/S0133_dict_frame.txt"
rUA2=(-0.111186980425604, -0.0147510320541111, 0.0447595702615915)	
rFA=(0.171656430869623, -0.0245042422329814, 0.0398833178232241)	
q1_imu=(0.139547584090428, 0.785105542601398, 0.60021792796274, 0.0622430534534179)
q2_imu=(0.146049569037773, -0.98469718824436, -0.00445454676991519, -0.0949796181325648)
END_COMMENT

#: << 'END_COMMENT'
if [ $ARGC -eq 0 ]; then
    echo "Calculating for all files!"
    read -p "Are u sure? [Y/n]" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for file in input/*; do
            if [ -f "$file" ]; then
                name=$(echo "$file" | cut -d '/' -f 2)
                line=$(grep "$name" data.txt)
                rUA2=$(echo "$line" | awk -F '\t' '{print $2}' | tr -d '()')
                rFA=$(echo "$line" | awk -F '\t' '{print $3}' | tr -d '()')
                q1_imu=$(echo "$line" | awk -F '\t' '{print $4}' | tr -d '()')
                q2_imu=$(echo "$line" | awk -F '\t' '{print $5}' | tr -d '()')
                ./map_elbow_acc ${file} ${freq} ${priNoise} ${gyrNoise} ${conNoise} ${dofNoise} ${rUA2[0]} ${rUA2[1]} ${rUA2[2]} ${rFA[0]} ${rFA[1]} ${rFA[2]} ${q1_imu[0]} ${q1_imu[1]} ${q1_imu[2]} ${q1_imu[3]} ${q2_imu[0]} ${q2_imu[1]} ${q2_imu[2]} ${q2_imu[3]} ${tol} ${lam} ${max_iter}
            fi
        done
    fi
elif [ $ARGC -eq 1 ]; then
    input_file=$"input/$1"
    line=$(grep "$1" data.txt)
    rUA2=$(echo "$line" | awk -F '\t' '{print $2}' | tr -d '()')
    rFA=$(echo "$line" | awk -F '\t' '{print $3}' | tr -d '()')
    q1_imu=$(echo "$line" | awk -F '\t' '{print $4}' | tr -d '()')
    q2_imu=$(echo "$line" | awk -F '\t' '{print $5}' | tr -d '()')
    # echo -e ${name}"\n"${rUA2}"\n"${rFA}"\n"${q1_imu}"\n"${q2_imu}
    
    ./map_elbow_acc ${input_file} ${freq} ${priNoise} ${gyrNoise} ${conNoise} ${dofNoise} ${rUA2[0]} ${rUA2[1]} ${rUA2[2]} ${rFA[0]} ${rFA[1]} ${rFA[2]} ${q1_imu[0]} ${q1_imu[1]} ${q1_imu[2]} ${q1_imu[3]} ${q2_imu[0]} ${q2_imu[1]} ${q2_imu[2]} ${q2_imu[3]} ${tol} ${lam} ${max_iter}
else
    echo "Wrong Input"
fi
# END_COMMENT

# ./map_elbow_acc ${file} ${freq} ${priNoise} ${gyrNoise} ${conNoise} ${dofNoise} ${rUA2[0]} ${rUA2[1]} ${rUA2[2]} ${rFA[0]} ${rFA[1]} ${rFA[2]} ${q1_imu[0]} ${q1_imu[1]} ${q1_imu[2]} ${q1_imu[3]} ${q2_imu[0]} ${q2_imu[1]} ${q2_imu[2]} ${q2_imu[3]} ${tol} ${lam} ${max_iter}
