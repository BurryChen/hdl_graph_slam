#!/bin/bash

# 1.arg
echo "path_arg : $1"

durations=(471 115 484 83 29 288 115 115 423 165 125)

# 2.SLAM, write odom file

for seq in 00 01 02 03 04 05 06 07 08 09 10
do

file=${1}/data/KITTI_${seq}_odom.txt
echo $seq $file
gnome-terminal -x bash -c "echo $seq;roslaunch hdl_graph_slam ndt_odom_kitti.launch odom_file:=$file &sleep 10s;rosbag play --clock /home/whu/data/loam_KITTI/velobag/velo_${seq}.bag -r 0.25;echo $seq over&&sleep 10s;exit"
i=10#$seq
time=`expr 60 + ${durations[i]} \* 4`
echo $time s
sleep $time
wait

done 

# 3.eva
cd '/home/whu/data/loam_KITTI/devkit_old/cpp' 
./evaluate_odometry ${1}
cd ~

# 4.error png
python '/home/whu/slam_ws/src/hdl_graph_slam/scripts/error_odom_png.py' ${1}
