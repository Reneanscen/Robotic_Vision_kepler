#!/bin/bash


ROOT_PATH=/home/kepler/catkin_ws/src/perception/src/object_detection




tmux ls | grep -q "sf_visual"



if [ $? -eq 0 ]; then
	echo "window exist"
else
	echo "window not exist"
	tmux new-session -d -s sf_visual		# 新建
	tmux split-window -h -t sf_visual		# 水平分割

	
	tmux select-pane -t 0
	tmux send-keys -t sf_visual "roslaunch feynman_camera feynman_usb.launch" C-m

	tmux select-pane -t 1
	tmux send-keys -t sf_visual "rosrun perception shunfeng_box_qr.py" C-m
	
fi

tmux attach-session -t sf_visual


