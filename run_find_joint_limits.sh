#!/bin/bash
# Find joint limits for SO-101 robot through teleoperation
# Move the robot through its entire task workspace using the leader arm
#
# Run this script in a terminal (not from Claude) - it requires interactive input

cd /home/gota/ggando/ml/lerobot

uv run python -m lerobot.scripts.find_joint_limits \
  --robot.type=so101_follower_end_effector \
  --robot.port=/dev/ttyACM0 \
  --robot.id=ggando_so101_follower \
  --robot.urdf_path=/home/gota/ggando/ml/so101-playground/models/so101_new_calib.urdf \
  --robot.target_frame_name=gripper_frame_link \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=ggando_so101_leader
