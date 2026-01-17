#!/bin/bash
# Record demonstrations for HIL-SERL training
# Uses leader arm for teleoperation

cd /home/gota/ggando/ml/lerobot

uv run python -m lerobot.scripts.rl.gym_manipulator \
  --config_path /home/gota/ggando/ml/so101-playground/env_config_so101.json
