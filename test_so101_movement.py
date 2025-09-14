#!/usr/bin/env python3
"""
Test script for SO-101 robot arm scripted movement
This script demonstrates basic motor control and movement patterns
"""

import time
import numpy as np
from pathlib import Path
from lerobot import available_robots
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

def main():
    print("SO-101 Robot Movement Test")
    print("=" * 40)
    
    # Check if SO-101 is available
    print(f"Available robots: {available_robots}")
    if 'so101' not in available_robots:
        print("ERROR: SO-101 robot not found in available robots!")
        return
    
    try:
        # Initialize the robot
        print("Initializing SO-101 robot...")
        
        # Create configuration for SO-101
        # You may need to adjust the port based on your setup
        config = SO101FollowerConfig(
            port="/dev/ttyACM0",  # Adjust this to your actual port
            max_relative_target=None,
            use_degrees=False,
            id="ggando_so101_follower",  # Specify your calibration file name
            calibration_dir=Path("/home/gota/.cache/huggingface/lerobot/calibration/robots/so101_follower")
        )
        
        robot = make_robot_from_config(config)
        
        print("Robot initialized successfully!")
        print(f"Robot type: {type(robot)}")
        
        # Connect to the robot
        print("Connecting to robot...")
        robot.connect(calibrate=False)
        
        # Get initial position
        print("\nReading initial position...")
        initial_obs = robot.get_observation()
        print(f"Initial position: {initial_obs}")
        
        # Test gripper movement (motor ID 6)
        print("\nTesting gripper movement (motor ID 6)...")
        
        # Get motor names from the initial observation
        motor_names = [key.removesuffix('.pos') for key in initial_obs.keys() if key.endswith('.pos')]
        print(f"Available motors: {motor_names}")
        
        # Look for gripper motor (ID 6)
        gripper_motor = None
        for motor in motor_names:
            if '6' in motor or 'gripper' in motor.lower():
                gripper_motor = motor
                break
        
        if gripper_motor:
            print(f"Found gripper motor: {gripper_motor}")
            
            # Get current gripper position for reference
            current_gripper_pos = initial_obs.get(f"{gripper_motor}.pos", 0.0)
            print(f"Current gripper position: {current_gripper_pos}")
            
            # Three open/close cycles with increased range
            print("\n=== GRIPPER TRIPLE CYCLE ===")
            
            # Use expanded range: 2.0 (more closed) to 9.5 (more open)
            gripper_positions = [
                2.0,   # Closed position (cycle 1)
                9.5,   # Open position (cycle 1)
                2.0,   # Closed position (cycle 2)
                9.5,   # Open position (cycle 2)
                2.0,   # Closed position (cycle 3)
                9.5,   # Open position (cycle 3)
            ]
            
            movement_names = [
                "Cycle 1: Closing", 
                "Cycle 1: Opening", 
                "Cycle 2: Closing", 
                "Cycle 2: Opening",
                "Cycle 3: Closing",
                "Cycle 3: Opening"
            ]
            
            for target_pos, name in zip(gripper_positions, movement_names):
                print(f"\n{name}...")
                action = {f"{gripper_motor}.pos": target_pos}
                print(f"Target: {target_pos}")
                
                try:
                    robot.send_action(action)
                    print("Command sent! Waiting 1 second...")
                    time.sleep(1)
                    
                    # Check final position
                    obs = robot.get_observation()
                    final_pos = obs.get(f"{gripper_motor}.pos", "N/A")
                    print(f"Position: {final_pos}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    break
        else:
            print("Gripper motor (ID 6) not found!")
            print("Falling back to first available motor for demonstration...")
            if motor_names:
                demo_motor = motor_names[0]
                demo_movements = [
                    {f"{demo_motor}.pos": 0.2},
                    {f"{demo_motor}.pos": -0.2},
                    {f"{demo_motor}.pos": 0.0},
                ]
                
                for i, action in enumerate(demo_movements):
                    print(f"Demo movement {i+1}: {action}")
                    try:
                        robot.send_action(action)
                        time.sleep(2)
                    except Exception as e:
                        print(f"Error: {e}")
                        break
        
        print("\nMovement test completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Make sure your SO-101 robot is properly connected and powered on.")
        print("Check USB connections and device permissions.")
        return
    
    finally:
        # Clean shutdown
        try:
            if 'robot' in locals():
                print("\nShutting down robot...")
                robot.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()