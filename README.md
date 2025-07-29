# Autonomous_FF_Drone
A work-in-progress project to develop an autonomous drone in a custom GODOT environment.

## Overview
This project focuses on building an autonomous drone using Behavior Cloning (BC) models, integrated with a Finite State Machine (FSM).
Each BC model is trained for a specific flight phase: takeoff, navigation, approaching targets, landing, parking, etc.
The FSM decides which flight phase the drone is currently in and switches between models accordingly.
The aim is to achieve human-like drone control in a simulated environment with obstacles and dynamic tasks.

## Key Features

### Custom GODOT Environment
A fully custom 3D environment simulating a city-like layout with buildings, poles, towers, and other obstacles.
Collisions end the episode, enforcing safe navigation.

### Physics-Aware Drone Dynamics
Custom-coded drone physics (not 100% real-world accurate but close enough for testing).
Every input to roll, pitch, and yaw matters under the effects of gravity and environmental forces.
G-force calculations for collisions and hard landings that can lead to mission failure.

### Behavior Cloning + FSM Architecture
Multiple BC models specialized for each mission phase.
FSM handles phase transitions and ensures each model runs in the right context.

### Training and Inference Pipeline
Training supports sequence-based BC with both LiDAR data and state features.
Inference server communicates with GODOT in real-time and returns actions based on live state information.

### Project Status
This is a one-person project and still a work in progress. The pipeline, models, and environment are being improved continuously.
The drone can already handle basic autonomous navigation and task execution (for example, approaching a target and returning).
The next goal is to make policies smoother, more robust, and closer to how a human expert would control the drone.

### Final Comments 
Feedback, suggestions, and contributions are welcome!
Ideas for improving BC models or FSM design are always welcome!
Enhancements to the GODOT environment or drone physics
New scenarios and testing
Data Acquisition - through manual episode run and/or DAgger
If you are interested, open an issue, fork the repository, or reach out.

# Acknowledgments
This project of BC training, FSM and RL development is novel for me and has taken a substantial amount of study to get to this point. It has taken a lot of experimentation and coding to reach this stage. Therefore, any constructive feedback would be greatly appreciated!

#### THANKS!
