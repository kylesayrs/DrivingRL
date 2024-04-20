# Driving Reinforcement Learning #
This repo implements a basic driving environment and driving agent trained using reinforcement learning.
<p align="center">
<img src="assets/obstacle_avoidance.gif" alt="Obstacle Avoidance"/>
</p>


## Driving Environment ##
The environment has three kinds of bodies: the car, obstacles, and the goal. When the car touches an obstacle, the agent receives a negative penalty. When the car touches the goal, the agent receives a positive reward. The gamma parameter controls how much the agent should care about future wards in relative to immediate rewards.  All three objects are placed randomly within the scene and an obstacle border is placed around the scene to prevent the vehicle from going out of bounds. Limits on the maximum car acceleration and speed are imposed.


## Observations ##
The agent observes the following dictionary of values at each time step:
| Observation | Significance | Format |
| ----------- | ------------ | ------ |
| Car Velocity | Speed of car in X and Y axes | (velocity_x, velocity_y) |
| Car Angle | Angle wich car is facing | (cos(theta), sin(theta)) |
| Car Angle Velocity | Speed at which car angle is changing | (theta, ) |
| Car Visual | Distance from car to obstacle along each visual ray | (dist0, dist1, ...) |
| Goal Angle | Angle between car angle and goal | (cos(theta), sin(theta)) |
| Goal Distance | Distance from car to goal | (distance, ) |

## Driving Agent ##
The agent was trained using proximal policy optimization, which computes the advantage of a given policy against baselines rewards in order to compute a gradient with which to train the policy network. Given that the agent performs actions in a continuous action space, the deep deterministic policy gradient (DDPG) was also evaluated but found to be less sample efficient than PPO. 

<p align="center">
<img src="assets/ppo.png" alt="Proximal Policy Optimization"/>
</p>

At each time step the agent infers the following actions:
| Observation | Significance | Format |
| ----------- | ------------ | ------ |
| Forward Acceleration | Change in car velocity in the forward direction, can be negative | (acceleration, ) |
| Angular Acceleration | Change in car angle relative to current angle | (angle_acceleration, ) |


## Lessons learned ##
During the experimentation with a car demo environment, the PPO reinforcement learning agent was trained to navigate towards the goal while avoiding obstacles. However, an unexpected behavior was observed where the trained agent consistently spun in circles instead of moving directly towards the target. Upon investigation, it was discovered that the absence of tire friction in the system, coupled with the agent's limited ability to measure only forward velocity, led to the suboptimal strategy of spinning to perceive velocity from all directions before potentially colliding with an obstacle. This issue was resolved by providing the agent with global velocity measurements, which enabled it to adopt a more efficient and direct path towards the target.

This experience highlights the intelligence of reinforcement learning agents, as they can sometimes uncover aspects of the environment that are not immediately apparent to the designer. It is therefore important to carefully design the environment and provide the agent with the necessary information to make informed decisions.
