## This is a repo with all the project I have done
1. LiDAR-Based SLAM: Implemented a simultaneous localization and mapping on a differential-drive robot with 2D-LiDAR scan, RGBD images and IMU data.
  - Improve robot odometery estimated by IMU measurement using LiDAR scan via iterative closest point (ICP).
  - Built up a map of the environment by using using LiDAR scan and RGBD images
  - Optimize the robot trajectory estimates for 15% by further by using the GTSAM library with loop-closure constraints
2. Visual-Based SLAM: Implemented visual-based simultaneous localization and mapping (SLAM) with extended Kalman filter (EKF) algorithm on a moving car.
  - Used IMU input and EKF prediction to predict the car pose (SE(3)) over time and then use it as reference to do predict the landmark pose through EKF update step
  - Improve both the localization and mapping by using stereo camera input
3. Path Planning: Designed algorithm that can help find the fastest trajectory to goal and avoid obstacles for 9 3D maps.
  - Decrease the collision rate down to 3% and shorten the path 20% by implementing a generalized A* algorithm.
  - Implementing RRT and RRT* algorithm
3. Robot Manipulation: Designed software that can manipulate a 5R robot arm on top of a chassis with Mecanum wheels.
  - Shorten the path-finding time 40% by implementing RRT algorithm, and then shorten the path 20% by using RRT*
  - Designed a trajectory generator that helps robot gripper generate trajectory for a given task. It can make sure smooth trajectory, no collision and use find the shortest path for the robot end-effector
  - Designed a controller using inverse kinematic and inverse dynamic, to make sure the end-effector moves as expected
  - Implemented a PID controller that helps avoid singularities，lower the error overshoot by 25%，shorten the response time by 30% and make sure not oscillation.
  - Use RGBD input make the arm able to track a moving object by using image Jacobian and inverse dynamic
4. The Door&Key (D&K) problem:
  - The D&K problem is to make sure the agent finds the goal within the D&K environment which is a  2D grid map that randomly generates an agent, a goal,  some obstacles, a key and a door(can’t be passed without key).
  - Used dynamic programming to solve the D&K problem.
5. Infinite-Horizon Stochastic Optimal Control Problem:
  - Successfully track a noised differential drive by implementing receding-horizon certainty equivalent control (CEC) and generalized policy iteration (GPI).
6. Orientation Tracking :
  - Managed to track the orientation of a 3D rotating body by implementing gradient descent algorithm on IMU data. Stitched some RGB images into a panorama by using the tracking results.
7. 
