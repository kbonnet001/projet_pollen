# 📖 Developer documentation

## Function Testing
To test the implemented functions, a set of test functions is included in:

📂 **`/home/reachy/dev/reachy2_symbolic_ik/src/example/test_movement_test.py`**

This file, largely inspired by **`movement_test.py`**, includes:

**`load_model`**: Loads the required URDF model for Pinocchio or Pink.
**`test_sphere`**: Evaluates the effectiveness of spherical barriers.

## Inverse Kinematics Methods
📂 **`/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/ik_methods.py`**

This file contains the programmed inverse kinematics functions for Pinocchio and Pink, along with other essential utilities. The functions below may eventually replace symbolic_inverse_kinematics_continuous in:

📂 **`/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/control_ik.py`**

**`symbolic_inverse_kinematics_continuous_with_pinocchio`**: Solves inverse kinematics using Pinocchio, without constraint handling.
**`symbolic_inverse_kinematics_continuous_with_pink`**: Solves inverse kinematics using the QP method with Pink.
**`symbolic_inverse_kinematics_continuous_with_pink_sphere`**: Solves inverse kinematics using the QP method with Pink, integrating spherical barriers at the end-effectors.
**`symbolic_inverse_kinematics_continuous_with_pink_V2`**: Solves inverse kinematics using the QP method with Pink, integrating both the previous spherical barriers and some other modification like correcting the joint and velocity constraint.
## Metrics and Visualization
These functions utilize:

📂 **`compute_metrics.py`**: Computes various performance metrics.
📂 **`CSVLogger.py`**: Saves metrics into a CSV file.
📂 **`plot_metrics.py`**

Several plotting functions are available to analyze the inverse kinematics resolution:  

### Joint Angles and Velocities  
- **`plot_q`**: Joint angles at each iteration.  
- **`plot_velocity`**: Derivative of joint angles over time for each iteration.  
- **`plot_velocity_std`**: Standard deviation of joint angle derivatives over time (computed over the last 10 iterations).  

### Differences Between Iterations  
- **`plot_ecart_q`**: Difference between the joint angle vectors at **t = i + 1** and **t = i** for each iteration.  
- **`plot_ecart_pos_rot`**: Difference between positions (quaternions) at **t = i + 1** and **t = i** for each iteration.  

### Trajectory and Goal Comparison  
- **`plot_translations_and_draw`**:  
  - Plots **x, y, and z** positions for each iteration.  
  - Displays the drawn trajectory.  
  - Optional parameters:  
    - `plot_goal=True`: Compares the result with the true goal.  
    - `plot_pollen=True`: Compares the result with Pollen's current resolution. 
