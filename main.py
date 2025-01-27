import numpy as np
from ik_continous_pin import symbolic_inverse_kinematics_continuous_for_test
import pinocchio as pin


# Load UR robot arm
# This path refers to Pinocchio source code but you can define your own directory here.
# Remarque : le fichier urdf reachy a été modifié pour mettre les bon path des fichiers de meshes
# il faut ajuster mais dans le docker il n'y aura rien à changer !
urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf"
prefix = "r" # r ou l

current_joints = [0.0, -0.17453292519943295, -0.2617993877991494, 0.0, 0.0, 0.0, 0.0]
# ça devrait être cette pose (d'après Pollen): 
# np.array(
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, -0.2],
#         [0, 0, 1, -0.66],
#         [0, 0, 0, 1],
#     ]
#
# Mais pinocchio trouve cela (voir test fichier tests_pin.py) : 
# Joint 7: Transformation
#   R =
#            1  5.30131e-14 -8.22751e-07
#  4.66081e-13            1  6.31319e-07
#  8.22751e-07 -6.31319e-07            1
#   p = -0.0451998       -0.2       0.44

# ---------------------------------------------------------------------
# Exemple de goal_pose (position et orientation)

goal_position= np.array([-0.0451998 + 1e-1, -0.2 + 1e-1, 0.44 + 1e-1])
goal_orientation = np.array([0.0, 0.0, 0.0])  # Angles d'Euler (en radians)

# Représentation goal_pose dans code Pollen
goal_pose = np.array([goal_position, goal_orientation], dtype=float)

# remarque : matrix_rot n'existe pas dans le code Pollen, c'est juste pour faire des tests
matrix_rot =  np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])


symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot, debug=True, plot=True)



