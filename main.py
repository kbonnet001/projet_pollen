
import numpy as np
from build_reduced_model import symbolic_inverse_kinematics_continuous_for_test


# Load UR robot arm
# This path refers to Pinocchio source code but you can define your own directory here.
# Remarque : le fichier urdf reachy a été modifié pour mettre les bon path des fichiers de meshes
# il faut ajuster mais dans le docker il n'y aura rien à changer !
urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf"
prefix = "r" # r ou l


# ---------------------------------------------------------------------
# Exemple de goal_pose (position et orientation)
# goal_position = np.array([-0.20, -0.39,  0.79])  # Translation, pris de l'exemple "dessiner carré" point A
goal_orientation = np.array([0.0, 0.0, 0.0])  # Angles d'Euler (en radians)

# goto(1, 1, 0)
goal_position= np.array([-0.07036811, -0.10607062,  0.44850763 + 1e-2])
# Représentation goal_pose
goal_pose = np.array([goal_position, goal_orientation], dtype=float)

# remarque : matrix_rot n'existe pas dans le code, c'est juste pour faire des tests
matrix_rot =  np.array([
            [2.22044605e-16, -0.00000000e+00, -1.00000000e+00],
            [0.00000000e+00,  1.00000000e+00, -0.00000000e+00],
            [1.00000000e+00,  0.00000000e+00,  2.22044605e-16],
        ])
matrix_rot =  np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0],
        ])
matrix_rot = np.array([[ 9.65925856e-01, -2.54887032e-01,  4.49426576e-02],
                       [ 2.58818935e-01,  9.51251352e-01, -1.67730807e-01],
                       [ 6.43660370e-07,  1.73647534e-01,  9.84807867e-01]])
# -------------------------------------


# current_joints = [2.726055496581969, -39.46273957135307, -12.748116930137442, -96.15064249358801, 
#                   -2.0934329169282213, 11.418022966642084, -28.905708907141634]
# current_joints =  [0.029034681618213654, -0.9231657385826111, -0.568751335144043, -1.1559369564056396, 0.35746899247169495, -0.22973781824111825, -0.5916433334350589]
# current_joints = [ -0.31236106, -1.17958713 ,-0.06547543 ,-1.94147157 , 0.26235183 , 0.69959795,  -1.08927315]
# current_joints = [0,0,0,0,0,0,0]
current_joints = [0.0, -0.17453292519943295, -0.2617993877991494, 0.0, 0.0, 0.0, 0.0]
# ça devrait être : 
# np.array(
#     [
#         [1, 0, 0, 0],
#         [0, 1, 0, -0.2],
#         [0, 0, 1, -0.66],
#         [0, 0, 0, 1],
#     ]

current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)
# current_joints = symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot)



# -----------------------------------------
# exemple 1 : 
# current_joints =  [0.029034681618213654, -0.9231657385826111, -0.568751335144043, -1.1559369564056396, 0.35746899247169495, -0.22973781824111825, -0.5916433334350589]

# goal_position= (0.4, -0.5, -0.02 + 0.0001)
# # Représentation goal_pose
# goal_pose = np.array([goal_position, goal_orientation], dtype=float)

# # remarque : matrix_rot n'existe pas dans le code, c'est juste pour faire des tests
# matrix_rot =  np.array([
#             [2.22044605e-16, -0.00000000e+00, -1.00000000e+00],
#             [0.00000000e+00,  1.00000000e+00, -0.00000000e+00],
#             [1.00000000e+00,  0.00000000e+00,  2.22044605e-16],
#         ])

# Attendu : +0.01
#---------
# 1e ik =  [ 0.03436254 -0.92832338 -0.5660241  -1.15871577  0.35144505 -0.22996927
#  -0.60071662]

# Attendu +0.01
# 1e ik =  [ 0.03768264 -0.93346919 -0.58131807 -1.18295735  0.36320413 -0.20622169 -0.60985649]

# attendu + 0.0001
# [ 0.03395566 -0.92783008 -0.56444604 -1.15621563  0.35028492 -0.23237248,  -0.59981507]

# attendu + 1e-4
# [ 0.03395566 -0.92783008 -0.56444604 -1.15621563  0.35028492 -0.23237248  -0.59981507]

# attendu + 1e-5
