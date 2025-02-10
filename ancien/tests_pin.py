import pinocchio as pin
import numpy as np
from urdf_tools import *

H_TW = np.array([
    [1.000, 0.000, 0.000,-0.045],
    [0.000, 1.000, 0.000, 0.000],
    [0.000, 0.000, 1.000, 1.000],
    [0.000, 0.000, 0.000, 1.000],
])

H_HP = np.array([
    [1.000, 0.000, 0.000, 0.000],
    [0.000, 1.000, 0.000, 0.000],
    [0.000, 0.000, 1.000, 0.100],
    [0.000, 0.000, 0.000, 1.000],
])

def test_foward_kinematics_pin(model, current_joints, pose_ref) : 

    data = model.createData()

    # The end effector corresponds to the 7th joint
    q = np.array(current_joints) 

    # Afficher la transformation pour chaque joint (de oMi) au départ ! i = 0
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
    for i, oMi in enumerate(data.oMi):
        print(f"Joint {i}: Transformation\n{oMi}") # print(np.dot(data.oMi[7], np.linalg.inv(H_TW)))

    print("On devrait avoir : ")
    oMdes = pin.SE3(pose_ref[0:3,0:3], pose_ref[0:3,3]) # matrice de rotation et position
    print(oMdes)


# Test
#-------------------------------
urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf"
prefix = "r" # r ou l
model = reduce_model_pin_arm(urdf_filename, prefix)

# control_ik default pose
test_1 = [ 
    [0.0, -0.17453292519943295, -0.2617993877991494, 0.0, 0.0, 0.0, 0.0],

    np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -0.2],
            [0, 0, 1, -0.66],
            [0, 0, 0, 1],
        ]
    )
]

test_2 = [
    [-0.3138059973716736, -1.171371340751648, -0.06977630406618118, -1.9414715766906738, 0.2660022974014287, 
     0.7010691761970521, -1.0821599960327146], 

    np.array(
        [
            [-5.09038399e-07,  3.53408358e-07, -1.00000000e+00,  3.99999885e-01],
            [ 8.22356317e-07,  1.00000000e+00,  3.53407939e-07, -2.99999865e-01],
            [ 1.00000000e+00, -8.22356137e-07, -5.09038690e-07,  7.27754452e-08],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ] )
]



test_3 = [
    [0.0, 0.17453292519943295, 0.2617993877991494, 0.0, 0.0, 0.0, 0.0],
    np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0.2],
        [0, 0, 1, -0.66],
        [0, 0, 0, 1],
    ]),
]

test_foward_kinematics_pin(model, test_1[0], test_1[1])
# Joint 7: Transformation
#   R =
#            1  5.30131e-14 -8.22751e-07
#  4.66081e-13            1  6.31319e-07
#  8.22751e-07 -6.31319e-07            1
#   p = -0.0451998       -0.2       0.44

# On devrait avoir : 
#   R =
# 1 0 0
# 0 1 0
# 0 0 1
#   p =     0  -0.2 -0.66     

#-------------------------------------------------
# test_foward_kinematics_pin(model, test_2[0], test_2[1])
# Joint 7: Transformation
#   R =
# -5.09038e-07 -3.00181e-07           -1
#  8.22356e-07            1 -3.00182e-07
#            1 -8.22356e-07 -5.09038e-07
#   p = 0.2548   -0.3      1

# On devrait avoir : 
#   R =
# -5.09038e-07  3.53408e-07           -1
#  8.22356e-07            1  3.53408e-07
#            1 -8.22356e-07 -5.09039e-07
#   p =         0.4        -0.3 7.27754e-08

#-------------------------------------------------
# bras gauche
# test_foward_kinematics_pin(reduce_model_pin_arm(urdf_filename, "l"), test_3[0], test_3[1])

# Joint 7: Transformation
#   R =
#            1 -5.32352e-14 -4.84428e-07
#  3.59167e-13            1  6.31319e-07
#  4.84428e-07 -6.31319e-07            1
#   p = -0.0451999        0.2       0.44

# On devrait avoir : 
#   R =
# 1 0 0
# 0 1 0
# 0 0 1
#   p =     0   0.2 -0.66








# current_joints = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.17453292519943295, 0.2617993877991494, 0.0, 0.0, 0.0, 0.0, 
#     0.0, 0.0, 0.0, 0.0, 0.0]

# model_reachy = pin.buildModelFromUrdf("/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf")
# test_foward_kinematics_pin(model_reachy, current_joints, test_1[1])

