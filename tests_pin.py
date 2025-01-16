import pinocchio as pin
import numpy as np
from urdf_tools import *

def test_foward_kinematics_pin(model, current_joints, pose_ref) : 

    data = model.createData()

    # The end effector corresponds to the 7th joint
    q = np.array(current_joints) 

    # Afficher la transformation pour chaque joint (de oMi) au départ ! i = 0
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
    for i, oMi in enumerate(data.oMi):
        print(f"Joint {i}: Transformation\n{oMi}")

    print("On devrait avoir : ")
    oMdes = pin.SE3(pose_ref[0:3,0:3], pose_ref[0:3,3]) # matrice de rotation et position
    print(oMdes)



# Test
#-------------------------------
urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf"
prefix = "r" # r ou l
model = reduce_model_pin_arm_2(urdf_filename, prefix)

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


test_foward_kinematics_pin(model, test_1[0], test_1[1])