from pathlib import Path
import numpy as np
import pinocchio as pin

import numpy as np
from numpy.linalg import norm, solve
from os.path import abspath

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt


# Goal: Build a reduced model from an existing URDF model by fixing the desired joints
# at a specified position.
# Load UR robot arm
# This path refers to Pinocchio source code but you can define your own directory here.
# pinocchio_model_dir = Path(__file__).parent.parent / "models"
# model_path = pinocchio_model_dir / "example-robot-data/robots"
# mesh_dir = pinocchio_model_dir
# You should change here to set up your own URDF file
urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/essai/projet_pollen/reachy.urdf"
# mesh_dir = "~/reachy_ws/src/reachy2_core/reachy_description/meshes"
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_filename)
# Check dimensions of the original model
print("standard model: dim=" + str(len(model.joints)))
#for jn in model.joints:
   #print(jn)
#print("-" * 30)
# Create a list of joints to NOT lock
prefix = "r"
jointsToNotLock = [f"{prefix}_shoulder_pitch",f"{prefix}_shoulder_roll", f"{prefix}_elbow_arm_link", 
                   f"{prefix}_elbow_yaw",f"{prefix}_elbow_pitch", f"{prefix}_wrist_roll", f"{prefix}_wrist_pitch", f"{prefix}_wrist_yaw"]
print(jointsToNotLock)
#Get the Id of all existing joints
jointsToLockIDs = []
initialJointConfig = np.ones(len(model.joints)-1)
i=-1
# for jn in model.joints:
#    i+=1
#    if jn.getJointName() not in jointsToNotLock: #if true
#       jointsToLockIDs.append(jn.index)
#       initialJointConfig[i] = 0

for i, jn in enumerate(model.joints):  # Utilisez enumerate pour obtenir l'index
#    if i ==0:
#       continue
   joint_name = model.names[i]  # Accédez au nom du joint via model.names
   #print(i)
   print(joint_name)
   if joint_name not in jointsToNotLock:  # Si le joint doit être verrouillé
        jointsToLockIDs.append(jn.id)  # Utilisez jn.id pour l'ID du joint
        # jointsToLockIDs.append(i)
        initialJointConfig[i-1] = 0  # Fixez la configuration initiale à 0

print(jointsToLockIDs)

# Get the ID of all existing joints
#jointsToLockIDs = []
#for jn in jointsToLock:
#    if model.existJointName(jn):
#        jointsToLockIDs.append(model.getJointId(jn))
#    else:
#        print("Warning: joint " + str(jn) + " does not belong to the model!")
# Set initial position of both fixed and revoulte joints
#initialJointConfig = np.array(
#    [
#        0,
#        0,
#        0,  # shoulder and elbow
#        1,
#        1,
#        1,
#    ]
#)  # gripper)


# Option 1: Only build the reduced model in case no display needed:
model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)
model=model_reduced 
# Nb joints = 8 (nq=7,nv=7)
#   Joint 0 universe: parent=0
#   Joint 1 r_shoulder_pitch: parent=0
#   Joint 2 r_shoulder_roll: parent=1
#   Joint 3 r_elbow_yaw: parent=2
#   Joint 4 r_elbow_pitch: parent=3
#   Joint 5 r_wrist_roll: parent=4
#   Joint 6 r_wrist_pitch: parent=5
#   Joint 7 r_wrist_yaw: parent=6

data = model.createData()

# Exemple de goal_pose (position et orientation)
# goal_position = np.array([0.4, -0.5, -0.2]) 
goal_position = np.array([-0.20, -0.39,  0.79])  # Translation, pris de l'exemple dessiner carré
goal_orientation = np.array([0.0, 0.0, 0.0])  # Angles d'Euler (en radians)

# Représentation goal_pose
goal_pose = np.array([goal_position, goal_orientation], dtype=float)
# -------------------------------------

# current_joints: list[float]
# current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# current_joints = pin.neutral(model)
current_joints = [2.726055496581969, -39.46273957135307, -12.748116930137442, -96.15064249358801, 
                  -2.0934329169282213, 11.418022966642084, -28.905708907141634]

# Afficher la transformation pour chaque joint (de oMi)
pin.forwardKinematics(model, data, np.array(current_joints))
for i, oMi in enumerate(data.oMi):
    print(f"Joint {i}: Transformation\n{oMi}")
# -----------------------------------------------------------------------

# The end effector corresponds to the 7th joint
JOINT_ID = 7

# Conversion des angles d'Euler en matrice de rotation
rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()
# à enlever après
rotation_matrix = np.array(
        [
            [0.302073,  0.898826, -0.317591],
            [0.1714, -0.378935, -0.909412],
            [-0.937749,  0.220274, -0.268524],
        ])

# Création de l'objet SE3
# and its desired pose is given as
# goal_pose_temp=np.array([0.4, -0.5, 0.0])
oMdes = pin.SE3(rotation_matrix, goal_pose[0]) # matrice de rotation et position

# q = pin.neutral(model)
q = np.array(current_joints) # plutot mettre là où on est actuellement
eps = 1e-4
IT_MAX = 100000 # original 1000
DT = 1e-4 # original e-1
damp = 1e-7 # original e-12


i = 0
list_val=[]
while True:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
    iMd = data.oMi[JOINT_ID].actInv(oMdes)  # Calcul de l'erreur

    # oMdes
    # SE3(array([[ 1. ,  0. ,  0. ,  0.4],[ 0. ,  1. ,  0. , -0.5],[ 0. ,  0. ,  1. ,  0. ],[ 0. ,  0. ,  0. ,  1. ]]))

    # data.oMi[JOINT_ID]
    # SE3(array([[ 9.65925856e-01, -2.54887032e-01,  4.49426576e-02, -7.03681117e-02],
    # [ 2.58818935e-01,  9.51251352e-01, -1.67730807e-01, -1.06070621e-01],
    # [ 6.43660370e-07,  1.73647534e-01,  9.84807867e-01,  4.48507627e-01],
    # [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]))

    # computing an error in SO(3) as a six-dimensional vector.
    err = pin.log(iMd).vector  # Erreur dans l'espace des twists

    if norm(err) < eps: # cas où ça converge :)
        success = True
        break
    if i >= IT_MAX: # on atteint la limite de boucle :(
        success = False
        break

    J = pin.computeJointJacobian(model, data, q, JOINT_ID)  # Jacobienne
    J = -np.dot(pin.Jlog6(iMd.inverse()), J)  # Jacobienne modifiée pour réduire l'erreur
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  # Résolution par moindres carrés
    q = pin.integrate(model, q, v * DT)  # Intégrer le déplacement des jointures

    if not i % 10:
        print(f"{i}: error = {err.T}") # pour avoir un suivi de ce qu'il se passe
    i += 1

    if not i % 100 : 
        list_val.append(err.T[0])


if success:
    print("Convergence achieved!")
    state = "converged"
    ik_joints = q.tolist()  # Résultats des positions des jointures
    is_reachable = True
else:
    print(
        "\nWarning: the iterative algorithm has not reached convergence "
        "to the desired precision"
    )
    state = "not converged"
    ik_joints = current_joints  # Retourner les positions actuelles si pas convergé
    is_reachable = False

plt.plot(list_val)
plt.show()