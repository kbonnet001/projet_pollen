# /home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/ik_pinocchio.py

import time

import numpy as np
from numpy.linalg import norm, solve

from reachy2_sdk import ReachySDK

import pinocchio as pin 
from os.path import abspath

from typing import Tuple

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import numpy.typing as npt

from reachy2_symbolic_ik.utils import (
    get_best_continuous_theta,
    get_best_theta_to_current_joints,
    get_euler_from_homogeneous_matrix,
    limit_theta_to_interval,
    tend_to_preferred_theta,
)

DEBUG = False


reachy = ReachySDK(host="localhost")

H_WT = np.array([
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

def get_current_joints(prefix) :

    reachy_arm = getattr(reachy, f"{prefix}_arm")

    return np.array([
    reachy_arm.shoulder.pitch.present_position,
    reachy_arm.shoulder.roll.present_position,
    reachy_arm.elbow.yaw.present_position,
    reachy_arm.elbow.pitch.present_position,
    reachy_arm.wrist.roll.present_position,
    reachy_arm.wrist.pitch.present_position,
    reachy_arm.wrist.yaw.present_position
    ])

def reduce_model_pin_arm(
        prefix: str, 
        urdf_filename: str = "/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf",
        debug: bool = False
        )-> Tuple[pin.Model, pin.Data]:
    """
    Create a reduced model of reachy.urdf with pinocchio from a chosen prefix

    Nb joints = 8 (nq=7,nv=7)
      Joint 0 universe: parent=0
      Joint 1 {prefix}_shoulder_pitch: parent=0
      Joint 2 {prefix}_shoulder_roll: parent=1
      Joint 3 {prefix}_elbow_yaw: parent=2
      Joint 4 {prefix}_elbow_pitch: parent=3
      Joint 5 {prefix}_wrist_roll: parent=4
      Joint 6 {prefix}_wrist_pitch: parent=5
      Joint 7 {prefix}_wrist_yaw: parent=6
      
      """

    # Goal: Build a reduced model from an existing URDF model by fixing the desired joints
    # at a specified position.
    model, _, _ = pin.buildModelsFromUrdf(abspath(urdf_filename))
    # Check dimensions of the original model
    if debug:
        print("standard model: dim=" + str(len(model.joints)))

    # Create a list of joints to NOT lock
    jointsToNotLock = [f"{prefix}_shoulder_pitch",f"{prefix}_shoulder_roll", f"{prefix}_elbow_arm_link", 
                    f"{prefix}_elbow_yaw",f"{prefix}_elbow_pitch", f"{prefix}_wrist_roll", f"{prefix}_wrist_pitch", f"{prefix}_wrist_yaw"] 
    if debug:
        print(jointsToNotLock)
    #Get the Id of all existing joints
    jointsToLockIDs = []
    initialJointConfig = np.ones(len(model.joints)-1)
    i=-1

    for i, jn in enumerate(model.joints):  # enumerate pour obtenir l'index
        joint_name = model.names[i]  # nom du joint via model.names
        if debug:
            print(joint_name)
        if joint_name not in jointsToNotLock:  # Si le joint doit être verrouillé
            jointsToLockIDs.append(jn.id)  # jn.id pour l'ID du joint
            initialJointConfig[i-1] = 0  # Fixez la configuration initiale à 0

    if debug:
        print(jointsToLockIDs)

    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)
    data = model_reduced.createData()

    return model_reduced, data

def symbolic_inverse_kinematics_continuous_pin(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_pose, 
        debug: bool=False, 
        plot:bool=False):

    if debug : 
        print("--------------\nsymbolic_inverse_kinematics_continuous_pin\n--------------")
    
    # Début timer
    start_time = time.time()

    # The end effector corresponds to the 7th joint
    JOINT_ID = len(current_joints)
    q = np.array(current_joints) 

    # Conversion des angles d'Euler en matrice de rotation
    # Pour code Pollen, a utiliser avec goal_pose_orientation
    matrix_rot = R.from_euler("xyz", goal_pose[1]).as_matrix()

    position = np.array(goal_pose[0]).reshape(3, 1)
    rotation_matrix = R.from_euler('xyz', goal_pose[1]).as_matrix()
    H_THd = np.vstack((np.hstack((rotation_matrix, position)), [0, 0, 0, 1]))

    H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)
    if debug:
        print("goal pose = ", goal_pose)
        print("H_THd=", H_THd)
        print("H_WPd=",H_WPd)

    # oMdes = pin.SE3(matrix_rot, goal_pose[0])
    oMdes = pin.SE3(H_WPd[0:3,0:3], H_WPd[0:3,3]) # matrice de rotation et position

    if debug : 
        # Afficher la transformation pour chaque joint (de oMi) au départ ! i = 0
        pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
        pin.forwardKinematics(model, data, q)
        for i, oMi in enumerate(data.oMi):
            print(f"Joint {i}: Transformation\n{oMi}")

    # Paramètres
    eps_pos = 1e-2 # ordre du centimètre
    eps_or = 1e-2 # 0.01 rad 
    IT_MAX = 2000 # original 1000
    DT = 1e-2 # original e-1
    damp = 1e-6 # original e-12

    if plot : 
        # list_val = []
        list_val = [[] for _ in range(6)]

    i = 0

    while True:
        pin.forwardKinematics(model, data, q) # Cinematique directe de q_i --> pose_i = OMi
        pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
        iMd = data.oMi[JOINT_ID].actInv(oMdes)  # Calcul de l'erreur pour atteindre la pose

        # computing an error in SO(3) as a six-dimensional vector.
        err = pin.log(iMd).vector  

        if norm(err[0:3]) < eps_pos and norm(err[3:]) < eps_or : # cas où ça converge :)
            success = True
            break
        if i >= IT_MAX: # on atteint la limite de boucle :(
            success = False
            break

        J = pin.computeJointJacobian(model, data, q, JOINT_ID)  # Jacobienne
        J = -np.dot(pin.Jlog6(iMd.inverse()), J)  # Jacobienne modifiée pour réduire l'erreur
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  # Résolution par moindres carrés
        q = pin.integrate(model, q, v * DT)  # Intégrer le déplacement des jointures

        if not i % 100 and (debug or plot):
            if debug : 
                print(f"{i}: error = {err.T}") # pour avoir un suivi de ce qu'il se passe
                print(f"Transformation\n{data.oMi[JOINT_ID]}")
            if plot : 
                for k in range (len(err.T)): 
                    list_val[k].append(err.T[k])
                # list_val.append(err.T[0]) # pour plot

        # A ajuster, peut être inutile
        # Affiner la recherche, réduire le pas DT        
        # if not i % 1000 and (i==10000):
        #     DT=DT*1e-1

        #     if debug : 
        #         print("DT = ", DT)

        i += 1

    # Fin du chronométrage
    elapsed_time = time.time() - start_time

    if debug : 
        print("----------\nq_f = ", q.tolist() )
        final_pose = data.oMi[JOINT_ID]
        print(f"Final Pose: \n{final_pose}")
        print(f"Goal Pose: \n{oMdes}")

    if success:
        print("Convergence achieved!")
        state = "converged"
        ik_joints = np.array(q.tolist())  # Résultats des positions des jointures
        is_reachable = True
    else:
        print(
            "\nWarning: the iterative algorithm has not reached convergence "
            "to the desired precision"
        )
        state = "not converged"

        ik_joints = current_joints  # Retourner les positions actuelles si pas convergé
        is_reachable = False

    if plot : 
        # plt.plot(list_val)
        # plt.show()
        # Tracé des graphes
        plt.figure(figsize=(12, 8))
        list_error = ["Error position x", "Error position y", "Error position z", 
                    "Error orientation x", "Error orientation y", "Error orientation z"]
        for k in range(6):
            plt.plot(list_val[k], label=list_error[k])
        plt.xlabel('Iteration (x100)')
        plt.ylabel('Error values')
        plt.title('Evolution of error values')
        plt.legend()
        plt.grid()
        plt.show()

    print(f"Total time to converge: {elapsed_time:.2f} seconds")
    return ik_joints, is_reachable, state


def get_joints_from_pin(model, data, H_THd, prefix):

    current_joints_rad = np.deg2rad(get_current_joints(prefix))

    # on change goal pose avant de la donner
    H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)

    joint_pin_rad, _, _ = symbolic_inverse_kinematics_continuous_pin(model, data, current_joints_rad, H_WPd, H_WPd[0:3,0:3], debug=False, plot=False)
    
    return np.rad2deg(joint_pin_rad)


