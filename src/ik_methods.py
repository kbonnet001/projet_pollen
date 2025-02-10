import time#

import numpy as np
from numpy.linalg import solve

from reachy2_sdk import ReachySDK

import pinocchio as pin 
from os.path import abspath

from scipy.spatial.transform import Rotation as R

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, FrameTask
import qpsolvers
from loop_rate_limiters import RateLimiter
from pink.barriers.body_spherical_barrier import BodySphericalBarrier

import sys
sys.path.append('/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/')
from CSVLogger import CSVLogger
from compute_metrics import compute_metrics


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

def reduce_model_pin_arm(
        prefix: str, 
        urdf_filename: str = "/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf",
        debug: bool = True
        )-> pin.Model:
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


def get_current_joints_all(reachy: ReachySDK) :

    return np.array([
    reachy.l_arm.shoulder.pitch.present_position,
    reachy.l_arm.shoulder.roll.present_position,
    reachy.l_arm.elbow.yaw.present_position,
    reachy.l_arm.elbow.pitch.present_position,
    reachy.l_arm.wrist.roll.present_position,
    reachy.l_arm.wrist.pitch.present_position,
    reachy.l_arm.wrist.yaw.present_position, 

    0,0,0,0,0, 

    0,0,0,

    reachy.r_arm.shoulder.pitch.present_position,
    reachy.r_arm.shoulder.roll.present_position,
    reachy.r_arm.elbow.yaw.present_position,
    reachy.r_arm.elbow.pitch.present_position,
    reachy.r_arm.wrist.roll.present_position,
    reachy.r_arm.wrist.pitch.present_position,
    reachy.r_arm.wrist.yaw.present_position,

    0,0,0,0,0,

    ])

def get_current_joints(reachy: ReachySDK, prefix) :

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

def get_joints_from_chosen_method(reachy: ReachySDK, model, data, H_THd, prefix, method):

    plot = True
    debug = False

    current_joints_rad = np.deg2rad(get_current_joints(reachy, prefix))

    # on change goal pose avant de la donner
    H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)

    if method == "pin" or method == "pinocchio" :
        joint_rad = symbolic_inverse_kinematics_continuous_with_pinocchio(model, data, current_joints_rad, H_WPd, prefix, plot, debug)

    elif method == "pink" : 
        joint_rad = symbolic_inverse_kinematics_continuous_with_pink(model, data, current_joints_rad, H_WPd, prefix, plot, debug)
    
    else : 
        raise ValueError(f"'{method}' is not a valid method.")

    return np.rad2deg(joint_rad)


def get_joints_from_chosen_method_poses(reachy: ReachySDK, model, data, poses, method, d_min) : 

    q = np.deg2rad(get_current_joints_all(reachy))

    goal_poses = []
    for pose in poses : # on peut faire rapide
        goal_poses.append(np.dot(np.dot(H_WT, pose), H_HP))


    joint_rad = symbolic_inverse_kinematics_continuous_with_pink_sphere(
    model, data,
    q,
    goal_poses,
    solver="osqp", d_min=d_min, plot = True, debug=False)

    return np.rad2deg(joint_rad[:7]), np.rad2deg(joint_rad[15:22])

############################################################################################################################################
def symbolic_inverse_kinematics_continuous_with_pinocchio(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_pose, 
        prefix,
        plot = True,
        debug: bool=False,
        csv_filename = "/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/pinocchio/metrics_pinocchio_{prefix}.csv"
        ):

    if debug : 
        print("--------------\nsymbolic_inverse_kinematics_continuous_with_pinocchio\n--------------")
    
    # Début timer
    start_time = time.time()

    # The end effector corresponds to the 7th joint
    JOINT_ID = len(current_joints)
    q = np.array(current_joints) 

    # oMdes = pin.SE3(matrix_rot, goal_pose[0])
    oMdes = pin.SE3(goal_pose[:3,:3], goal_pose[0:3,3]) # matrice de rotation et position

    if debug : 
        # Afficher la transformation pour chaque joint (de oMi) au départ ! i = 0
        pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
        pin.forwardKinematics(model, data, q)
        for i, oMi in enumerate(data.oMi):
            print(f"Joint {i}: Transformation\n{oMi}")

    # Paramètres
    DT = 1e-1 # original e-1
    damp = 1e-6 # original e-12

    pin.forwardKinematics(model, data, q) # Cinematique directe de q_i --> pose_i = OMi
    pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
    iMd = data.oMi[JOINT_ID].actInv(oMdes)  # Calcul de l'erreur pour atteindre la pose

    # computing an error in SO(3) as a six-dimensional vector.
    err = pin.log(iMd).vector  

    J = pin.computeJointJacobian(model, data, q, JOINT_ID)  # Jacobienne
    J = -np.dot(pin.Jlog6(iMd.inverse()), J)  # Jacobienne modifiée pour réduire l'erreur
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  # Résolution par moindres carrés
    q = pin.integrate(model, q, v * DT)  # Intégrer le déplacement des jointures

    # Fin du chronométrage
    elapsed_time = time.time() - start_time

    print(f"Total time to converge: {elapsed_time:.6f} seconds")

    if plot : 

        goal_pose_torso = np.dot(np.dot(np.linalg.inv(H_WT), goal_pose), np.linalg.inv(H_HP))

        pin.forwardKinematics(model, data, np.array(q.tolist()))
        pin.updateFramePlacements(model, data)
        pose_compute_torso = np.dot(np.dot(np.linalg.inv(H_WT), data.oMi[JOINT_ID].homogeneous), np.linalg.inv(H_HP))

        compute_metrics(goal_pose_torso, current_joints, prefix, "pinocchio", np.array(q.tolist()), pose_compute_torso)

    
    return q.tolist()

#############################################################################

def symbolic_inverse_kinematics_continuous_with_pink(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_pose, 
        prefix: str,
        plot=False,
        debug: bool=False):

    if debug : 
        print("--------------\nsymbolic_inverse_kinematics_continuous_with_pink\n--------------")
    
    # Début timer
    start_time = time.time()

    # The end effector corresponds to the 7th joint
    oMdes = pin.SE3(goal_pose[:3,:3], goal_pose[0:3,3])
    link = f"{prefix}_wrist_yaw"
    configuration = pink.Configuration(model, data, current_joints) # configuration initiale

    # tasks 
    end_effector_task = pink.FrameTask(
            link,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=1e-4,  # tuned for this setup
        )   
    posture_task = pink.PostureTask(
            cost=1e-3,  # [cost] / [rad]
    )
    
    # Ajout des tâches et barrières
    tasks = [end_effector_task, posture_task]

    end_effector_task.set_target(oMdes)
    posture_task.set_target_from_configuration(configuration)


    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    print(f" ## dt : {dt}")

    # # Compute velocity and integrate it into next configuration le vrai
    velocity = pink.solve_ik(configuration, tasks, dt, solver=solver, safety_break =False)
    configuration.integrate_inplace(velocity, dt)

    elapsed_time = time.time() - start_time

    print(f"Total time to converge: {elapsed_time:.6f} seconds")

    if plot : 

        goal_pose_torso = np.dot(np.dot(np.linalg.inv(H_WT), goal_pose), np.linalg.inv(H_HP))
        pose_compute_world = configuration.get_transform(f"{prefix}_wrist_yaw", "world")
        pose_compute_torso = np.dot(np.dot(np.linalg.inv(H_WT), pose_compute_world), np.linalg.inv(H_HP))

        compute_metrics(goal_pose_torso, current_joints, prefix, "pink", configuration.q, pose_compute_torso)
    
    return configuration.q

#########################################################################################

def symbolic_inverse_kinematics_continuous_with_pink_sphere(
        model, data,
        q,
        goal_poses,
        solver="osqp", d_min = 0.2, plot=False, debug=False, 
        csv_filename = "/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/pink_sphere/metrics_pink_sphere.csv"
        ):

    if debug:
        print("\n----- Inverse Kinematics with Pink -----\n")

    # Création d'une seule configuration pour les deux bras
    config = pink.Configuration(model, data, q)

    # Création des tâches pour chaque bras
    # pour position_cost : 1000*base_ratio
    # pour orientation_cost : base_ratio*180/np.pi
    base_ratio = 0.05
    task_l = FrameTask("l_wrist_yaw", position_cost=1000*base_ratio, orientation_cost=base_ratio*180/np.pi) #2.87
    task_r = FrameTask("r_wrist_yaw", position_cost=1000*base_ratio, orientation_cost=base_ratio*180/np.pi)

    oMdes_l = pin.SE3(goal_poses[0][:3,:3], goal_poses[0][0:3, 3])
    oMdes_r = pin.SE3(goal_poses[1][:3,:3], goal_poses[1][0:3, 3])

    task_l.set_target(oMdes_l)
    task_r.set_target(oMdes_r)

    # Ajouter une posture pour stabiliser le mouvement
    posture_task = PostureTask(cost=1e-3)
    posture_task.set_target_from_configuration(config)

    # Définition de la barrière de collision
    ee_barrier = BodySphericalBarrier(
        ("l_wrist_yaw", "r_wrist_yaw"),
        d_min= d_min,
        gain=100.0,
        safe_displacement_gain=1.0,
    )

    # Regrouper toutes les tâches
    tasks = [task_l, task_r, posture_task]
    barriers = [ee_barrier]

    # Solveur QP
    dt = 1e-2
    velocity = solve_ik(config, tasks, dt, solver=solver, barriers=barriers)
    input(f"velocity = {velocity}")

    # Mise à jour de la configuration
    config.integrate_inplace(velocity, dt)

    if debug : 
        print(f"config.q =  {config.q}")

    if plot : 

        goal_pose_torso_l = np.dot(np.dot(np.linalg.inv(H_WT), goal_poses[0]), np.linalg.inv(H_HP))
        goal_pose_torso_r = np.dot(np.dot(np.linalg.inv(H_WT), goal_poses[1]), np.linalg.inv(H_HP))

        pose_compute_l_world = config.get_transform(f"l_wrist_yaw", "world")
        pose_compute_l_torso = np.dot(np.dot(np.linalg.inv(H_WT), pose_compute_l_world), np.linalg.inv(H_HP))

        pose_compute_r_world = config.get_transform(f"r_wrist_yaw", "world")
        pose_compute_r_torso = np.dot(np.dot(np.linalg.inv(H_WT), pose_compute_r_world), np.linalg.inv(H_HP))

        compute_metrics(goal_pose_torso_l, q[:7], "l", "pink_sphere", config.q[:7], pose_compute_l_torso)
        compute_metrics(goal_pose_torso_r, q[15:22], "r", "pink_sphere", config.q[15:22], pose_compute_r_torso)


    return config.q



