import time

import numpy as np
from numpy.linalg import norm, solve

from reachy2_sdk import ReachySDK

import pinocchio as pin 
from os.path import abspath

from typing import Tuple

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import pink
import pink.limits
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, DampingTask
from pink.utils import custom_configuration_vector
from pink.barriers.body_spherical_barrier import BodySphericalBarrier
import qpsolvers
from loop_rate_limiters import RateLimiter

import sys
sys.path.append('/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/')
from CSVLogger import CSVLogger
from compute_metrics import compute_metrics


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
            initialJointConfig[i-1] = 0  # Fixez la config initiale à 0

    if debug:
        print(jointsToLockIDs)

    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)
    data = model_reduced.createData()

    return model_reduced, data

def modify_joint_limit(
        model : pin.Model, 
        extrema_name: str,
        indices : np.array,
        margin : float,
        values : np.array) :
    if extrema_name == "upper":
        for i in range(len(indices)):
            model.upperPositionLimit[indices[i]] = values[i] - margin
    if extrema_name == "lower":
        for i in range(len(indices)):
            model.lowerPositionLimit[indices[i]] = values[i] + margin
    #print(f"model limits {model.upperPositionLimit}")
    #print(f"model limits {model.lowerPositionLimit}")

def get_current_joints(reachy: ReachySDK, prefix : str="all") :

    if prefix=="r" or prefix=="l":

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
    
    elif prefix=="all":

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


def get_joints_from_chosen_method(reachy: ReachySDK, model, data, H_THd, prefix, method, d_min=0.20, blocked_joints=None):
    #print("test get_joints_from_chosen_method")

    plot = True
    debug = False

    margin = 10e-3

    shoulder_pitch = [ -6*np.pi, 6*np.pi ]
    shoulder_roll_r = [ -np.pi, np.pi/6 ]
    shoulder_roll_l = [ -np.pi/6, np.pi ]
    elbow_yaw = [ -6*np.pi, 6*np.pi ]
    elbow_pitch = [- 2.26, 0.06 ]
    wrist_roll = [- np.pi/6, np.pi/6 ]
    wrist_pitch = [- np.pi/6, np.pi/6 ]
    wrist_yaw = [ -6*np.pi, 6*np.pi ]


    if prefix !="all":
        if prefix == "r":
            shoulder_roll =  shoulder_roll_r
        elif prefix == "l":
            shoulder_roll =  shoulder_roll_l

        modify_joint_limit(model, "lower", indices=np.array([0, 1, 2, 3, 4, 5, 6]), margin=margin, values = np.array([
            shoulder_pitch[0], 
            shoulder_roll[0], 
            elbow_yaw[0], 
            elbow_pitch[0], 
            wrist_roll[0], 
            wrist_pitch[0], 
            wrist_yaw[0]]))
        modify_joint_limit(model, "upper", indices=np.array([0, 1, 2, 3, 4, 5, 6]), margin=margin, values = np.array([
            shoulder_pitch[1], 
            shoulder_roll[1], 
            elbow_yaw[1], 
            elbow_pitch[1], 
            wrist_roll[1], 
            wrist_pitch[1], 
            wrist_yaw[1]]))
    
    else:
        modify_joint_limit(model, "lower", indices=np.array([0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21]), margin=margin, values = np.array([
            shoulder_pitch[0], 
            shoulder_roll_l[0], 
            elbow_yaw[0], 
            elbow_pitch[0], 
            wrist_roll[0], 
            wrist_pitch[0], 
            wrist_yaw[0],

            shoulder_pitch[0], 
            shoulder_roll_r[0], 
            elbow_yaw[0], 
            elbow_pitch[0], 
            wrist_roll[0], 
            wrist_pitch[0], 
            wrist_yaw[0]]))

        modify_joint_limit(model, "upper", indices=np.array([0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21]), margin=margin, values = np.array([
            shoulder_pitch[1], 
            shoulder_roll_l[1], 
            elbow_yaw[1], 
            elbow_pitch[1], 
            wrist_roll[1], 
            wrist_pitch[1], 
            wrist_yaw[1],

            shoulder_pitch[1], 
            shoulder_roll_r[1], 
            elbow_yaw[1], 
            elbow_pitch[1], 
            wrist_roll[1], 
            wrist_pitch[1], 
            wrist_yaw[1]]))

    if method == "pin" or method == "pinocchio" :

        # on change goal pose avant de la donner
        H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)
        current_joints_rad = np.deg2rad(get_current_joints(reachy, prefix))
        joint_rad= symbolic_inverse_kinematics_continuous_with_pinocchio(model, data, current_joints_rad, H_WPd, H_WPd[0:3,0:3], 
                                                                    debug=False, plot=False)
    elif method == "pink" : 

        # on change goal pose avant de la donner
        H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)
        current_joints_rad = np.deg2rad(get_current_joints(reachy, prefix))
        joint_rad = symbolic_inverse_kinematics_continuous_with_pink(model, data, current_joints_rad, H_WPd, 
                                                                    prefix, plot=False, debug=False)
    elif method == "pink_sphere" : 

        goal_poses = []
        for pose in H_THd : # on peut faire rapide
            goal_poses.append(np.dot(np.dot(H_WT, pose), H_HP))

        current_joints_rad = np.deg2rad(get_current_joints(reachy))
        joint_rad = symbolic_inverse_kinematics_continuous_with_pink_merged(model, data, current_joints_rad, goal_poses, 
                                                                    prefix, debug=False, plot=False, d_min = d_min, blocked_joints=blocked_joints)
        
    else : 
        raise ValueError(f"'{method}' is not a valid method.")
    
    return np.rad2deg(joint_rad)
    
#############################################################################

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

        compute_metrics(goal_pose_torso, current_joints, prefix, "pinocchio", np.array(q.tolist()), pose_compute_torso, velocity = np.array(v) )

    
    return q.tolist()

def sigmoid(max, seuil, stretch, x): return max/(1+np.exp(-(x-seuil)/stretch))
#La fonction sigmoid est une fonction d'activation étant égale  
def softplus(beta, seuil, x): return 1/beta*np.log(1+np.exp(beta*(x-seuil)))

def control_function(current_joint, activated):
    diff_joints = np.abs(current_joint)
    if activated: 
        return sigmoid(2, np.pi/2, 1, diff_joints)
    else: 
        return sigmoid(2, 5*np.pi, np.pi, diff_joints)

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
    config = pink.Configuration(model, data, current_joints) # config initiale

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
    posture_task.set_target_from_configuration(config)


    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    print(f" ## dt : {dt}")

    # # Compute velocity and integrate it into next config le vrai
    velocity = pink.solve_ik(config, tasks, dt, solver=solver, safety_break =False)
    config.integrate_inplace(velocity, dt)

    elapsed_time = time.time() - start_time

    print(f"Total time to converge: {elapsed_time:.6f} seconds")

    if plot : 

        goal_pose_torso = np.dot(np.dot(np.linalg.inv(H_WT), goal_pose), np.linalg.inv(H_HP))
        pose_compute_world = config.get_transform(f"{prefix}_wrist_yaw", "world")
        pose_compute_torso = np.dot(np.dot(np.linalg.inv(H_WT), pose_compute_world), np.linalg.inv(H_HP))

        compute_metrics(goal_pose_torso, current_joints, prefix, "pink", config.q, pose_compute_torso, velocity = velocity)
    
    return config.q

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

    # Création d'une seule config pour les deux bras
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

    # Mise à jour de la config
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

        compute_metrics(goal_pose_torso_l, q[:7], "l", "pink_sphere", config.q[:7], pose_compute_l_torso, velocity = velocity[:7])
        compute_metrics(goal_pose_torso_r, q[15:22], "r", "pink_sphere", config.q[15:22], pose_compute_r_torso, velocity = velocity[15:22])

    q = config.q
    return q[:7], q[15:22]

#################################################################################################################################

def symbolic_inverse_kinematics_continuous_with_pink_merged(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_poses, 
        prefix: str,
        d_min = 0.2,
        rewind: bool=False,
        debug: bool=False, 
        plot:bool=False,
        blocked_joints: np.array = None,
        csv_filename = "/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/pink_sphere/metrics_pink_sphere.csv"
        ):

    if debug : 
        print("\n----- Inverse Kinematics with Pink -----\n")
    
    # Début timer
    start_time = time.time()
    t = time.time()

    # Création d'une seule config pour les deux bras
    config = pink.Configuration(model, data, current_joints) # config initiale

    #base_ratio = 5*10e-1
    base_ratio = 10e-3

    # tasks 

    # Création des tâches pour chaque bras
    # pour position_cost : 1000*base_ratio
    # pour orientation_cost : base_ratio*180/np.pi
    r_ee_task = FrameTask(
            "r_wrist_yaw",
            position_cost = 1000*base_ratio, # [cost] / [m]
            orientation_cost = base_ratio*180/np.pi,  # [cost] / [rad]
        )   

    oMdes_r = pin.SE3(goal_poses[1][:3,:3], goal_poses[1][0:3, 3])
    r_ee_task.set_target(oMdes_r)

    l_ee_task = FrameTask(
            "l_wrist_yaw",
            position_cost = 1000*base_ratio, # [cost] / [m]
            orientation_cost = base_ratio*180/np.pi,  # [cost] / [rad]
        ) 
    
    oMdes_l = pin.SE3(goal_poses[0][:3,:3], goal_poses[0][0:3, 3])
    l_ee_task.set_target(oMdes_l)

    # r_elbow_task = FrameTask(
    #         "r_elbow_yaw",
    #         position_cost= 2*10e-4,
    #         #position_cost= 10e0,  # [cost] / [m]
    #         orientation_cost=10e-12,  # [cost] / [rad]
    #         lm_damping = 10e-7
    #     )  
    # r_elbow_task.set_target(pin.SE3(np.eye(3), np.array([0.075, -0.5, -0.25 ])))

    # Ajouter une posture pour stabiliser le mouvement
    posture_task = PostureTask(
            cost= 2 * 10e-5
            #cost=10e-1,  # [cost] / [rad]
    )
    posture_task.set_target_from_configuration(config)

    cost_zero_posture = control_function(current_joints[0], rewind)
    # zero_posture = PostureTask(
    #     #cost = cost_zero_posture
    #     cost = [10, 0, 10, 10, 10, 10, 10]
    # )
    # zero_posture.set_target(np.zeros(27))

    if cost_zero_posture > 1.5:
        rewind = True
    else :
        rewind = False

    damping_task = DampingTask( 
        cost = 2 * 10e-6
        #cost = 10e-2
    )

    # Définition de la barrière de collision
    ee_barrier = BodySphericalBarrier(
        ("l_wrist_yaw", "r_wrist_yaw"),
        d_min= d_min,
        gain=100.0,
        safe_displacement_gain=1.0,
    )

    model.velocityLimit = np.ones(27)*100

    q_limit = pink.limits.ConfigurationLimit(model, config_limit_gain=0.6)
    v_limit = pink.limits.VelocityLimit(model)
    a_limit = pink.limits.AccelerationLimit(model, np.ones(27)*100)

    # Regrouper toutes les tâches
    tasks = [l_ee_task, r_ee_task, damping_task, posture_task]
    barriers = [ee_barrier]
    limits=[q_limit, v_limit, a_limit]

    # print(f"model velocity limit = {model.velocityLimit}")   
    # print(f"joints with velocity limits = {v_limit.joints}")

    
    #print(f"accelaration joints = {a_limit.indices}")

    # print(f"model limits {model.upperPositionLimit}")
    # print(f"model limits {model.lowerPositionLimit}")
    # print(f"q_limit.joints : {q_limit.joints}") 


    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    dt = 1e-2
    #print(f" ## dt : {dt}")
    
    # current_joints_f = get_current_joints(prefix)
    # print(f"q_shoulder_pitch {prefix} : {int(current_joints_f[0])}")
    # print(f"q_shoulder_roll {prefix} : {int(current_joints_f[1])}")
            
    # Compute velocity and integrate it into next config
    velocity = pink.solve_ik(
        config, 
        tasks, 
        dt, 
        solver=solver, 
        barriers=barriers,
        limits=limits,
        damping=10e-8, 
        safety_break=True)
    
    #input(f"velocity = {velocity}")

    # q = config.integrate(velocity, dt)
    # q_avant = q[4:7].copy()
    # q[4:7] = limit_orbita3d_joints(q[4:7], np.pi/4)
    # print(f" diff q {q[4:7] - q_avant}")
    # config.update(q)

    # Mise à jour de la config
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

        compute_metrics(goal_pose_torso_l, current_joints[:7], "l", "pink_sphere", config.q[:7], pose_compute_l_torso, velocity = velocity[:7])
        compute_metrics(goal_pose_torso_r, current_joints[15:22], "r", "pink_sphere", config.q[15:22], pose_compute_r_torso, velocity = velocity[15:22])

    q = config.q
    return q[:7], q[15:22]
