import time

import numpy as np
from numpy.linalg import solve

import pink.barriers

import pinocchio as pin 

from scipy.spatial.transform import Rotation as R

import pink
import qpsolvers
from loop_rate_limiters import RateLimiter

import sys
sys.path.append('/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/')
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

projet_pollen_folder_path = "/home/reachy/dev/reachy2_symbolic_ik/src"
    
#############################################################################

def symbolic_inverse_kinematics_continuous_with_pinocchio(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_pose, 
        prefix,
        plot = True,
        debug: bool=False,
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
    end_effector_task = pink.tasks.FrameTask(
            link,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
            lm_damping=1e-4,  # tuned for this setup
        )   
    posture_task = pink.tasks.PostureTask(
            cost=1e-3,  # [cost] / [rad]
    )
    
    # model.velocityLimit = np.ones(7)*100

    # q_limit = pink.limits.ConfigurationLimit(model, config_limit_gain=0.6)
    # v_limit = pink.limits.VelocityLimit(model)
    # a_limit = pink.limits.AccelerationLimit(model, np.ones(7)*100)

    # Ajout des tâches et barrières
    tasks = [end_effector_task, posture_task]

    end_effector_task.set_target(oMdes)
    posture_task.set_target_from_configuration(config)


    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period

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
        ):

    if debug:
        print("\n----- Inverse Kinematics with Pink Sphere-----\n")

    # Création d'une seule config pour les deux bras
    config = pink.Configuration(model, data, q)

    # Création des tâches pour chaque bras
    # pour position_cost : 1000*base_ratio
    # pour orientation_cost : base_ratio*180/np.pi
    base_ratio = 0.05
    task_l = pink.tasks.FrameTask("l_wrist_yaw", position_cost=1000*base_ratio, orientation_cost=base_ratio*180/np.pi) #2.87
    task_r = pink.tasks.FrameTask("r_wrist_yaw", position_cost=1000*base_ratio, orientation_cost=base_ratio*180/np.pi)

    oMdes_l = pin.SE3(goal_poses[0][:3,:3], goal_poses[0][0:3, 3])
    oMdes_r = pin.SE3(goal_poses[1][:3,:3], goal_poses[1][0:3, 3])

    task_l.set_target(oMdes_l)
    task_r.set_target(oMdes_r)

    # Ajouter une posture pour stabiliser le mouvement
    posture_task = pink.tasks.PostureTask(cost=1e-3)
    posture_task.set_target_from_configuration(config)

    # Définition de la barrière de collision
    ee_barrier = pink.barriers.BodySphericalBarrier(
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
    velocity = pink.solve_ik(config, tasks, dt, solver=solver, barriers=barriers)

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

def sigmoid(max, seuil, stretch, x): return max/(1+np.exp(-(x-seuil)/stretch))

def zero_posture_cost(current_joints, blocked_joints, activated=None):

    zero_posture_cost = np.ones(27)*10e-2

    for i in blocked_joints:
        zero_posture_cost[i] = 10

    # # Work in progress
    # # Cette partie permettrai d'avoir une fonction d'activation obligeant le robot à se rembobiner
    # # Cependant, il faudrait revoir la bonne correspondance des courbes avec les poids souhaités
    # # Ainsi que l'implémentation de la variable activated qui indiquerai si une des articulations a été récemment proches de ses limites

    # for i in [0, 2, 6, 15, 17, 21]:
    #     if activated[i]: 
    #         zero_posture_cost[i] += sigmoid(1, np.pi/2, 1, np.abs(current_joints[i]))
    #     else: 
    #         zero_posture_cost[i] += sigmoid(1, 5*np.pi, np.pi, np.abs(current_joints[i]))
    
    return zero_posture_cost

def symbolic_inverse_kinematics_continuous_with_pink_V2(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_poses, 
        d_min = 0.2,
        debug: bool=False, 
        plot:bool=False,
        blocked_joints: np.array = [],
        ):

    if debug : 
        print("\n----- Inverse Kinematics with Pink V2 -----\n")
    
    # Début timer
    start_time = time.time()
    t = time.time()

    # Création d'une seule config pour les deux bras
    config = pink.Configuration(model, data, current_joints) # config initiale

    # tasks 

    # Création des tâches pour chaque bras

    base_ratio = 5*10e-4

    r_ee_task = pink.tasks.FrameTask(
            "r_wrist_yaw",
            position_cost = 1000*base_ratio, # [cost] / [m]
            orientation_cost = base_ratio*180/np.pi,  # [cost] / [rad]
        )   

    oMdes_r = pin.SE3(goal_poses[1][:3,:3], goal_poses[1][0:3, 3])
    r_ee_task.set_target(oMdes_r)

    l_ee_task = pink.tasks.FrameTask(
            "l_wrist_yaw",
            position_cost = 1000*base_ratio, # [cost] / [m]
            orientation_cost = base_ratio*180/np.pi,  # [cost] / [rad]
        ) 
    
    oMdes_l = pin.SE3(goal_poses[0][:3,:3], goal_poses[0][0:3, 3])
    l_ee_task.set_target(oMdes_l)

    # r_elbow_task = pink.tasks.FrameTask(
    #         "r_elbow_yaw",
    #         position_cost= 2*10e-5,
    #         orientation_cost=10e-12,  # [cost] / [rad]
    #         lm_damping = 10e-7
    #     )  
    # r_elbow_task.set_target(pin.SE3(np.eye(3), np.array([0.075, -0.5, -0.25 ])))

    # Ajouter une posture pour stabiliser le mouvement
    posture_task = pink.tasks.PostureTask(
            cost= 2 * 10e-6 # [cost] / [rad] 
    )
    posture_task.set_target_from_configuration(config)

    posture_zero_task = pink.tasks.PostureTask(
        cost = zero_posture_cost(current_joints, blocked_joints)
    )
    posture_zero_task.set_target(np.zeros(27))

    damping_task = pink.tasks.DampingTask( 
        cost = 2 * 10e-7
        #cost = 10e-2
    )

    # Définition de la barrière de collision
    ee_barrier = pink.barriers.BodySphericalBarrier(
        ("l_wrist_yaw", "r_wrist_yaw"),
        d_min= d_min,
        gain=10.0,
        safe_displacement_gain=1.0,
    )

    model.velocityLimit = np.ones(27)*100

    q_limit = pink.limits.ConfigurationLimit(model, config_limit_gain=0.6)
    v_limit = pink.limits.VelocityLimit(model)
    a_limit = pink.limits.AccelerationLimit(model, np.ones(27)*1000)

    # Regrouper toutes les tâches
    tasks = [l_ee_task, r_ee_task, damping_task, posture_task, posture_zero_task]
    barriers = [ee_barrier]
    limits=[q_limit, v_limit, a_limit]

    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    dt = 1e-2
            
    # Compute velocity and integrate it into next config
    velocity = pink.solve_ik(
        config, 
        tasks, 
        dt, 
        solver=solver, 
        barriers=barriers,
        limits=limits,
        damping=10e-8, 
        safety_break=True) #original True
    
    #input(f"velocity = {velocity}")

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

        compute_metrics(goal_pose_torso_l, current_joints[:7], "l", "pink_V2", config.q[:7], pose_compute_l_torso, velocity = velocity[:7])
        compute_metrics(goal_pose_torso_r, current_joints[15:22], "r", "pink_V2", config.q[15:22], pose_compute_r_torso, velocity = velocity[15:22])

    q = config.q
    return q[:7], q[15:22]
