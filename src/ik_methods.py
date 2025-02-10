import time

import numpy as np
from numpy.linalg import norm, solve

from reachy2_sdk import ReachySDK

import pinocchio as pin 
from os.path import abspath

from typing import Tuple

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from reachy2_symbolic_ik.utils import limit_orbita3d_joints
import pink
import pink.limits
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, DampingTask
from pink.utils import custom_configuration_vector
import qpsolvers
from loop_rate_limiters import RateLimiter


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
    #if debug:
        #print("standard model: dim=" + str(len(model.joints)))

    # Create a list of joints to NOT lock
    jointsToNotLock = [f"{prefix}_shoulder_pitch",f"{prefix}_shoulder_roll", f"{prefix}_elbow_arm_link", 
                    f"{prefix}_elbow_yaw",f"{prefix}_elbow_pitch", f"{prefix}_wrist_roll", f"{prefix}_wrist_pitch", f"{prefix}_wrist_yaw"] 
    #if debug:
        #print(jointsToNotLock)
    #Get the Id of all existing joints
    jointsToLockIDs = []
    initialJointConfig = np.ones(len(model.joints)-1)
    i=-1

    for i, jn in enumerate(model.joints):  # enumerate pour obtenir l'index
        joint_name = model.names[i]  # nom du joint via model.names
        #if debug:
            #print(joint_name)
        if joint_name not in jointsToNotLock:  # Si le joint doit être verrouillé
            jointsToLockIDs.append(jn.id)  # jn.id pour l'ID du joint
            initialJointConfig[i-1] = 0  # Fixez la configuration initiale à 0

    #if debug:
        #print(jointsToLockIDs)

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


def get_current_joints(prefix : str) :

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

def get_joints_from_chosen_method(model, data, H_THd, prefix, method, blocked_joints=None):
    #print("test get_joints_from_chosen_method")

    if prefix == "r":
        shoulder_roll_lower_limit = -np.pi
        shoulder_roll_upper_limit = np.pi/6
    elif prefix == "l":
        shoulder_roll_lower_limit = -np.pi/6
        shoulder_roll_upper_limit = np.pi

    margin = 10e-3

    modify_joint_limit(model, "upper", indices=np.array([0,1,2,3,4,5,6]), margin=margin, values = np.array([
        6*np.pi, 
        shoulder_roll_upper_limit, 
        6*np.pi, 
        0.06, 
        np.pi/6, 
        np.pi/6, 
        6*np.pi]))
    modify_joint_limit(model, "lower", indices=np.array([0,1,2,3,4,5,6]), margin=margin, values = np.array([
        - 6*np.pi, 
        shoulder_roll_lower_limit,
        - 6*np.pi,
        - 2.26,
        - np.pi/6,
        - np.pi/6,
        - 6*np.pi])) 

    current_joints_rad = np.deg2rad(get_current_joints(prefix))

    # on change goal pose avant de la donner
    H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)

    if method == "pin" or method == "pinocchio" :
        # model:pin.Model, 
        # data:pin.Data, 
        # current_joints:np.array, 
        # goal_pose, 
        # matrix_rot, 
        # debug: bool=False, 
        # plot:bool=False:
        joint_rad= symbolic_inverse_kinematics_continuous_with_pinocchio(model, data, current_joints_rad, H_WPd, H_WPd[0:3,0:3], 
                                                                    debug=False, plot=False)
    elif method == "pink" : 
        joint_rad = symbolic_inverse_kinematics_continuous_with_pink_(model, data, current_joints_rad, H_WPd, H_WPd[0:3,0:3], 
                                                                    prefix, debug=False, plot=False, blocked_joints=blocked_joints)
    return np.rad2deg(joint_rad)

def sigmoid(max, seuil, stretch, x): return max/(1+np.exp(-(x-seuil)/stretch))
#La fonction sigmoid est une fonction d'activation étant égale  
def softplus(beta, seuil, x): return 1/beta*np.log(1+np.exp(beta*(x-seuil)))

def control_function(current_joint, activated):
    diff_joints = np.abs(current_joint)
    if activated: 
        return sigmoid(2, np.pi/2, 1, diff_joints)
    else: 
        return sigmoid(2, 5*np.pi, np.pi, diff_joints)

def symbolic_inverse_kinematics_continuous_with_pink_(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_pose, 
        matrix_rot, 
        prefix: str,
        rewind: bool=False,
        debug: bool=False, 
        plot:bool=False,
        blocked_joints: np.array = None,
        ):

    #if debug : 
        #print("--------------\nsymbolic_inverse_kinematics_continuous_with_pink\n--------------")
    
    # Début timer
    start_time = time.time()
    t = time.time()

    # The end effector corresponds to the 7th joint
    oMdes = pin.SE3(matrix_rot, goal_pose[0:3,3])
    link = f"{prefix}_wrist_yaw"
    elbow_link = f"{prefix}_elbow_yaw"
    configuration = pink.Configuration(model, data, current_joints) # configuration initiale

    # Paramètres
    eps_pos = 1e-3 # ordre du centimètre
    eps_or = 1e-3 # 0.01 rad 
    IT_MAX = 1 # original 1000
    i=0    

    #base_ratio = 5*10e-1
    base_ratio = 10e-3

    # tasks 
    end_effector_task = FrameTask(
            link,
            position_cost = 1000*base_ratio, # [cost] / [m]
            orientation_cost = base_ratio*180/np.pi,  # [cost] / [rad]
            lm_damping = 10e-7,  # tuned for this setup
        )   
    
    r_elbow_task = FrameTask(
            elbow_link,
            position_cost= 2*10e-4,
            #position_cost= 10e0,  # [cost] / [m]
            orientation_cost=10e-12,  # [cost] / [rad]
            lm_damping = 10e-7
        )  

    

    posture_task = PostureTask(
            cost= 2 * 10e-5
            #cost=10e-1,  # [cost] / [rad]
    )

    damping_task = DampingTask( 
        cost = 2 * 10e-6
        #cost = 10e-2
        )

    cost_zero_posture = control_function(current_joints[0], rewind)

    zero_posture = PostureTask(
        #cost = cost_zero_posture
        cost = [10, 0, 10, 10, 10, 10, 10]
    )

    if cost_zero_posture > 1.5:
        rewind = True
    else :
        rewind = False

    #tasks = [end_effector_task, r_elbow_task,  posture_task]
    tasks = [end_effector_task, damping_task, posture_task, zero_posture]

    q_limit = pink.limits.ConfigurationLimit(model, config_limit_gain=0.6)


    model.velocityLimit = np.ones(7)*100
    v_limit = pink.limits.VelocityLimit(model)

    # print(f"model velocity limit = {model.velocityLimit}")   
    # print(f"joints with velocity limits = {v_limit.joints}")

    a_limit = pink.limits.AccelerationLimit(model, np.ones(7)*100)
    #print(f"accelaration joints = {a_limit.indices}")

    # print(f"model limits {model.upperPositionLimit}")
    # print(f"model limits {model.lowerPositionLimit}")
    # print(f"q_limit.joints : {q_limit.joints}")

    goal_pose= oMdes.copy()
    end_effector_task.set_target(goal_pose)
    #end_effector_task.set_target(pin.SE3(np.eye(3), np.array([-2, -2, 2 ])))
    r_elbow_task.set_target(pin.SE3(np.eye(3), np.array([0.075, -0.5, -0.25 ])))
    posture_task.set_target_from_configuration(configuration)
    zero_posture.set_target(np.zeros(7))



    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    #print(f" ## dt : {dt}")
    
    current_joints_f = get_current_joints(prefix)
    print(f"q_shoulder_pitch {prefix} : {int(current_joints_f[0])}")
    print(f"q_shoulder_roll {prefix} : {int(current_joints_f[1])}")

    if prefix == "r":
        limits=[q_limit, v_limit, a_limit]
    else:
        limits=[q_limit, v_limit]
    
    while True:
        i+=1
        
        # Compute velocity and integrate it into next configuration
        velocity = pink.solve_ik(
            configuration, 
            tasks, 
            dt, 
            solver=solver, 
            limits=limits,
            damping=10e-8, 
            safety_break=False)

        # q = configuration.integrate(velocity, dt)
        # q_avant = q[4:7].copy()
        # q[4:7] = limit_orbita3d_joints(q[4:7], np.pi/4)
        # print(f" diff q {q[4:7] - q_avant}")
        # configuration.update(q)

        configuration.integrate_inplace(velocity, dt)

        # Calcul de l'erreur de position et d'orientation
        oMcurrent = configuration.get_transform_frame_to_world(link)
        pos_error = norm(oMcurrent.translation - oMdes.translation)
        rot_error = norm(pin.log(oMcurrent.rotation.T @ oMdes.rotation))

        if pos_error < eps_pos and rot_error < eps_or:
            success = True
            break

        if i >= IT_MAX: # on atteint la limite de boucle :(
            success = False
            break
        


        # Visualize result at fixed FPS
        rate.sleep()
        t += dt
        #print("configuration.q = ", configuration.q)
    
    #print("nombre d'iteration : ", i)
    elapsed_time = time.time() - start_time

    #print(f"Total time to converge: {elapsed_time:.2f} seconds")
    return configuration.q

def symbolic_inverse_kinematics_continuous_with_pinocchio(
        model:pin.Model, 
        data:pin.Data, 
        current_joints:np.array, 
        goal_pose, 
        matrix_rot, 
        debug: bool=False, 
        plot:bool=False):

    #if debug : 
        #print("--------------\nsymbolic_inverse_kinematics_continuous_with_pinocchio\n--------------")
    
    # Début timer
    start_time = time.time()

    # The end effector corresponds to the 7th joint
    JOINT_ID = len(current_joints)
    q = np.array(current_joints) 

    # Conversion des angles d'Euler en matrice de rotation
    # Pour code Pollen, a utiliser avec goal_pose_orientation
    # rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()

    # oMdes = pin.SE3(matrix_rot, goal_pose[0])
    oMdes = pin.SE3(matrix_rot, goal_pose[0:3,3]) # matrice de rotation et position

    # if debug : 
    #     # Afficher la transformation pour chaque joint (de oMi) au départ ! i = 0
    #     pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
    #     pin.forwardKinematics(model, data, q)
    #     for i, oMi in enumerate(data.oMi):
    #         print(f"Joint {i}: Transformation\n{oMi}")

    # Paramètres
    eps_pos = 1e-3 # ordre du centimètre
    eps_or = 1e-3 # 0.01 rad 
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
            #if debug : 
                #print(f"{i}: error = {err.T}") # pour avoir un suivi de ce qu'il se passe
            if plot : 
                for k in range (len(err.T)): 
                    list_val[k].append(err.T[k])
                # list_val.append(err.T[0]) # pour plot

        # A ajuster, peut être inutile
        # Affiner la recherche, réduire le pas DT        
        # if not i % 1000 and (i==10000):
        #     DT=DT*1e-1

        #     if debug : 
        #         #print("DT = ", DT)

        i += 1

    # Fin du chronométrage
    elapsed_time = time.time() - start_time

    if debug : 
        #print("----------\nq_f = ", q.tolist() )
        final_pose = data.oMi[JOINT_ID]
        #print(f"Final Pose: \n{final_pose}")
        #print(f"Goal Pose: \n{oMdes}")

    if success:
        #print("Convergence achieved!")
        state = "converged"
        ik_joints = q.tolist()  # Résultats des positions des jointures
        is_reachable = True
    else:
        #print(
        #     "\nWarning: the iterative algorithm has not reached convergence "
        #     "to the desired precision"
        # )
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

    #print(f"Total time to converge: {elapsed_time:.2f} seconds")
    return ik_joints
