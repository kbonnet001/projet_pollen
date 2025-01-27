import logging
import time

import numpy as np
from numpy.linalg import norm, solve
import numpy.typing as npt

from reachy2_sdk import ReachySDK
from reachy2_sdk.reachy_sdk import GoToHomeId

import pinocchio as pin 
from os.path import abspath

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

def build_pose_matrix(x: float, y: float, z: float) -> npt.NDArray[np.float64]:
    # The effector is always at the same orientation in the world frame
    return np.array(
        [
            [0, 0, -1, x],
            [0, 1, 0, y],
            [1, 0, 0, z],
            [0, 0, 0, 1],
        ]
    )

def symbolic_inverse_kinematics_continuous_for_test(model, data, current_joints, goal_pose, matrix_rot, H_WT = H_WT, debug=False, plot=False):

    if debug : 
        print("--------------\nsymbolic_inverse_kinematics_continuous_for_test\n--------------")
    
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
    eps_pos = 1e-4 # ordre du centimètre
    eps_or = 1e-4 # 0.01 rad 
    IT_MAX = 2000 # original 1000
    DT = 1e-2 # original e-1
    damp = 1e-6 # original e-12

    if plot : 
        # list_val = []
        list_val = [[] for _ in range(6)]

    i = 0

    while True:
        pin.forwardKinematics(model, data, q) # Cinematique directe de q_i --> pose_i = OMi

        # On met dans le bon reférentiel : World --> Torso
        # for i in range(JOINT_ID) :
        #     data.oMi[i] = np.dot(np.linalg.inv(H_WT), data.oMi[i])

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
        # print("q_ = [ 1e ik =   [-0.36472107 -1.19028604 -0.040131   -1.94078744  0.29933633  0.71357533 -1.11834547]")
        final_pose = data.oMi[JOINT_ID]
        print(f"Final Pose: \n{final_pose}")
        print(f"Goal Pose: \n{oMdes}")

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
    return q.tolist()

def reduce_model_pin_arm(urdf_filename: str, prefix: str)-> pin.Model:
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
    model, _, _ = pin.buildModelsFromUrdf(urdf_filename)
    # Check dimensions of the original model
    # print("standard model: dim=" + str(len(model.joints)))

    # Create a list of joints to NOT lock
    jointsToNotLock = [f"{prefix}_shoulder_pitch",f"{prefix}_shoulder_roll", f"{prefix}_elbow_arm_link", 
                    f"{prefix}_elbow_yaw",f"{prefix}_elbow_pitch", f"{prefix}_wrist_roll", f"{prefix}_wrist_pitch", f"{prefix}_wrist_yaw"] 
    # print(jointsToNotLock)
    #Get the Id of all existing joints
    jointsToLockIDs = []
    initialJointConfig = np.ones(len(model.joints)-1)
    i=-1

    for i, jn in enumerate(model.joints):  # enumerate pour obtenir l'index
        joint_name = model.names[i]  # nom du joint via model.names
        # print(joint_name)
        if joint_name not in jointsToNotLock:  # Si le joint doit être verrouillé
            jointsToLockIDs.append(jn.id)  # jn.id pour l'ID du joint
            initialJointConfig[i-1] = 0  # Fixez la configuration initiale à 0

    # print(jointsToLockIDs)

    # Option 1: Only build the reduced model in case no display needed:
    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)

    return model_reduced


def get_current_joints() :

    return np.array([reachy.r_arm.shoulder.pitch.present_position,
    reachy.r_arm.shoulder.roll.present_position,
    reachy.r_arm.elbow.yaw.present_position,
    reachy.r_arm.elbow.pitch.present_position,
    reachy.r_arm.wrist.roll.present_position,
    reachy.r_arm.wrist.pitch.present_position,
    reachy.r_arm.wrist.yaw.present_position
    ])



def test_diff_pollen_pin(model, data, H_WT = H_WT, H_HP = H_HP):


    time.sleep(1)

    # joint = [0.0, -0.17453292519943295, -0.2617993877991494, 0.0, 0.0, 0.0, 0.0]
    q = get_current_joints()
    
    pin.forwardKinematics(model, data, np.deg2rad(q))

    for i, oMi in enumerate(data.oMi):
        print(f"Joint oMi {i}: Transformation\n{oMi}")


    # print("\n", np.dot(np.linalg.inv(H_WT), data.oMi[7]))

    # print("\n",np.dot(H_WT, np.linalg.inv(oMi)))

    pose_pollen = reachy.r_arm.forward_kinematics(q)
    print("\nPollen\n", pose_pollen) # on met rien il fait tout seul
    H_TP = np.dot(np.linalg.inv(H_WT), data.oMi[7])
    # print("\n H_TP :", H_TP)

    H_TPd = np.dot(pose_pollen, H_HP)
    # print("\nH_TPd", H_TPd)

    H_WPd = np.dot(np.dot(H_WT, pose_pollen), H_HP)
    print("\nH_WPd ", H_WPd)

    diff_rot = data.oMi[7].rotation - H_WPd[0:3, 0:3]
    diff_pos = (data.oMi[7].translation)- H_WPd[0:3,3]
    print("\ndiff_rot = ", diff_rot)
    print("\ndiff_pos = ", diff_pos)

    # diff_rot = data.oMi[7].rotation - pose_pollen[0:3, 0:3]
    # diff_pos = (data.oMi[7].translation)- pose_pollen[0:3,3]
    # print("\ndiff_rot = ", diff_rot)
    # print("\ndiff_pos = ", diff_pos)

    # diff_pos_2 = (H_TP[0:3,3])- pose_pollen[0:3,3]
    # print("\ndiff_pos_2 = ", diff_pos_2)

    # diff_pos_3 = (H_TP[0:3,3])- H_TPd[0:3,3]
    # print("\ndiff_pos_3 = ", diff_pos_3)



def test(model, data, pose_pollen, H_HP=H_HP) : 

    time.sleep(1)

    current_joints = get_current_joints()

    # on change goal pose avant de la donner
    H_WPd = np.dot(np.dot(H_WT, pose_pollen), H_HP)

    joint_pin = symbolic_inverse_kinematics_continuous_for_test(model, data, np.deg2rad(current_joints), H_WPd, H_WPd[0:3,0:3], debug=True, plot=False)
    print("joint_pin en rad = ", joint_pin)
    print("joint_pin en deg = ", np.rad2deg(joint_pin))

    pin.forwardKinematics(model, data, np.array(joint_pin))
    for i, oMi in enumerate(data.oMi):
        print(f"Joint oMi {i}: Transformation\n{oMi}")

    pollen_pose = reachy.r_arm.forward_kinematics(np.rad2deg(joint_pin))
    print("pollen pose = ", pollen_pose)
    
    return joint_pin



def draw_square(model, data, reachy: ReachySDK) -> None:
    # In A position, the effector is at (0.4, -0,5, -0.2) in the world frame
    # In B position, the effector is at (0.4, -0.5, 0) in the world frame
    # In C position, the effector is at (0.4, -0.3, 0.0) in the world frame
    # In D position, the effector is at (0.4, -0.3, -0.2) in the world frame
    # see https://docs.pollen-robotics.com/sdk/first-moves/kinematics/ for Reachy's coordinate system

    # time.sleep(10)


    # urdf_filename = abspath("/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf")

    # # Charger le modèle URDF
    # model = reduce_model_pin_arm(urdf_filename, "r")
    # data = model.createData()

    test_diff_pollen_pin(model, data)
    input("Bras le long du corps :")

    # Going from A to B
    target_pose = build_pose_matrix(0.4, -0.5, 0)
    current_pos = reachy.r_arm.forward_kinematics()
    
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    test_diff_pollen_pin(model, data)
    input("Pose B :")



    # -----
    # Going from B to C
    # target_pose = build_pose_matrix(0.4, -0.3, 0)
    for y in np.arange(-0.5, -0.25, 0.05):
        print("y =", y)
        target_pose = build_pose_matrix(0.4, y, 0)

        joints_pin_rad = test(model, data, target_pose)
        print("H_WPd = ", np.dot(np.dot(H_WT, target_pose), H_HP))

        reachy.r_arm.goto_joints(np.rad2deg(joints_pin_rad), 1, degrees=True)
        # time.sleep(2)
        # current_pos = reachy.r_arm.forward_kinematics()
        # current_joints = get_current_joints()
        # print("current_joints = ", current_joints)
        # print("joints_pin = ", np.rad2deg(joints_pin_rad))
        # input(" prochaine étape :")

    input("Pose C atteinte :")

    #----


    # -----
    # Going from B to C
    # target_pose = build_pose_matrix(0.4, -0.3, 0)
    # for y in np.arange(-0.5, -0.25, 0.05):
    target_pose = build_pose_matrix(0.4, -0.45, 0)
    # print("y =", y)
    # target_pose = build_pose_matrix(0.4, y, 0)
    current_pos = reachy.r_arm.forward_kinematics()
    # ik = reachy.r_arm.inverse_kinematics(target_pose)
    joints_pin = test(model, data, target_pose)
    print("H_WPd = ", np.dot(np.dot(H_WT, target_pose), H_HP))
    # pin.forwardKinematics(model, data, np.array(joints_pin))
    # for i, oMi in enumerate(data.oMi):
    #     print(f"Joint oMi {i}: Transformation\n{oMi}")
    input("on a calculer les joints avec pin, faisons maintenant avec pollen :")
    
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, 1, degrees=True)
    time.sleep(2)
    current_pos = reachy.r_arm.forward_kinematics()
    current_joints = get_current_joints()
    print("current_joints = ", current_joints)
    print("joints_pin = ", joints_pin)
    test_diff_pollen_pin(model, data)
    input("fin du tet :")

    # current_pos = reachy.r_arm.forward_kinematics()
    # print("Pose C: ", current_pos)

    # test(model, data, target_pose)
    # test_diff_pollen_pin(model, data)
    input("Pose C :")

    target_pose = build_pose_matrix(0.4, -0.3, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)
    input("Pose true C :")

    #----

    target_pose = build_pose_matrix(0.4, -0.3, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    # print("Pose B: ", current_pos)

    test_diff_pollen_pin(model, data)
    input("Pose C :")

    # Going from C to D
    target_pose = build_pose_matrix(0.4, -0.3, -0.2)
    # target_pose = build_pose_matrix(0.4, -0.3, 0)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    # print("Pose D: ", current_pos)

    # test(model, data, target_pose)
    test_diff_pollen_pin(model, data)
    input("Pose D :")

    # Going from D to A
    target_pose = build_pose_matrix(0.4, -0.5, -0.2)
    # target_pose = build_pose_matrix(0.4, -0.2, 0)
    # print("ik : ", ik)
    ik = reachy.r_arm.inverse_kinematics(target_pose)
    reachy.r_arm.goto_joints(ik, 2.0, degrees=True)

    current_pos = reachy.r_arm.forward_kinematics()
    # print("Pose A: ", current_pos)

    # test(model, data, target_pose)
    test_diff_pollen_pin(model, data)
    input("Pose A :")


def move_to_point_A(reachy: ReachySDK) -> None:
    # position of point A in space
    target_pose = build_pose_matrix(0.4, -0.5, -0.2)
    # get the position in the joint space
    joints_positions = reachy.r_arm.inverse_kinematics(target_pose)
    # move Reachy's right arm to this point
    move_id = reachy.r_arm.goto_joints(joints_positions, duration=2)

    # wait for movement to finish
    while not reachy.is_move_finished(move_id):
        time.sleep(0.1)


def wait_for_pose_to_finish(ids: GoToHomeId) -> None:
    # pose is a special move function that returns 3 move id

    while not reachy.is_move_finished(ids.head):
        time.sleep(0.1)

    while not reachy.is_move_finished(ids.r_arm):
        time.sleep(0.1)

    while not reachy.is_move_finished(ids.l_arm):
        time.sleep(0.1)


if __name__ == "__main__":
    print("Reachy SDK example: draw square")

    logging.basicConfig(level=logging.INFO)
    reachy = ReachySDK(host="localhost")

    if not reachy.is_connected:
        exit("Reachy is not connected.")

    print("Turning on Reachy")
    reachy.turn_on()

    time.sleep(0.2)

    # model pin de l'urdf réduit (1 bras)
    urdf_filename = abspath("/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf")

    # Charger le modèle URDF
    model = reduce_model_pin_arm(urdf_filename, "r")
    data = model.createData()

    print("Set to Zero pose ...")
    move_ids = reachy.set_pose("default")
    wait_for_pose_to_finish(move_ids)

    # print("Set to Elbow 90 pose ...")
    # move_ids = reachy.set_pose("elbow_90")
    # wait_for_pose_to_finish(move_ids)

    # print("Move to point A")
    # move_to_point_A(reachy)

    print("Draw a square with the right arm ...")
    draw_square(model, data, reachy)


    print("Set to Zero pose ...")
    move_ids = reachy.set_pose("default")
    wait_for_pose_to_finish(move_ids)

    print("Turning off Reachy")
    reachy.turn_off()

    time.sleep(0.2)

    exit("Exiting example")
