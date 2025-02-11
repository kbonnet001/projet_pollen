
import time

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R

from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix
from ik_methods import get_joints_from_chosen_method, reduce_model_pin_arm, get_current_joints
from compute_metrics import compute_metrics


import pinocchio as pin
from os.path import abspath
import os


PLOT = False


def load_models(method, arm = ["l", "r"]) : 

    path_urdf = "/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf"

    model = {}
    data = {}

    if method == "pinocchio" or method == "pink" : 
        for prefix in arm : 
            model[prefix], data[prefix] = reduce_model_pin_arm(prefix)

    elif method == "pink_sphere" : 
        model, _, _ = pin.buildModelsFromUrdf(abspath(path_urdf))
        data = model.createData()

    return model, data

def go_to_pose(reachy: ReachySDK, pose: npt.NDArray[np.float64], arm: str, method:str, 
               model, data) -> None:

    if method == "pink_sphere"  or method=="pink_final": 
        ik = get_joints_from_chosen_method(reachy, model, data, pose, "all", method)
        
        for joint, goal_pos in zip(reachy.l_arm.joints.values(), ik[0]):
            
            joint.goal_position = goal_pos
        for joint, goal_pos in zip(reachy.r_arm.joints.values(), ik[1]):
            joint.goal_position = goal_pos

    elif method == "pollen" : 
        ik = reachy_arm.inverse_kinematics(pose)

        if PLOT : 
            current_joint = np.deg2rad(get_current_joints(reachy, arm[0]))
            pose_pollen = reachy_arm.forward_kinematics(ik)
            compute_metrics(pose, current_joint, arm[0], "pollen", np.deg2rad(ik), pose_pollen, velocity = [])


    elif method == "pink" or method == "pinocchio":
        reachy_arm = getattr(reachy, arm)
        ik = get_joints_from_chosen_method(reachy, model[arm[0]], data[arm[0]], pose, arm[0], method)
        for joint, goal_pos in zip(reachy_arm.joints.values(), ik):
            joint.goal_position = goal_pos


    else : 
        raise ValueError(f"'{method}' is not a valid method.")

    # real_pose = reachy.r_arm.forward_kinematics(ik)
    # pose_diff = np.linalg.norm(pose - real_pose)
    # print(f"pose diff {pose_diff}")
    

    reachy.send_goal_positions()

def move_to_first_point(reachy: ReachySDK, orientation, position, prefix) -> None:
    # position of point A in space
    rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
    target_pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)

    # get the position in the joint space
    reachy_arm = getattr(reachy, prefix)
    joints_positions = reachy_arm.inverse_kinematics(target_pose)
    # move Reachy's right arm to this point
    #print("Pollen Joint", joints_positions)
    reachy_arm.goto_joints(joints_positions, duration=2)
    # do time.sleep 


def make_line(
    reachy: ReachySDK, start_pose: npt.NDArray[np.float64], end_pose: npt.NDArray[np.float64], method, model, data, nbr_points: int = 100
) -> None:
    start_position = start_pose[0]
    end_position = end_pose[0]
    start_orientation = start_pose[1]
    end_orientation = end_pose[1]

    # Left arm
    l_start_position = np.array([start_position[0], -start_position[1], start_position[2]])
    l_end_position = np.array([end_position[0], -end_position[1], end_position[2]])
    l_start_orientation = np.array([-start_orientation[0], start_orientation[1], -start_orientation[2]])
    l_end_orientation = np.array([-end_orientation[0], end_orientation[1], -end_orientation[2]])

    for i in range(nbr_points):
        r_position = start_position + (end_position - start_position) * (i / nbr_points)
        orientation = start_orientation + (end_orientation - start_orientation) * (i / nbr_points)
        r_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
        r_pose = make_homogenous_matrix_from_rotation_matrix(r_position, r_rotation_matrix)
        

        l_position = l_start_position + (l_end_position - l_start_position) * (i / nbr_points)
        l_orientation = l_start_orientation + (l_end_orientation - l_start_orientation) * (i / nbr_points)
        l_rotation_matrix = R.from_euler("xyz", l_orientation).as_matrix()
        l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)

        if method != "pink_sphere" : 

            go_to_pose(reachy, r_pose, "r_arm", method, model, data)
            go_to_pose(reachy, l_pose, "l_arm", method, model, data)

        else :
            go_to_pose(reachy, [l_pose, r_pose], "all", method, model, data)

        time.sleep(0.01)
        # if i ==0 :
        #     time.sleep(3)


def make_circle(
    reachy: ReachySDK,
    center: npt.NDArray[np.float64],
    orientation: npt.NDArray[np.float64],
    radius: float,
    method,
    model, 
    data,
    nbr_points: int = 100,
    number_of_turns: int = 3,
    
) -> None:

    Y_r = center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))
    Z = center[2] + radius * np.sin(np.linspace(0, 2 * np.pi, nbr_points))
    X = center[0] * np.ones(nbr_points)
    Y_l = -center[1] + radius * np.cos(np.linspace(0, 2 * np.pi, nbr_points))

    for k in range(number_of_turns):
        for i in range(nbr_points):
            r_position = np.array([X[i], Y_r[i], Z[i]])
            r_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            r_pose = make_homogenous_matrix_from_rotation_matrix(r_position, r_rotation_matrix)
            

            l_position = np.array([X[i], Y_l[i], Z[i]])
            l_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)

            if method != "pink_sphere":
                go_to_pose(reachy, r_pose, "r_arm", method, model, data)
                go_to_pose(reachy, l_pose, "l_arm", method, model, data)
            else:
                go_to_pose(reachy, [l_pose, r_pose], "all", method, model, data)

            time.sleep(0.01)


def make_rectangle(
    reachy: ReachySDK,
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    D: npt.NDArray[np.float64],
    method,
    model, 
    data,
    nbr_points: int = 20,
    number_of_turns: int = 3,
) -> None:
    orientation = [0, -np.pi / 2, 0]

    # Go to the firt point A (pollen)
    #move_to_first_point(reachy, orientation, A, "r_arm")
    #move_to_first_point(reachy, orientation, np.array([A[0], -A[1], A[2]]), "l_arm")
    time.sleep(2)

    for i in range(number_of_turns):
        make_line(reachy, np.array([A, orientation]), np.array([B, orientation]), method, model, data, nbr_points)
        make_line(reachy, np.array([B, orientation]), np.array([C, orientation]), method, model, data, nbr_points)
        make_line(reachy, np.array([C, orientation]), np.array([D, orientation]), method, model, data, nbr_points)
        make_line(reachy, np.array([D, orientation]), np.array([A, orientation]), method, model, data, nbr_points)


def turn_hand(reachy: ReachySDK, position: npt.NDArray[np.float64], orientation_init: npt.NDArray[np.float64], method, model, data) -> None:
    orientation = [orientation_init[0], orientation_init[1], orientation_init[2]]
    for j in range(2):
        for i in range(100):
            rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            pose = make_homogenous_matrix_from_rotation_matrix(position, rotation_matrix)
            print(pose)
            go_to_pose(reachy, pose, "r_arm", method, model, data)
            time.sleep(0.05)
            orientation[2] += 2 * np.pi / 100
            if orientation[2] > np.pi:
                orientation[2] = -np.pi
            print(orientation)


def test_sphere(
    reachy: ReachySDK,
    method,
    model, 
    data,
    nbr_points: int = 50,
) -> None:
    
    orientation = [0, -np.pi / 2, 0] # orientation

    A = np.array([0.4, -0.5, -0.2]) 
    B = np.array([0.4, -0.01, -0.2]) # on va sur le centre de torse, les deux mains
    C = np.array([0.4, -0.5, 0.1]) 
    D = np.array([0.4, -0.01, -0.1]) 

    # Go to the firt point A (pollen)
    move_to_first_point(reachy, orientation, A, "r_arm")
    move_to_first_point(reachy, orientation, np.array([A[0], -A[1], A[2]]), "l_arm")
    time.sleep(2)

    make_line(reachy, np.array([A, orientation]), np.array([B, orientation]), method, model, data, nbr_points)
    time.sleep(3)
    make_line(reachy, np.array([B, orientation]), np.array([A, orientation]), method, model, data, nbr_points)
    time.sleep(3)
    make_line(reachy, np.array([A, orientation]), np.array([C, orientation]), method, model, data, nbr_points)
    time.sleep(3)
    make_line(reachy, np.array([C, orientation]), np.array([D, orientation]), method, model, data, nbr_points)
    time.sleep(3)
    make_line(reachy, np.array([D, orientation]), np.array([C, orientation]), method, model, data, nbr_points)
    time.sleep(3)



def make_semi_circle_z(
    reachy: ReachySDK,
    method,
    model, 
    data,
    radius,
    z = -0.2,
    nbr_points: int = 100,
    number_of_turns = 1, 
) -> None:
    
    orientation = [0, -np.pi / 2, 0] # orientation
    radius = 0.7

    point_start = np.array([0.6, -0.2, -0.1])
    # Go to the firt point (pollen)
    move_to_first_point(reachy, orientation, point_start, "r_arm")
    move_to_first_point(reachy, orientation, np.array([point_start[0], -point_start[1], point_start[2]]), "l_arm")
    time.sleep(2)


    center=np.array([0.0, -0.2, z])

    Y_r = center[1] + radius * - np.sin(np.linspace(0, np.pi, nbr_points))
    Z = center[2] * np.ones(nbr_points)
    X = center[0] + radius * np.cos(np.linspace(0, np.pi, nbr_points))
    Y_l = -center[1] + radius * np.sin(np.linspace(0, np.pi, nbr_points))


    for k in range(number_of_turns):
        for i in range(nbr_points):
            r_position = np.array([X[i], Y_r[i], Z[i]])
            r_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            r_pose = make_homogenous_matrix_from_rotation_matrix(r_position, r_rotation_matrix)

            l_position = np.array([X[i], Y_l[i], Z[i]])
            l_rotation_matrix = R.from_euler("xyz", orientation).as_matrix()
            l_pose = make_homogenous_matrix_from_rotation_matrix(l_position, l_rotation_matrix)
            
            if method != "pink_sphere":
                go_to_pose(reachy, r_pose, "r_arm", method, model, data)
                go_to_pose(reachy, l_pose, "l_arm", method, model, data)

            else:
                go_to_pose(reachy, [l_pose, r_pose], "all", method, model, data)

            print("position = ", X[i], Y_r[i], Z[i])

            time.sleep(0.01)


def move_q(
model,
data,
reachy : ReachySDK,
indices : np.array,
new_q : np.array,
prefix : str
) -> None:
    new_joints = get_current_joints(reachy, prefix)
    j=0
    for i in indices:
        new_joints[i] =  new_q[j]
        #print(f"new_joint {i}= {int(new_joints[i])}")
        j+=1

    arm = f"{prefix}_arm"
    reachy_arm = getattr(reachy, arm)
    pose = reachy_arm.forward_kinematics(new_joints)

    go_to_pose(reachy, np.array(pose), arm, "pink", model, data)
    time.sleep(0.001)

def move_q_all(
model,
data,
reachy : ReachySDK,
indices : np.array,
new_q : np.array,
) -> None:
    new_joints = get_current_joints(reachy)
    j=0
    for i in indices:
        new_joints[i] =  new_q[j]
        #print(f"new_joint {i}= {int(new_joints[i])}")
        j+=1

    l_pose = reachy.l_arm.forward_kinematics(new_joints[:7])
    r_pose = reachy.r_arm.forward_kinematics(new_joints[15:22])

    go_to_pose(reachy, np.array([l_pose, r_pose]), "all", "pink", model, data)
    time.sleep(0.001)

##########################################

def main_test() -> None:
    print("Trying to connect on localhost Reachy...")
    reachy = ReachySDK(host="localhost")

    time.sleep(1.0)
    if reachy._grpc_status == "disconnected":
        print("Failed to connect to Reachy, exiting...")
        return

    reachy.turn_on()

    print("Set to Zero pose ...")
    reachy.set_pose("default")
    time.sleep(3)

    #############################
    #############################

    method = "pink_sphere" #pollen, pinocchio, pink, pink_sphere

    model, data = load_models(method)
    
    path = f"/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/{method}/metrics_{method}.csv"
    folder_path = f"/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/{method}"

    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

    make_semi_circle_z(reachy, method, model, data, radius=0.2, nbr_points= 50)
    input("fin")


    #     ############

    if method == "pink_sphere" : 

        # print("Test pink with barrier sphere)")
        # test_sphere(reachy, method, model, data)
        
        print("Making a rectangle (pink sphere)")
        A = np.array([0.4, -0.5, -0.3])
        B = np.array([0.4, -0.5, -0.1])
        C = np.array([0.4, -0.2, -0.1])
        D = np.array([0.4, -0.2, -0.3])
        make_rectangle(reachy, A, B, C, D, method, model, data, number_of_turns=1)

    else : 

        # print("Making a circle")
        # center = np.array([0.4, -0.4, -0.2])
        # orientation = np.array([0, -np.pi / 2, 0])
        # radius = 0.10
        # make_circle(reachy, center, orientation, radius, method, model, data, number_of_turns=1)

        # time.sleep(1.0)

        print("Making a rectangle")
        A = np.array([0.4, -0.5, -0.3])
        B = np.array([0.4, -0.5, -0.1])
        C = np.array([0.4, -0.2, -0.1])
        D = np.array([0.4, -0.2, -0.3])
        make_rectangle(reachy, A, B, C, D, method, model, data, number_of_turns=1)

        time.sleep(1.0)

    print("Finished testing, disconnecting from Reachy...")
    time.sleep(0.5)
    reachy.disconnect()


if __name__ == "__main__":
    main_test()
