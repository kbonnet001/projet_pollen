import numpy as np

from reachy2_sdk import ReachySDK

import pinocchio as pin 
from os.path import abspath

from scipy.spatial.transform import Rotation as R

import ik_methods

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

def load_models(method, arm = ["l", "r"]) : 

    path_urdf = "/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf"

    model = {}
    data = {}

    if method == "pinocchio" or method == "pink" : 
        for prefix in arm : 
            model[prefix], data[prefix] = reduce_model_pin_arm(prefix)

    elif method == "pink_V2" : 
        model, _, _ = pin.buildModelsFromUrdf(abspath(path_urdf))
        data = model.createData()

    return model, data

def modify_joint_limit(
        model : pin.Model, 
        indices : np.array,
        margin : float,
        values : np.array) :

    for i in range(len(indices)):
            model.lowerPositionLimit[indices[i]] = values[i][0] + margin
            model.upperPositionLimit[indices[i]] = values[i][1] - margin

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


def get_joints_from_chosen_method(reachy: ReachySDK, model, data, H_THd, prefix, method, d_min=0.20, blocked_joints=[]):

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

        modify_joint_limit(model, indices=np.array([0, 1, 2, 3, 4, 5, 6]), margin=margin, values = np.array([
            shoulder_pitch, 
            shoulder_roll, 
            elbow_yaw, 
            elbow_pitch, 
            wrist_roll, 
            wrist_pitch, 
            wrist_yaw]))
    
    else:
        modify_joint_limit(model, indices=np.array([0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20, 21]), margin=margin, values = np.array([
            shoulder_pitch, 
            shoulder_roll_l, 
            elbow_yaw, 
            elbow_pitch, 
            wrist_roll, 
            wrist_pitch, 
            wrist_yaw,

            shoulder_pitch, 
            shoulder_roll_r, 
            elbow_yaw, 
            elbow_pitch, 
            wrist_roll, 
            wrist_pitch, 
            wrist_yaw]))

    if method == "pin" or method == "pinocchio" :

        # on change goal pose avant de la donner
        H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)
        current_joints_rad = np.deg2rad(get_current_joints(reachy, prefix))
        joint_rad= ik_methods.symbolic_inverse_kinematics_continuous_with_pinocchio(model, data, current_joints_rad, H_WPd, H_WPd[0:3,0:3], 
                                                                    debug=False, plot=False)
    elif method == "pink" : 

        # on change goal pose avant de la donner
        H_WPd = np.dot(np.dot(H_WT, H_THd), H_HP)
        current_joints_rad = np.deg2rad(get_current_joints(reachy, prefix))
        joint_rad = ik_methods.symbolic_inverse_kinematics_continuous_with_pink(model, data, current_joints_rad, H_WPd, 
                                                                    prefix, plot=False, debug=False)
    elif method == "pink_V2" : 

        goal_poses = []
        for pose in H_THd : # on peut faire rapide
            goal_poses.append(np.dot(np.dot(H_WT, pose), H_HP))

        current_joints_rad = np.deg2rad(get_current_joints(reachy))
        joint_rad = ik_methods.symbolic_inverse_kinematics_continuous_with_pink_V2(model, data, current_joints_rad, goal_poses, 
                                                                    debug=False, plot=False, d_min = d_min, blocked_joints=blocked_joints)
        
    else : 
        raise ValueError(f"'{method}' is not a valid method.")
    
    return np.rad2deg(joint_rad)
