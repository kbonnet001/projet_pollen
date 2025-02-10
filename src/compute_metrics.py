import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

import sys
sys.path.append('/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/')
from CSVLogger import CSVLogger


def compute_metrics(goal_pose, current_joints, prefix, method, q_compute, poses_compute):

    """
    Calcule les écarts de position, rotation et angles des articulations pour différentes méthodes (Pink, Pinocchio, etc.).
    """
    csv_filename = f"/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/{method}/metrics_{method}_{prefix}.csv"

    ecart_q = np.abs(np.array(current_joints) - np.array(q_compute))

    translation = poses_compute[:3,3]
    q_current = R.from_matrix(poses_compute[:3,:3]).as_quat()

    ecart_distance, ecart_angle = compute_ecart_rot_pos_prefix(goal_pose, q_current, translation)

    # hearders

    headers, log_data = log_logger(prefix, translation, current_joints, ecart_q, ecart_distance, ecart_angle)

    logger = CSVLogger(csv_filename, headers)
    logger.log(log_data)



def log_logger(prefix, translations, current_joints, ecart_q, ecart_distance, ecart_angle) : 

    headers = []
    log_data = []

    headers+= [f"q_{prefix}_{i}" for i in range(len(current_joints))] + \
            [f"translation_{prefix}_{i}" for i in range(len(translations))] + \
            [f"ecart_q_{prefix}_{i}" for i in range(len(ecart_q))] + \
            [f"Ecart_pos_{prefix}", f"Ecart_rot_{prefix}"]

    log_data+=[*current_joints, *translations, *ecart_q, ecart_distance, ecart_angle]

    
    return headers, log_data
              

def compute_ecart_rot_pos_prefix(goal_poses, q_currents, translations) : 

    ecart_distance = np.linalg.norm(goal_poses[:3, 3] - translations)

    # Calcul de l'écart de rotation
    rot_goal = R.from_matrix(goal_poses[:3, :3]).as_quat()
    rot_delta = R.from_quat(rot_goal) * R.from_quat(q_currents).inv()
    ecart_angle = np.degrees(2 * np.arccos(rot_delta.as_quat()[-1]))
    if ecart_angle > 180:
        ecart_angle = abs(ecart_angle - 360)

    return ecart_distance, ecart_angle
    