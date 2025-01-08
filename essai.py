import pinocchio
print(pinocchio.__version__)

import numpy as np
from numpy.linalg import norm, solve
from os.path import abspath

from scipy.spatial.transform import Rotation as R


# définitions des variables (on essaye de prendre des trucs ok)

# Compréhension : 
# --------------

# goal_pose: npt.NDArray[np.float64]

# On donne une matrice homogène et on récupère la position 
# Avec l'orientation on calcule angle de euler --> vecteur 3
# def get_euler_from_homogeneous_matrix(
#     homogeneous_matrix: npt.NDArray[np.float64], degrees: bool = False
# ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     position = homogeneous_matrix[:3, 3]
#     rotation_matrix = homogeneous_matrix[:3, :3]
#     euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=degrees)
#     return position, euler_angles

# goal_position, goal_orientation = get_euler_from_homogeneous_matrix(M)
# goal_pose = np.array([goal_position, goal_orientation])
# Goal Pose sous la forme : [[x x x],[x x x]]

# Exemple de goal_pose (position et orientation)
goal_position = np.array([0.4, -0.5, 0])  # Translation, pris de l'exemple dessiner carré
goal_orientation = np.array([0.0, np.pi/4, np.pi/2])  # Angles d'Euler (en radians)


# Représentation goal_pose
goal_pose = np.array([goal_position, goal_orientation], dtype=float)
# -------------------------------------

# current_joints: list[float]
current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# -----------------------------------------------------------------------


# Définir le chemin vers le fichier URDF
urdf_filename = abspath("/home/kloe-bonnet/Documents/ENSEIRB/essai/projet_pollen/reachy.urdf")

# Charger le modèle URDF
model = pinocchio.buildModelFromUrdf(urdf_filename)
data = model.createData()

# The end effector corresponds to the 7th joint
JOINT_ID = 7

# Conversion des angles d'Euler en matrice de rotation
rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()

# Création de l'objet SE3
# and its desired pose is given as
# goal_pose_temp=np.array([0.4, -0.5, 0.0])
oMdes = pinocchio.SE3(rotation_matrix, goal_pose[0]) # matrice de rotation et position

# q = pinocchio.neutral(model)
q = np.array(current_joints) # plutot mettre là où on est actuellement
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12


i = 0
while True:
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
    iMd = data.oMi[JOINT_ID].actInv(oMdes)  # Calcul de l'erreur
    err = pinocchio.log(iMd).vector  # Erreur dans l'espace des twists

    if norm(err) < eps: # cas où ça converge :)
        success = True
        break
    if i >= IT_MAX: # on atteint la limite de boucle :(
        success = False
        break

    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # Jacobienne
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)  # Jacobienne modifiée pour réduire l'erreur
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  # Résolution par moindres carrés
    q = pinocchio.integrate(model, q, v * DT)  # Intégrer le déplacement des jointures

    if not i % 10:
        print(f"{i}: error = {err.T}") # pour avoir un suivi de ce qu'il se passe
    i += 1

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

# # Nouvelle fonction : 
# # ------------------
def symbolic_inverse_kinematics_continuous(
    self,
    name: str,
    goal_pose: npt.NDArray[np.float64],
    interval_limit: npt.NDArray[np.float64],
    current_joints: list[float],
    current_pose: npt.NDArray[np.float64],
    preferred_theta: float,
    d_theta_max: float,
) -> Tuple[list[float], bool, str]:
    """Compute the inverse kinematics of the goal pose M with continuous control.
    Args:
        name: r_arm or l_arm
        goal_pose: position and euler angles of the goal pose
        interval_limit
        current_joints
        current_pose
        preferred_theta
        d_theta_max: maximum angle difference between two consecutive theta
    """

    # à laisser, pour checker si met trop de temps 
    t = time.time()
    state = ""
    if abs(t - self.last_call_t[name]) > self.call_timeout:
        self.previous_sol[name] = []
        if DEBUG:
            print(f"{name} Timeout reached. Resetting previous_sol {t},  {self.last_call_t[name]}")
    self.last_call_t[name] = t


    # Définir le chemin vers le fichier URDF
    #######################################################################################################
    # changer ici pour que ça prenne le bon urdf en fonction du bras (s'aider de "name")
    # Définir le chemin vers le fichier URDF
    urdf_filename = abspath("/home/kloe-bonnet/Documents/ENSEIRB/essai/projet_pollen/reachy.urdf")

    # Charger le modèle URDF
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    data = model.createData()

    # The end effector corresponds to the 7th joint
    JOINT_ID = 7

    # Conversion des angles d'Euler en matrice de rotation
    rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()

    # Création de l'objet SE3
    # and its desired pose is given as
    # goal_pose_temp=np.array([0.4, -0.5, 0.0])
    oMdes = pinocchio.SE3(rotation_matrix, goal_pose[0]) # matrice de rotation et position

    # q = pinocchio.neutral(model)
    q = np.array(current_joints) # plutot mettre là où on est actuellement
    eps = 1e-4
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-12


    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        pinocchio.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
        iMd = data.oMi[JOINT_ID].actInv(oMdes)  # Calcul de l'erreur
        err = pinocchio.log(iMd).vector  # Erreur dans l'espace des twists

        if norm(err) < eps: # cas où ça converge :)
            success = True
            break
        if i >= IT_MAX: # on atteint la limite de boucle :(
            success = False
            break

        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # Jacobienne
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)  # Jacobienne modifiée pour réduire l'erreur
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  # Résolution par moindres carrés
        q = pinocchio.integrate(model, q, v * DT)  # Intégrer le déplacement des jointures

        if not i % 10:
            print(f"{i}: error = {err.T}") # pour avoir un suivi de ce qu'il se passe
        i += 1

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



    if self.previous_sol[name] == []:
        # if the arm moved since last call, we need to update the previous_sol
        # self.previous_sol[name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # TODO : Get a current position that take the multiturn into consideration
        # Otherwise, when there is no call for more than call_timeout, the joints will be cast between -pi and pi
        # -> If you pause a rosbag during a multiturn and restart it, the previous_sol will be wrong by 2pi
        self.previous_sol[name] = current_joints
        current_goal_position, current_goal_orientation = get_euler_from_homogeneous_matrix(current_pose)
        current_pose_tuple = np.array([current_goal_position, current_goal_orientation])
        is_reachable, interval, theta_to_joints_func = self.symbolic_ik_solver[name].is_reachable_no_limits(
            current_pose_tuple,
        )
        best_prev_theta, state_previous_theta = get_best_theta_to_current_joints(
            theta_to_joints_func, 20, current_joints, name, preferred_theta
        )
        self.previous_theta[name] = best_prev_theta

        if DEBUG:
            print(f"{name}, previous_theta: {self.previous_theta[name]}")

    (
        is_reachable,
        interval,
        theta_to_joints_func,
        state_reachable,
    ) = self.symbolic_ik_solver[
        name
    ].is_reachable(goal_pose)
    # self.print_log(f"{name} state_reachable: {state_reachable}")
    if is_reachable:
        is_reachable, theta, state_theta = get_best_continuous_theta(
            self.previous_theta[name],
            interval,
            theta_to_joints_func,
            d_theta_max,
            preferred_theta,
            self.symbolic_ik_solver[name].arm,
        )
        # is_reachable, theta, state_theta = get_best_continuous_theta2(
        #     self.previous_theta[name],
        #     interval,
        #     theta_to_joints_func,
        #     10,
        #     d_theta_max,
        #     self.preferred_theta[name],
        #     self.symbolic_ik_solver[name].arm,
        # )
        if not is_reachable:
            state = "limited by shoulder"
        theta, state_interval = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
        self.previous_theta[name] = theta
        ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])

    else:
        if DEBUG:
            print(f"{name} Pose not reachable before even reaching theta selection. State: {state_reachable}")
        is_reachable_no_limits, interval, theta_to_joints_func = self.symbolic_ik_solver[name].is_reachable_no_limits(
            goal_pose
        )
        if is_reachable_no_limits:
            is_reachable_no_limits, theta = tend_to_preferred_theta(
                self.previous_theta[name],
                interval,
                theta_to_joints_func,
                d_theta_max,
                goal_theta=preferred_theta,
            )
            theta, state = limit_theta_to_interval(theta, self.previous_theta[name], interval_limit)
            self.previous_theta[name] = theta
            ik_joints, elbow_position = theta_to_joints_func(theta, previous_joints=self.previous_sol[name])
        else:
            print(f"{name} Pose not reachable, this has to be fixed by projecting far poses to reachable sphere")
            raise RuntimeError("Pose not reachable in symbolic IK. We crash on purpose while we are on the debug sessions.")
        state = state_reachable

    if DEBUG:
        print(f"State: {state}")

    return ik_joints, is_reachable, state