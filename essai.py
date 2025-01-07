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
urdf_filename = abspath("/home/kloe-bonnet/Documents/ENSEIRB/essai/reachy.urdf")

# Charger le modèle URDF
model = pinocchio.buildModelFromUrdf(urdf_filename)
data = model.createData()

# The end effector corresponds to the 7th joint
JOINT_ID = 7

# Conversion des angles d'Euler en matrice de rotation
rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()

# Vérifier et convertir en types compatibles Eigen
goal_translation = np.asarray(goal_pose[0], dtype=np.float64).reshape(3)  # Assurer un vecteur 3D
rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)  # Assurer une matrice 3x3

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


# MODIFIE: Retourner les résultats comme demandé
# return ik_joints, is_reachable, state



# def symbolic_inverse_kinematics_continuous_fsdvdv( #modifications
#        self,
#        name: str,
#        goal_pose: npt.NDArray[np.float64],
#        interval_limit: npt.NDArray[np.float64],
#        current_joints: list[float],
#        current_pose: npt.NDArray[np.float64],
#        preferred_theta: float,
#        d_theta_max: float,
#    ) -> Tuple[list[float], bool, str]:
#    """Compute the inverse kinematics of the goal pose M with continuous control.
#    Args:
#        name: r_arm or l_arm
#        goal_pose: position and euler angles of the goal pose
#        interval_limit
#        current_joints
#        current_pose
#        preferred_theta
#        d_theta_max: maximum angle difference between two consecutive theta
#    """


#    # On charge le model urdf du robot
#    if not hasattr(self, "pinocchio_model"):  # On fait seulement si pas déjà fait
#        self.pinocchio_model = pinocchio.buildModelFromUrdf(self.urdf_path) # pas sure...
#        self.pinocchio_data = self.pinocchio_model.createData()


#    model = self.pinocchio_model
#    data = self.pinocchio_data


#    # JOINT_ID = self.get_joint_id(name) 
#    JOINT_ID = 7 # MODIFIE: Utilisation de la fonction pour identifier l'articulation
#    oMdes = pinocchio.SE3(
#        pinocchio.utils.rpyToMatrix(*goal_pose[3:]), goal_pose[:3]
#    )  # MODIFIE: Créer la pose cible en SE3 avec orientation et position
#    # on donne position, orientation
#    # pas sure de ça


#    q = np.array(current_joints)  # MODIFIE: Initialiser `q` avec les positions actuelles
#    # probablement marche pas car on dit pas [0] ou [1]
#    # car current_joints [[bras 1], [bras 2]]
#    eps = 1e-4
#    IT_MAX = 1000
#    DT = 1e-1
#    damp = 1e-12


#    i = 0
#    while True:
#        pinocchio.forwardKinematics(model, data, q)
#        pinocchio.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
#        iMd = data.oMi[JOINT_ID].actInv(oMdes)  # Calcul de l'erreur
#        err = pinocchio.log(iMd).vector  # Erreur dans l'espace des twists

#        if norm(err) < eps: # cas où ça converge :)
#            success = True
#            break
#        if i >= IT_MAX: # on atteint la limite de boucle :(
#            success = False
#            break

#        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # Jacobienne
#        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)  # Jacobienne modifiée pour réduire l'erreur
#        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  # Résolution par moindres carrés
#        q = pinocchio.integrate(model, q, v * DT)  # Intégrer le déplacement des jointures

#        if not i % 10:
#            print(f"{i}: error = {err.T}") # pour avoir un suivi de ce qu'il se passe
#        i += 1

#    if success:
#        print("Convergence achieved!")
#        state = "converged"
#        ik_joints = q.tolist()  # Résultats des positions des jointures
#        is_reachable = True
#    else:
#        print(
#            "\nWarning: the iterative algorithm has not reached convergence "
#            "to the desired precision"
#        )
#        state = "not converged"
#        ik_joints = current_joints  # Retourner les positions actuelles si pas convergé
#        is_reachable = False


#    # MODIFIE: Retourner les résultats comme demandé
#    return ik_joints, is_reachable, state
