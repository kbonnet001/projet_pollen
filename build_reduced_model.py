import numpy as np
import pinocchio as pin

import numpy as np
from numpy.linalg import norm, solve


from scipy.spatial.transform import Rotation as R
from urdf_tools import reduce_model_pin_arm

import matplotlib.pyplot as plt

# ----------------------------------------------------------

def symbolic_inverse_kinematics_continuous_for_test(urdf_filename, prefix, current_joints, goal_pose, matrix_rot, debug=True, plot=False):

    if debug : 
        print("--------------\nsymbolic_inverse_kinematics_continuous_for_test\n--------------")

    # model pin de l'urdf réduit (1 bras)
    model = reduce_model_pin_arm(urdf_filename, prefix)
    data = model.createData()

    # The end effector corresponds to the 7th joint
    JOINT_ID = 7
    q = np.array(current_joints) 

    # Conversion des angles d'Euler en matrice de rotation
    # Pour code Pollen, a utiliser avec goal_pose_orientation
    rotation_matrix = R.from_euler("xyz", goal_pose[1]).as_matrix()

    oMdes = pin.SE3(matrix_rot, goal_pose[0]) # matrice de rotation et position

    if debug : 
        # Afficher la transformation pour chaque joint (de oMi) au départ ! i = 0
        pin.updateFramePlacements(model, data)  # Mettre à jour la cinématique avant, peut être pas utile
        pin.forwardKinematics(model, data, q)
        for i, oMi in enumerate(data.oMi):
            print(f"Joint {i}: Transformation\n{oMi}")

    # Paramètres
    eps = 1e-4
    IT_MAX = 50000 # original 1000
    DT = 1e-3 # original e-1
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

        if norm(err) < eps: # cas où ça converge :)
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
        if not i % 1000 and (i==10000):
            DT=DT*1e-1

            if debug : 
                print("DT = ", DT)

        i += 1

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

    return q.tolist()

