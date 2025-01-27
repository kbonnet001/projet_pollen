import pinocchio as pin
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
import numpy as np

# Pourrait être plus propre, je laisse dans l'état le temps de s'assurer que tout fonctionne correctement
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
    print("standard model: dim=" + str(len(model.joints)))

    # Create a list of joints to NOT lock
    jointsToNotLock = [f"{prefix}_shoulder_pitch",f"{prefix}_shoulder_roll", f"{prefix}_elbow_arm_link", 
                    f"{prefix}_elbow_yaw",f"{prefix}_elbow_pitch", f"{prefix}_wrist_roll", f"{prefix}_wrist_pitch", f"{prefix}_wrist_yaw"] 
    print(jointsToNotLock)
    #Get the Id of all existing joints
    jointsToLockIDs = []
    initialJointConfig = np.ones(len(model.joints)-1)
    i=-1

    for i, jn in enumerate(model.joints):  # enumerate pour obtenir l'index
        joint_name = model.names[i]  # nom du joint via model.names
        print(joint_name)
        if joint_name not in jointsToNotLock:  # Si le joint doit être verrouillé
            jointsToLockIDs.append(jn.id)  # jn.id pour l'ID du joint
            initialJointConfig[i-1] = 0  # Fixez la configuration initiale à 0

    print(jointsToLockIDs)

    # Option 1: Only build the reduced model in case no display needed:
    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)

    return model_reduced



# Example usage with your reduced model
# urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf"
# prefix = "r" # r ou l


