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

    for i, jn in enumerate(model.joints):  # Utilisez enumerate pour obtenir l'index
        joint_name = model.names[i]  # Accédez au nom du joint via model.names
        print(joint_name)
        if joint_name not in jointsToNotLock:  # Si le joint doit être verrouillé
            jointsToLockIDs.append(jn.id)  # Utilisez jn.id pour l'ID du joint
            initialJointConfig[i-1] = 0  # Fixez la configuration initiale à 0

    print(jointsToLockIDs)

    # Option 1: Only build the reduced model in case no display needed:
    model_reduced = pin.buildReducedModel(model, jointsToLockIDs, initialJointConfig)

    return model_reduced


# def export_model_to_urdf(model: pin.Model, urdf_filename: str):
#     """
#     Export a Pinocchio model to a URDF file.
#     """
#     robot = Element('robot', attrib={'name': 'reduced_model'})

#     # Add links
#     for link in model.frames:
#         if link.type == pin.FrameType.BODY:
#             link_element = SubElement(robot, 'link', attrib={'name': link.name})

#     # Add joints
#     for joint in model.joints:
#         if joint.idx_q == -1:  # Skip the universe joint
#             continue
        
#         joint_element = SubElement(robot, 'joint', attrib={
#             'name': joint.name,
#             'type': 'revolute' if joint.nq == 1 else 'fixed'
#         })
#         parent_link = model.frames[joint.parent_id].name if joint.parent_id != 0 else 'base_link'
#         child_link = joint.name
        
#         # Parent and child
#         SubElement(joint_element, 'parent', attrib={'link': parent_link})
#         SubElement(joint_element, 'child', attrib={'link': child_link})

#         # Axis of rotation
#         axis = joint.placement.translation
#         SubElement(joint_element, 'axis', attrib={
#             'xyz': f"{axis[0]} {axis[1]} {axis[2]}"
#         })

#     # Write the URDF file
#     tree = ElementTree(robot)
#     with open(urdf_filename, 'wb') as file:
#         tree.write(file)




# Example usage with your reduced model
urdf_filename = "/home/kloe-bonnet/Documents/ENSEIRB/dfdf/projet_pollen/reachy.urdf"
prefix = "r" # r ou l

# reduced_model = reduce_model_pin_arm(urdf_filename, prefix)
# export_model_to_urdf(reduced_model, "reduced_model.urdf")



def reduce_model_pin_arm_2(urdf_filename: str, prefix: str) -> pin.Model:
    """
    Create a reduced model of the URDF with Pinocchio for a specific arm defined by a prefix.
    Ensures parallel structure is maintained.

    Parameters:
        urdf_filename (str): Path to the URDF file.
        prefix (str): Prefix for the arm joints to preserve (e.g., 'left' or 'right').

    Returns:
        pin.Model: A reduced model containing only the specified arm's joints.
    """
    # Load the full model from the URDF file
    model, _, _ = pin.buildModelsFromUrdf(urdf_filename)
    print("Standard model: dim=" + str(len(model.joints)))

    # Define the joints to preserve based on the prefix
    joints_to_keep = [
        f"{prefix}_shoulder_pitch",
        f"{prefix}_shoulder_roll",
        f"{prefix}_elbow_yaw",
        f"{prefix}_elbow_pitch",
        f"{prefix}_wrist_roll",
        f"{prefix}_wrist_pitch",
        f"{prefix}_wrist_yaw",
    ]

    print(f"Joints to keep: {joints_to_keep}")

    # Initialize lists for joint IDs to lock and their configuration
    joints_to_lock_ids = []
    initial_joint_config = np.ones(len(model.joints) - 1)  # Configurations for all non-fixed joints

    # Loop through all joints in the model
    for i, joint in enumerate(model.joints):
        joint_name = model.names[i]  # Get the joint name
        print(f"Checking joint: {joint_name}")

        if joint_name not in joints_to_keep and joint_name != "universe":
            # If the joint is not in the list to keep, lock it
            joints_to_lock_ids.append(joint.id)
            initial_joint_config[i - 1] = 0  # Set the locked joint's configuration to 0

    print(f"Joints to lock: {joints_to_lock_ids}")

    # Ensure the structure is consistent and that we only lock valid joints
    if len(joints_to_lock_ids) + len(joints_to_keep) + 1 != len(model.joints):
        raise ValueError("Mismatch in the number of joints to lock and keep. Check the URDF structure.")

    # Build the reduced model
    model_reduced = pin.buildReducedModel(model, joints_to_lock_ids, initial_joint_config)

    print("Reduced model created successfully.")
    return model_reduced

reduced_model = reduce_model_pin_arm_2(urdf_filename, prefix)

from pinocchio import buildModelFromUrdf, Model
import xml.etree.ElementTree as ET

def reduce_model(urdf_filename: str, prefix: str) -> Model:
    """
    Extracts and reduces a URDF file to include only the parts of a robot related to a specific prefix.
    
    Args:
        urdf_filename (str): Path to the URDF file.
        prefix (str): The prefix indicating the part of the robot to extract (e.g., 'r_' for the right arm).
    
    Returns:
        pin.Model: A Pinocchio model containing only the specified part of the robot.
    """
    # Parse the URDF file
    tree = ET.parse(urdf_filename)
    root = tree.getroot()
    
    # Define the URDF namespaces
    ns = {"": "http://www.ros.org/wiki/xacro"}
    
    # Filter the links and joints with the given prefix
    links = [link for link in root.findall("link", ns) if link.get("name", "").startswith(prefix)]
    joints = [joint for joint in root.findall("joint", ns) if joint.get("name", "").startswith(prefix)]
    
    # Build a new URDF tree with only the filtered links and joints
    reduced_urdf = ET.Element("robot", {"name": f"{prefix}_reduced"})
    
    for link in links:
        reduced_urdf.append(link)
    for joint in joints:
        reduced_urdf.append(joint)
    
    # Write the reduced URDF to a string
    reduced_urdf_string = ET.tostring(reduced_urdf, encoding="unicode")
    
    # Build the Pinocchio model from the reduced URDF
    model = buildModelFromUrdf(reduced_urdf_string)
    
    return model


reduce_model(urdf_filename, prefix) 