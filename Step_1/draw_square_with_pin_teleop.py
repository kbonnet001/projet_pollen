# /home/reachy/dev/reachy2-sdk/src/examples/draw_square_with_pin_teleop.py

import logging
import time

import numpy as np
import numpy.typing as npt

from reachy2_sdk import ReachySDK
from reachy2_sdk.reachy_sdk import GoToHomeId

from ik_pinocchio import reduce_model_pin_arm, get_joints_from_pin, test

import time

import numpy as np
# from basic_tests import build_pose_matrix
from google.protobuf.wrappers_pb2 import FloatValue
from reachy2_sdk_api.arm_pb2 import ArmCartesianGoal
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4

from reachy2_sdk import ReachySDK

reachy = ReachySDK(host="localhost")

reachy.r_arm._stub.TurnOn(reachy.r_arm._part_id)


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

def draw_square_with_pin(model, data, reachy: ReachySDK, step=0.05) -> None:
    # In A position, the effector is at (0.4, -0,5, -0.2) in the world frame
    # In B position, the effector is at (0.4, -0.5, 0) in the world frame
    # In C position, the effector is at (0.4, -0.3, 0.0) in the world frame
    # In D position, the effector is at (0.4, -0.3, -0.2) in the world frame
    # see https://docs.pollen-robotics.com/sdk/first-moves/kinematics/ for Reachy's coordinate system

    # pose A (0.4, -0.5, -0.2)
    # Going from A to B : (0.4, -0.5, 0)
    for z in np.arange(-0.2+step, 0.0+step, step):
        print("z=", z)
        goal = build_pose_matrix(0.4, -0.5, z)

        target = ArmCartesianGoal(
        id={"id": 1, "name": "r_arm"},
        goal_pose=Matrix4x4(data=goal.flatten().tolist()),
        duration=FloatValue(value=1.0),
        )

        reachy.r_arm._stub.SendArmCartesianGoal(target)
        time.sleep(0.5)

    time.sleep(0.5)
    current_pos = reachy.r_arm.forward_kinematics()
    print(f"Pose B (with pin) :\ncurrent pose = {current_pos}\n----")

    # Going from B to C : (0.4, -0.3, 0)
    for y in np.arange(-0.5+step, -0.3+step, step):
        goal = build_pose_matrix(0.4, y, 0)

        target = ArmCartesianGoal(
        id={"id": 1, "name": "r_arm"},
        goal_pose=Matrix4x4(data=goal.flatten().tolist()),
        duration=FloatValue(value=1.0),
        )

        reachy.r_arm._stub.SendArmCartesianGoal(target)
        time.sleep(0.5)

    time.sleep(0.5)
    current_pos = reachy.r_arm.forward_kinematics()
    print(f"Pose C (with pin) :\ncurrent pose = {current_pos}\n----")

    # Going from C to D : (0.4, -0.3, -0.2)
    for z in np.arange(0-step, -0.2-step, -step):
        goal = build_pose_matrix(0.4, -0.3, z)

        target = ArmCartesianGoal(
        id={"id": 1, "name": "r_arm"},
        goal_pose=Matrix4x4(data=goal.flatten().tolist()),
        duration=FloatValue(value=1.0),
        )

        reachy.r_arm._stub.SendArmCartesianGoal(target)
        time.sleep(0.5)

    time.sleep(0.5)
    current_pos = reachy.r_arm.forward_kinematics()
    print(f"Pose D (with pin) :\ncurrent pose = {current_pos}\n----")

    # Going from D to A : (0.4, -0.5, -0.2)
    for y in np.arange(-0.3-step, -0.5-step, -step):
        goal = build_pose_matrix(0.4, y, -0.2)

        target = ArmCartesianGoal(
        id={"id": 1, "name": "r_arm"},
        goal_pose=Matrix4x4(data=goal.flatten().tolist()),
        duration=FloatValue(value=1.0),
        )

        reachy.r_arm._stub.SendArmCartesianGoal(target)
        time.sleep(0.5)

    time.sleep(0.5)
    current_pos = reachy.r_arm.forward_kinematics()
    print(f"Pose A (with pin) :\ncurrent pose = {current_pos}\n----")

    time.sleep(3)


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

    # Load reduce URDF model of arm
    model, data = reduce_model_pin_arm("r")

    print("Set to Zero pose ...")
    move_ids = reachy.set_pose("default")
    wait_for_pose_to_finish(move_ids)

    # print("Set to Elbow 90 pose ...")
    # move_ids = reachy.set_pose("elbow_90")
    # wait_for_pose_to_finish(move_ids)

    print("Move to point A")
    move_to_point_A(reachy)
    time.sleep(3)

    print("Draw a square with the right arm (with pinocchio) ...")
    draw_square_with_pin(model, data, reachy)

    print("Set to Zero pose ...")
    move_ids = reachy.set_pose("default")
    wait_for_pose_to_finish(move_ids)

    print("Turning off Reachy")
    reachy.turn_off()

    time.sleep(0.2)

    exit("Exiting example")
