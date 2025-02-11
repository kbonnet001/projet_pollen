import numpy as np
import pinocchio
from numpy.linalg import norm, solve

model = pinocchio.buildSampleModelManipulator()
# Nb joints = 7 (nq=6,nv=6)
#   Joint 0 universe: parent=0
#   Joint 1 shoulder1_joint: parent=0
#   Joint 2 shoulder2_joint: parent=1
#   Joint 3 shoulder3_joint: parent=2
#   Joint 4 elbow_joint: parent=3
#   Joint 5 wrist1_joint: parent=4
#   Joint 6 wrist2_joint: parent=5

data = model.createData()

JOINT_ID = 6
oMdes = pinocchio.SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))

q = pinocchio.neutral(model)
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
while True:
    pinocchio.forwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)

    err = pinocchio.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print("%d: error = %s" % (i, err.T))
    i += 1

if success:
    print("Convergence achieved!")
else:
    print(
        "\n"
        "Warning: the iterative algorithm has not reached convergence "
        "to the desired precision"
    )

print(f"\nresult: {q.flatten().tolist()}")
print(f"\nfinal error: {err.T}")