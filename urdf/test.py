import pinocchio as pin 
from os.path import abspath

urdf_filename = "urdf/reachy.urdf"
model, _, _ = pin.buildModelsFromUrdf(abspath(urdf_filename))