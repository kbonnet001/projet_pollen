# Robotic Project

**Description:** Adaptation of Teleoperation Commands to Account for Intrinsic and Extrinsic Constraints: Application to the Reachy Robot

[📖 User documentation](docs/user) • [👨‍💻 Developer documentation](docs/developer) • [📈 Project report](docs/report) • [📚 Bibliography](docs/bibliography)
  
## 📄 This project in short
This project explores various alternatives to the inverse kinematics solution implemented by [Pollen Robotics](https://www.pollen-robotics.com/), [available on this repository](https://github.com/pollen-robotics/reachy2_symbolic_ik/tree/fd970e1d22fc0340af6339db56a5e15a9649d633), aiming to manage the robot's constraints and limitations. 
The Pinocchio and Pink libraries are used to carry out this work. Among the proposed solutions, the QP-based approach with barrier integration shows promising potential.

## 🚀 Quickstart


* **Install instructions**:
    *  Requirement :
          *  pinocchio : https://stack-of-tasks.github.io/pinocchio/download.html
          *  pink : https://github.com/stephane-caron/pink
          *  qpsolvers : https://qpsolvers.github.io/qpsolvers/installation.html
    
⚠️ : Those installation could change the current version of numpy
* **Launch instructions**:

To launch the programm, the git must first be installed in **Reachy stack**, to a path of your choice. Then, our recommandation is to install [src code](/src/) in **`/home/reachy/dev/reachy2_symbolic_ik/src`** where the original `control_ik` function is located. If the path is modified, change it accordingly in the variable `projet_pollen_folder_path` in the file **`src/ik_methods.py`**.
📂 **`assets/urdf/reachy.urdf`** must be placed in 📂 **`/home/reachy/reachy_ws/src/reachy2_core/reachy_description/urdf/`** for proper integration with the rest of the code.

Reachy must first be launched and when it is ready to start, it can be launched using the command line **`python3  {path}/projet_pollen/src/test_movement_test.py`** with `{path}` the previous path where you installed the projet_pollen folder. 

## 🔍 About this project

|       |        |
|:----------------------------:|:-----------------------------------------------------------------------:|
| 💼 **Client**                |  Rémi Fabre and Vincent Padois                                           |
| 🔒 **Confidentiality**       | **Public**                                     |
| ⚖️ **License**               |  [Choose a license](https://choosealicense.com/) *(1)*                  |
| 👨‍👨‍👦 **Authors**               |  Kloé Bonnet and Guillaume Lauga    |


