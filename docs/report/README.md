

# 📖 Report

## ⁉️ Specifications

[Pollen](https://www.pollen-robotics.com/), a start-up located in Bordeaux, develops [Reachy](https://github.com/pollen-robotics), an anthropomorphic robot primarily used for immersive teleoperation applications. In these applications, enhancing the user's sense of control is essential but must not come at the expense of safety. It is therefore crucial to ensure the optimal execution of commands while respecting both the robot's intrinsic and extrinsic constraints. Furthermore, continuously providing feedback on the robot's capabilities can contribute to this sense of control by reducing the gap between the commanded action and the one actually performed.  

The objective of the project is to explore alternatives to the inverse kinematics approach implemented by Pollen within the Reachy robot architecture, particularly in the context of teleoperation.

### Input
We rely on Reachy2’s existing hardware and software, including its analytical inverse kinematics solver. While an implementation for handling joint limits exists, it lacks robustness and needs improvement.

### Output
Our goal is to enhance motion control strategies to ensure smooth and reliable teleoperation, even near singularities and joint limits. This will involve refining existing solutions and developing new approaches to improve accuracy and responsiveness.

## 🔎 Implemented approch

To address this challenge, we conducted a [state-of-the-art review](docs/bibliography/etat_de_l_art.pdf) to explore alternative inverse kinematics approaches within Reachy2’s architecture. Based on this, we defined the following action plan:

* Implement inverse kinematics resolution using the Jacobian and the [Pinocchio library](https://github.com/stack-of-tasks/pinocchio).
* Solve inverse kinematics using a Quadratic Programming (QP) approach with the [Pink library](https://github.com/stephane-caron/pink).
* Introduce constraints and experiment with different combinations.
* Perform tests to compare methods, analyze results, and provide critical insights.

## 📈 Analysis of results

**Qualify** and **quantify** your achievements. Make measurements from your work, e.g.:

* **User tests**: Setup a methodology to test the efficiency of your project against users. It may use pre-experiment and post-experiment questionnaires. The most users the better to draw meaningful conclusions from your experiments. Radar diagrams are good to summarize such results.
* **Table of data**: Provide (short) extracts of data and relevant statistics (distribution, mean, standard deviation, p-values...)
* **Plots**: Most data are more demonstrative when represented as plots. 

Draw conclusions, **interpret** the results and make recommandations to your client for your future of the work.
It is totally fine to have results that are not as good as initially expected. Be honest and analyse why you did not manage to reach the objectives.
