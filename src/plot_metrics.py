import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
sys.path.append('/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/')

def plot_metrics_for_pink_sphere(path, df, method, d_min):
    """
    Plots the evolution of the distance between two pink spheres.
    
    Parameters:
        path (str): Directory containing the CSV file.
        filename (str): Name of the CSV file containing the metrics.
        method (str): Method identifier for saving the plot.
        d_min (float): Minimum distance threshold to be plotted as a horizontal line.
    """

    # Convert columns to numpy arrays for plotting
    iterations = df["Iteration"].to_numpy()
    distances = df["Distance"].to_numpy()

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, distances, marker="o", label="Distance")
    plt.axhline(y=d_min, color="r", linestyle="--", label=f"d_min = {d_min}")

    # Add labels, title, and legend
    plt.xlabel("Iteration")
    plt.ylabel("Distance (m)")
    plt.title("Evolution of the Distance between both spheres")
    plt.legend()

    # Save and display the plot
    plt.savefig(f"{path}/{method}_plot_distance_sphere.png", dpi=300)
    plt.show()

def plot_q(path, df, method):
    """
    Plots the deviation of the q values for the left and right arms using subplots,
    with the primary Y-axis in radians (fractions of π) and a secondary Y-axis in degrees.
    
    Parameters:
        path (str): Directory containing the CSV file.
        filename (str): Name of the CSV file containing the metrics.
        method (str): Method identifier for saving the plot.
    """

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm deviations
    ecart_l_cols = [col for col in df.columns if col.startswith("q_l")]
    ecart_r_cols = [col for col in df.columns if col.startswith("q_r")]

    # Definition of ticks in rad
    radian_ticks = np.array([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi])
    radian_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "0", 
                     r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\pi$"]

    def add_secondary_y_axis(ax):
        ax2 = ax.twinx()
        ax2.set_ylabel("Degrees", color='black')
        ax2.set_ylim(np.degrees(ax.get_ylim()[0]), np.degrees(ax.get_ylim()[1]))
        return ax2

    # Plot left arm deviations (radians primary, degrees secondary)
    for col in ecart_l_cols:
        axes[0].plot(iterations, df[col].to_numpy(), marker="o", label=col)
    axes[0].set_ylabel("Left Arm q (rad)")
    axes[0].set_yticks(radian_ticks)
    axes[0].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[0])
    axes[0].legend()

    # Plot right arm deviations (radians primary, degrees secondary)
    for col in ecart_r_cols:
        axes[1].plot(iterations, df[col].to_numpy(), marker="o", label=col)
    axes[1].set_ylabel("Right Arm q (rad)")
    axes[1].set_yticks(radian_ticks)
    axes[1].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[1])
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    plt.title("Variation of joint angle q for each pose")
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_q.png", dpi=300)
    plt.show()

def plot_velocity(path, df, method):
    """
    Plots the deviation of the q values for the left and right arms using subplots,
    with the primary Y-axis in radians (fractions of π) and a secondary Y-axis in degrees.
    
    Parameters:
        path (str): Directory containing the CSV file.
        filename (str): Name of the CSV file containing the metrics.
        method (str): Method identifier for saving the plot.
    """

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm deviations
    ecart_l_cols = [col for col in df.columns if col.startswith("velocity_l")]
    ecart_r_cols = [col for col in df.columns if col.startswith("velocity_r")]

    # Definition of ticks in rad
    radian_ticks = np.array([-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi])
    radian_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "0", 
                     r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\pi$"]

    def add_secondary_y_axis(ax):
        ax2 = ax.twinx()
        ax2.set_ylabel("Degrees/s", color='black')
        ax2.set_ylim(np.degrees(ax.get_ylim()[0]), np.degrees(ax.get_ylim()[1]))
        return ax2

    # Plot left arm deviations (radians primary, degrees secondary)
    for col in ecart_l_cols:
        axes[0].plot(iterations, df[col].to_numpy(), marker="o", label=col)
    axes[0].set_ylabel("Left Arm q (rad/s)")
    axes[0].set_yticks(radian_ticks)
    axes[0].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[0])
    axes[0].legend()

    # Plot right arm deviations (radians primary, degrees secondary)
    for col in ecart_r_cols:
        axes[1].plot(iterations, df[col].to_numpy(), marker="o", label=col)
    axes[1].set_ylabel("Right Arm q (rad/s)")
    axes[1].set_yticks(radian_ticks)
    axes[1].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[1])
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    plt.title("Variation of velocity for each pose")
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_velocity.png", dpi=300)
    plt.show()


def plot_velocity_std(path, df, method):
    """
    Plots the rolling standard deviation of the velocity values for the left and right arms using subplots.
    The primary Y-axis is in radians per second, with a secondary Y-axis in degrees per second.
    
    Parameters:
        path (str): Directory to save the plot.
        df (pd.DataFrame): Dataframe containing the velocity data.
        method (str): Method identifier for saving the plot.
    """

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm velocity
    velocity_l_cols = [col for col in df.columns if col.startswith("velocity_l")]
    velocity_r_cols = [col for col in df.columns if col.startswith("velocity_r")]

    # Compute rolling standard deviation with a window of 10
    df_std = df.copy()
    for col in velocity_l_cols + velocity_r_cols:
        df_std[col] = df[col].rolling(window=10, min_periods=1).std()

    # Definition of ticks in rad/s
    radian_ticks = np.array([0, np.pi/12, np.pi/6, np.pi/4])
    radian_labels = ["0", r"$\frac{\pi}{12}$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{4}$"]

    def add_secondary_y_axis(ax):
        ax2 = ax.twinx()
        ax2.set_ylabel("Degrees/s", color='black')
        ax2.set_ylim(np.degrees(ax.get_ylim()[0]), np.degrees(ax.get_ylim()[1]))
        return ax2

    # Plot left arm standard deviation
    for col in velocity_l_cols:
        axes[0].plot(iterations, df_std[col].to_numpy(), marker="o", label=col)
    axes[0].set_ylabel("Left Arm std q (rad/s)")
    axes[0].set_yticks(radian_ticks)
    axes[0].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[0])
    axes[0].legend()

    # Plot right arm standard deviation
    for col in velocity_r_cols:
        axes[1].plot(iterations, df_std[col].to_numpy(), marker="o", label=col)
    axes[1].set_ylabel("Right Arm std q (rad/s)")
    axes[1].set_yticks(radian_ticks)
    axes[1].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[1])
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    plt.suptitle("Rolling Standard Deviation of Velocity (Window = 10)")
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_velocity_std.png", dpi=300)
    plt.show()

def plot_ecart_q(path, df, method, tolerance):
    """
    Plots the deviation (écart) of the q values for the left and right arms using subplots.
    
    Parameters:
        filename (str): Path to the CSV file containing the metrics.
        tolerance (float): The tolerance threshold to be plotted as a horizontal line.
    """

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm deviations
    ecart_l_cols = [col for col in df.columns if col.startswith("ecart_q_l")]
    ecart_r_cols = [col for col in df.columns if col.startswith("ecart_q_r")]

    # Definition of ticks in rad
    radian_ticks = np.array([0, np.pi/12, np.pi/6, np.pi/4, np.pi/2])
    radian_labels = ["0", r"$\frac{\pi}{12}$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]

    def add_secondary_y_axis(ax):
        ax2 = ax.twinx()
        ax2.set_ylabel("Degrees", color='black')
        ax2.set_ylim(np.degrees(ax.get_ylim()[0]), np.degrees(ax.get_ylim()[1]))
        return ax2

    # Plot left arm deviations
    for col in ecart_l_cols:
        axes[0].plot(iterations, df[col].to_numpy(), marker="o", label=col)
    axes[0].axhline(y=tolerance, color="r", linestyle="--", label=f"tolerance = {tolerance}")
    axes[0].set_ylabel("Left Arm q (rad)")
    axes[0].set_yticks(radian_ticks)
    axes[0].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[0])
    axes[0].legend()

    # Plot right arm deviations
    for col in ecart_r_cols:
        axes[1].plot(iterations, df[col].to_numpy(), marker="o", label=col)
    axes[1].axhline(y=tolerance, color="r", linestyle="--", label=f"tolerance = {tolerance}")
    axes[1].set_ylabel("Left Arm q (rad)")
    axes[1].set_yticks(radian_ticks)
    axes[1].set_yticklabels(radian_labels)
    add_secondary_y_axis(axes[1])
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    plt.title("Variation of ecart q for each pose")
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_ecarts_q.png", dpi=300)
    plt.show()

def plot_ecart_pos_rot(path, df, method):
    """
    Plots the positional and rotational error side by side in a single figure.
    
    Parameters:
        path (str): Path to save the plot.
        df (DataFrame): Data containing the errors.
        method (str): Method name for saving the plot.
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    iterations = df["Iteration"].to_numpy()
    
    # Positional error plot (Left)
    ecart_l_pos = df["Ecart_pos_l"].to_numpy()
    ecart_r_pos = df["Ecart_pos_r"].to_numpy()
    
    axes[0].plot(iterations, ecart_l_pos, marker="o", label="Left Pos Error")
    axes[0].plot(iterations, ecart_r_pos, marker="o", label="Right Pos Error")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Position Error (m)")
    axes[0].set_title("Positional Error")
    axes[0].legend()
    
    # Rotational error plot (Right)
    ecart_l_rot = np.radians(df["Ecart_rot_l"].to_numpy())
    ecart_r_rot = np.radians(df["Ecart_rot_r"].to_numpy())
    
    ax1 = axes[1]
    ax2 = ax1.twinx()
    
    ax1.plot(iterations, ecart_l_rot, marker="o", label="Left Rot Error (rad)")
    ax1.plot(iterations, ecart_r_rot, marker="o", label="Right Rot Error (rad)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Rotational Error (radians)", color='black')
    ax1.set_title("Rotational Error")
    
    # Set Y-ticks in fractions of π
    radian_ticks = np.array([0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2])  # Degrees
    radian_labels = ["0", r"$\frac{\pi}{12}$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$"]
    ax1.set_yticks(np.radians(radian_ticks))
    ax1.set_yticklabels(radian_labels)
    
    # Secondary Y-axis for degrees
    ax2.set_ylabel("Rotational Error (deg)", color='black')
    ax2.set_ylim(ax1.get_ylim())  # Ensure same range
    ax2.set_yticks(np.rad2deg(radian_ticks))
    ax2.set_yticklabels(np.rad2deg(radian_ticks))
    
    ax1.legend(loc="upper left")
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_ecart_pos_rot.png", dpi=300)
    plt.show()
    
def plot_translations(path, df, method, plot_goal=True, plot_pollen=True, df_pollen=None):
    """
    Plots the evolution of x, y, z translations over iterations.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    iterations = df["Iteration"].to_numpy()
    label_pos = ["x", "y", "z"]
    colors = ["r", "g", "b"]

    for i in range(3):
        axes[i].plot(iterations, df[f'translation_l_{i}'].to_numpy(), marker="o", markersize=2, label=f"{label_pos[i]}_{method}", color=colors[i])
        if plot_goal:
            axes[i].plot(iterations, df[f'translation_goal_l_{i}'].to_numpy(), linestyle="--", marker="o", markersize=2, alpha=0.5, label=f"{label_pos[i]}_goal")
        if plot_pollen and method != "pollen" and isinstance(df_pollen, pd.DataFrame):
            axes[i].plot(iterations, df_pollen[f'translation_l_{i}'].to_numpy(), linestyle="--", marker="o", markersize=2, alpha=0.5, label=f"{label_pos[i]}_pollen")

        axes[i].set_ylabel(f"Position {label_pos[i]} (m)")
        axes[i].legend()

    fig.suptitle("Evolution of position for each pose (l_arm)", fontsize=12)
    axes[-1].set_xlabel("Iterations")
    
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_translations.png", dpi=300)
    plt.show()

def plot_movement_draw(path, df, method, plot_goal=True, plot_pollen=True, df_pollen=None):
    """
    Plots the movement of left and right arm in a 2D plane.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(df['translation_l_1'].to_numpy(), df['translation_l_2'].to_numpy(), marker="o", markersize=2, label=f"l_arm_{method}")
    ax.plot(df['translation_r_1'].to_numpy(), df['translation_r_2'].to_numpy(), marker="o", markersize=2, label=f"r_arm_{method}")

    if plot_goal:
        ax.plot(df['translation_goal_l_1'].to_numpy(), df['translation_goal_l_2'].to_numpy(), linestyle="--", marker="o", alpha=0.5, markersize=2, label="l_arm_goal")
        ax.plot(df['translation_goal_r_1'].to_numpy(), df['translation_goal_r_2'].to_numpy(), linestyle="--", marker="o", alpha=0.5, markersize=2, label="r_arm_goal")

    if plot_pollen and method != "pollen" and isinstance(df_pollen, pd.DataFrame):
        ax.plot(df_pollen['translation_l_1'].to_numpy(), df_pollen['translation_l_2'].to_numpy(), linestyle="--", marker="o", alpha=0.5, markersize=2, label="l_arm_pollen")
        ax.plot(df_pollen['translation_r_1'].to_numpy(), df_pollen['translation_r_2'].to_numpy(), linestyle="--", marker="o", alpha=0.5, markersize=2, label="r_arm_pollen")

    ax.set_xlabel('Position y (m)')
    ax.set_ylabel('Position z (m)')
    ax.set_title(f'Draw of movement in torso frame with method: {method}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_movement_draw.png", dpi=300)
    plt.show()


def add_distance_metric(path, csv_filename):
    """
    Adds a computed distance metric to the CSV file by calculating the Euclidean
    distance between two translation vectors (left and right arms).
    
    Parameters:
        path (str): Directory containing the CSV file.
        csv_filename (str): Name of the CSV file to be modified.
    """
    df = pd.read_csv(os.path.join(path, csv_filename))

    # Compute the Euclidean distance between left and right translations
    translations_l = np.array([df[f'translation_l_{i}'] for i in range(3)]).T
    translations_r = np.array([df[f'translation_r_{i}'] for i in range(3)]).T
    distances = np.linalg.norm(translations_l - translations_r, axis=1)

    df["Distance"] = distances
    df.to_csv(os.path.join(path, csv_filename), index=False)


def merge_csv_file(path, method, csv_filename):
    """
    Merges two CSV files containing left and right arm metrics into a single file.
    
    Parameters:
        path (str): Directory containing the CSV files.
        method (str): Method identifier for the CSV file names.
        csv_filename (str): Name of the resulting merged CSV file.
    """
    df1 = pd.read_csv(os.path.join(path, f"metrics_{method}_l.csv"))
    df2 = pd.read_csv(os.path.join(path, f"metrics_{method}_r.csv"))

    df_merged = pd.concat([df1, df2], ignore_index=False, axis=1)
    df_merged.to_csv(os.path.join(path, csv_filename), index=False)


def plot_all(method, path):
    """
    Generates and saves multiple plots based on the specified method and data.
    
    Parameters:
        method (str): Method identifier for file naming.
        path (str): Directory containing the CSV files.
    """
    csv_filename = f"metrics_{method}.csv"

    # Merge left and right arm data
    merge_csv_file(path, method, csv_filename)
    
    # If the method is "pink_sphere", compute and plot the distance metric
    if method == "pink_sphere": 
        add_distance_metric(path, csv_filename)
        df = pd.read_csv(os.path.join(path, csv_filename))
        plot_metrics_for_pink_sphere(path, df, method, 0.20)
    
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, csv_filename))
    # df = pd.read_csv(os.path.join(path, csv_filename), skiprows=range(1, 21))

    path_pollen = f"/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics/pollen"
    if os.path.exists(os.path.join(path_pollen, f"metrics_pollen_l.csv")) and os.path.exists(os.path.join(path_pollen, f"metrics_pollen_r.csv")) : 
        
        csv_filename_pollen = f"metrics_pollen.csv"
        merge_csv_file(path_pollen, "pollen", csv_filename_pollen)
        df_pollen = pd.read_csv(os.path.join(path_pollen, csv_filename_pollen))
        # df_pollen = pd.read_csv(os.path.join(path, csv_filename), skiprows=range(1, 21))

    # Generate and save various plots
    plot_q(path, df, method)
    if method!= "pollen" : 
        plot_velocity(path, df, method)
        plot_velocity_std(path, df, method)
    
    plot_ecart_q(path, df, method, tolerance=0.8)
    plot_ecart_pos_rot(path, df, method)
    plot_translations(path, df, method, plot_goal=True, plot_pollen=False, df_pollen=df_pollen)
    plot_movement_draw(path, df, method, plot_goal=True, plot_pollen=False, df_pollen=df_pollen)

##################
##################
method = "pink_V2"
path = f"/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics/{method}"
plot_all(method, path) 


