import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
sys.path.append('/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/')

def plot_metrics_for_pink_sphere(path, filename, method, d_min):
    """
    Plots the evolution of the distance between two pink spheres.
    
    Parameters:
        path (str): Directory containing the CSV file.
        filename (str): Name of the CSV file containing the metrics.
        method (str): Method identifier for saving the plot.
        d_min (float): Minimum distance threshold to be plotted as a horizontal line.
    """
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

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

def plot_q(path, filename, method):
    """
    Plots the deviation of the q values for the left and right arms using subplots,
    with the primary Y-axis in radians (fractions of π) and a secondary Y-axis in degrees.
    
    Parameters:
        path (str): Directory containing the CSV file.
        filename (str): Name of the CSV file containing the metrics.
        method (str): Method identifier for saving the plot.
    """
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm deviations
    ecart_l_cols = [col for col in df.columns if col.startswith("q_l")]
    ecart_r_cols = [col for col in df.columns if col.startswith("q_r")]

    # Definition of ticks in rad
    radian_ticks = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    radian_labels = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "0", 
                     r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]

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

    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_q.png", dpi=300)
    plt.show()

def plot_ecart_q(path, filename, method, tolerance):
    """
    Plots the deviation (écart) of the q values for the left and right arms using subplots.
    
    Parameters:
        filename (str): Path to the CSV file containing the metrics.
        tolerance (float): The tolerance threshold to be plotted as a horizontal line.
    """
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm deviations
    ecart_l_cols = [col for col in df.columns if col.startswith("ecart_q_l")]
    ecart_r_cols = [col for col in df.columns if col.startswith("ecart_q_r")]

    # Definition of ticks in rad
    radian_ticks = np.array([0, np.pi/12, np.pi/6, np.pi/4])
    radian_labels = ["0", r"$\frac{\pi}{12}$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{4}$"]

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
    
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_ecarts_q.png", dpi=300)
    plt.show()


def plot_ecart_pos(path, filename, method):
    """
    Plots the positional error (écart position) for both left and right arms.
    
    Parameters:
        filename (str): Path to the CSV file containing the metrics.
    """
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

    # Create the plot
    plt.figure(figsize=(8, 5))

    # Convert columns to numpy arrays
    iterations = df["Iteration"].to_numpy()

    # if method == "pink_sphere" : 
    ecart_l = df["Ecart_pos_l"].to_numpy()
    ecart_r = df["Ecart_pos_r"].to_numpy()

    plt.plot(iterations, ecart_l, marker="o", label="Left Pos Error")
    plt.plot(iterations, ecart_r, marker="o", label="Right Pos Error")

    # Add labels, title and legend
    plt.xlabel("Iteration")
    plt.ylabel("Position Error (m)")
    plt.title("Positional Error of each Pose")
    plt.legend()

    # Save and display the plot
    plt.savefig(f"{path}/{method}_plot_ecart_pos.png", dpi=300)
    plt.show()


def plot_ecart_rot(path, filename, method):
    """
    Plots the rotational error (écart rotation) for both left and right arms, 
    with the primary Y-axis in radians and a secondary Y-axis in degrees.
    
    Parameters:
        path (str): Path to the directory containing the file.
        filename (str): CSV file containing the metrics.
        method (str): Method name for saving the plot.
    """
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

    # Create the figure and primary Y-axis (radians)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Convert columns to numpy arrays
    iterations = df["Iteration"].to_numpy()
    ecart_l = df["Ecart_rot_l"].to_numpy()
    ecart_r = df["Ecart_rot_r"].to_numpy()

    # Convert degrees to radians
    ecart_l_rad = np.radians(ecart_l)
    ecart_r_rad = np.radians(ecart_r)

    # Plot data in radians
    ax1.plot(iterations, ecart_l_rad, marker="o", label="Left Rot Error (rad)")
    ax1.plot(iterations, ecart_r_rad, marker="o", label="Right Rot Error (rad)")

    # Labels and title
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Rotational Error (radians)", color='black')
    ax1.set_title("Rotational Error of each Pose")
    
    # Set Y-ticks in fractions of π
    radian_ticks = np.array([0, 15, 30])  # Degrees
    radian_labels = ["0", r"$\frac{\pi}{12}$", r"$\frac{\pi}{6}$"]
    ax1.set_yticks(np.radians(radian_ticks))
    ax1.set_yticklabels(radian_labels)

    # Secondary Y-axis for degrees
    ax2 = ax1.twinx()
    ax2.set_ylabel("Rotational Error (deg)", color='black')
    ax2.set_ylim(ax1.get_ylim())  # Ensure same range

    # Set Y-ticks for degrees
    ax2.set_yticks(np.radians(radian_ticks))
    ax2.set_yticklabels(radian_ticks)

    # Legends
    ax1.legend(loc="upper left")

    # Save and show plot
    plt.savefig(f"{path}/{method}_plot_ecart_rot_q.png", dpi=300)
    plt.show()


def plot_translations(path, filename, method):
    """
    Plots the deviation (écart) of the q values for the left and right arms using subplots.
    
    Parameters:
        filename (str): Path to the CSV file containing the metrics.
        tolerance (float): The tolerance threshold to be plotted as a horizontal line.
    """
    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

    # Create subplots for left and right arm deviations
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    iterations = df["Iteration"].to_numpy()

    # Get the column names for left and right arm deviations

    label_pos = ["x", "y", "z"]
    colors = ["r", "g", "b"]

    for i in range(3):
        axes[i].plot(iterations, df[f'translation_l_{i}'].to_numpy(), marker="o", label=label_pos[i], color=colors[i])
        axes[i].set_ylabel(f"Position {label_pos[i]} (m)")
        axes[i].legend()

    plt.title("Evolution of position for each pose")
    plt.tight_layout()
    plt.savefig(f"{path}/{method}_plot_translation.png", dpi=300)
    plt.show()

def plot_draw(path, filename, method, color="b"):
    """
    Plots the rotational error (écart rotation) for both left and right arms.
    
    Parameters:
        filename (str): Path to the CSV file containing the metrics.
    """

    # Load data from CSV
    df = pd.read_csv(os.path.join(path, filename))

    # Plot y et z
    plt.plot(df['translation_l_1'].to_numpy(), df['translation_l_2'].to_numpy(), marker="o", markersize = "2", label = "l_arm")
    plt.plot(df['translation_r_1'].to_numpy(), df['translation_r_2'].to_numpy(), marker="o", markersize = "2", label = "r_arm")

    plt.xlabel('Position y (m)')
    plt.ylabel('Position z (m)')
    plt.title('Draw of mouvement in torso frame')
    plt.legend()

    # Save and display the plot
    plt.savefig(f"{path}/{method}_plot_draw.png", dpi=300)
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
        plot_metrics_for_pink_sphere(path, csv_filename, method, 0.20)

    # Generate and save various plots
    plot_q(path, csv_filename, method)
    plot_ecart_q(path, csv_filename, method, tolerance=0.8)
    plot_ecart_pos(path, csv_filename, method)
    plot_ecart_rot(path, csv_filename, method)
    plot_translations(path, csv_filename, method)
    plot_draw(path, csv_filename, method)

##################
##################
method = "pink_sphere"
path = f"/home/reachy/dev/reachy2_symbolic_ik/src/reachy2_symbolic_ik/csv_files_for_metrics.py/{method}"
plot_all(method, path) 


