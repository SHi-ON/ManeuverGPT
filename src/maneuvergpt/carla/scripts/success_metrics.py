import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns

# Define the directory containing the CSV logs
LOGS_DIR = pathlib.Path(__file__).parent / 'logs'
OUTPUT_DIR = LOGS_DIR / 'plots'

def normalize_time(df):
    """
    Normalize the 'timestamp' column to start from zero.

    Args:
        df (pd.DataFrame): DataFrame with a 'timestamp' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'time' column starting from zero.
    """
    df = df.copy()
    df['time'] = (df['timestamp'] - df['timestamp'].min()) / 1000.0  # Convert to seconds
    return df

def calculate_metrics(df):
    """
    Calculate required metrics from the DataFrame.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        dict: Dictionary containing all calculated metrics.
    """
    df = df.copy()

    # Angle Difference
    initial_yaw = df.iloc[0]['pitch']
    final_yaw = df.iloc[-1]['pitch']
    angle_diff = abs(final_yaw - initial_yaw)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    df['angle_diff'] = angle_diff

    # Max and Average Roll/Pitch
    max_roll = df['roll'].abs().max()
    max_pitch = df['pitch'].abs().max()
    avg_roll = df['roll'].abs().mean()
    avg_pitch = df['pitch'].abs().mean()

    # Yaw Rate
    df['yaw_rate'] = df['yaw'].diff() / df['time'].diff()
    yaw_rate = df['yaw_rate'].abs().max()

    # Time Taken
    time_taken = df['time'].iloc[-1] - df['time'].iloc[0]

    # Distance Traveled
    df['delta_x'] = df['x'].diff()
    df['delta_y'] = df['y'].diff()
    df['delta_z'] = df['z'].diff()
    df['delta_distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2 + df['delta_z']**2)
    distance_traveled = df['delta_distance'].sum()

    # Jerk Calculation
    df['ax_diff'] = df['ax'].diff()
    df['ay_diff'] = df['ay'].diff()
    df['az_diff'] = df['az'].diff()
    df['jerk'] = np.sqrt(df['ax_diff']**2 + df['ay_diff']**2 + df['az_diff']**2) / df['time'].diff()
    max_jerk = df['jerk'].abs().max()

    # Safety Margin
    # Assuming there's a 'min_distance' column indicating distance to the nearest object
    if 'min_distance' in df.columns:
        min_safety_distance = df['min_distance'].min()
    else:
        min_safety_distance = np.nan  # Not available

    metrics = {
        'angle_difference': angle_diff,
        'time_taken': time_taken,
        'distance_traveled': distance_traveled,
        'max_roll': max_roll,
        'avg_roll': avg_roll,
        'max_pitch': max_pitch,
        'avg_pitch': avg_pitch,
        'yaw_rate': yaw_rate,
        'max_jerk': max_jerk,
        'min_safety_distance': min_safety_distance
    }

    return metrics

def compute_success_ratio(metrics, weights=None):
    """
    Compute the success ratio based on multiple metrics.

    Args:
        metrics (dict): Dictionary containing all evaluated metrics.
        weights (dict, optional): Weights for each metric. Defaults to predefined weights.

    Returns:
        tuple: (success_ratio, score_components)
    """
    if weights is None:
        # Define default weights for each metric
        weights = {
            'angle_difference': 0.4,
            'time_taken': 0.2,
            'distance_traveled': 0.2,
            'max_jerk': 0.1,
            'yaw_rate': 0.05,
            'max_roll': 0.025,
            'max_pitch': 0.025
        }

    # Normalize each metric based on predefined thresholds
    # These thresholds should be adjusted based on domain knowledge
    normalized_metrics = {
        'angle_difference': max(0, 1 - metrics['angle_difference'] / 10),  # Assuming 10° is max acceptable difference
        'time_taken': max(0, 1 - abs(metrics['time_taken'] - 10) / 5),    # Assuming 10 seconds target
        'distance_traveled': max(0, min(metrics['distance_traveled'] / 50, 1)),  # Assuming 50 meters as target
        'max_jerk': max(0, 1 - metrics['max_jerk'] / 5),                # Assuming 5 m/s³ as max jerk
        'yaw_rate': max(0, 1 - metrics['yaw_rate'] / 20),              # Assuming 20 deg/s as max yaw rate
        'max_roll': max(0, 1 - metrics['max_roll'] / 15),              # Assuming 15 degrees max roll
        'max_pitch': max(0, 1 - metrics['max_pitch'] / 15)             # Assuming 15 degrees max pitch
    }

    # Compute weighted sum
    success_ratio = 0
    score_components = {}
    for metric, weight in weights.items():
        score = normalized_metrics.get(metric, 0)
        score_components[metric] = score
        success_ratio += weight * score

    # Clamp success_ratio between 0 and 1
    success_ratio = max(0, min(success_ratio, 1))

    return success_ratio, score_components

def plot_success_ratios(all_metrics, output_file='maneuver_success_ratios.png'):
    """
    Plot the success ratios across trials as a line graph with a plain text formula annotation.
    """
    success_ratios = [metrics['success_ratio'] * 100 for metrics in all_metrics]
    trials = list(range(1, len(success_ratios) + 1))
    
    # Use constrained_layout for clear spacing
    plt.figure(figsize=(8, 5), constrained_layout=True)
    plt.plot(trials, success_ratios, marker='o', linestyle='-', color='blue', label='Success Ratio (%)')
    plt.axhline(y=100, color='green', linestyle='--', label='Perfect Success (100%)')
    plt.axhline(y=60, color='red', linestyle='--', label='Good Success Threshold (70%)')
    plt.xlabel('Trial Number')
    plt.ylabel('Success Ratio (%)')
    plt.title('Maneuver Success Ratios Across Trials')
    plt.grid(True)
    plt.legend(loc='upper left')
    # Place plain text annotation in the top-right corner of the axes
    sr_formula = "Success Ratio = Σ (w_i * Normalized Metric_i)"
    plt.gca().text(0.98, 1.02, sr_formula, transform=plt.gca().transAxes,
             horizontalalignment='right', verticalalignment='bottom',
             fontsize=9, color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.savefig(OUTPUT_DIR / output_file, dpi=300)
    plt.close()
    print(f"Success Ratios plot saved to '{OUTPUT_DIR / output_file}'.")

def plot_key_metrics(all_metrics, output_file='key_metrics_overview.png'):
    """
    Generate a multi-panel line graph for key metrics:
     - Angle Difference
     - Time Taken
     - Distance Traveled
     - Max Jerk

    Each subplot includes a plain text formula annotation placed above the data.
    """
    # Extract metrics lists
    trials = list(range(1, len(all_metrics) + 1))
    angle_diff = [metrics['angle_difference'] for metrics in all_metrics]
    time_taken = [metrics['time_taken'] for metrics in all_metrics]
    distance_traveled = [metrics['distance_traveled'] for metrics in all_metrics]
    max_jerk = [metrics['max_jerk'] for metrics in all_metrics]

    # Calculate means to be shown in titles (optional)
    mean_angle = np.mean(angle_diff)
    mean_time = np.mean(time_taken)
    mean_distance = np.mean(distance_traveled)
    mean_jerk = np.mean(max_jerk)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Angle Difference Plot
    axs[0, 0].plot(trials, angle_diff, marker='o', linestyle='-', color='blue', label='Angle Diff (°)')
    axs[0, 0].set_xlabel('Trial Number')
    axs[0, 0].set_ylabel('Angle Difference (°)')
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc='upper left')
    angle_formula = "Angle Diff = |final_yaw - initial_yaw| if ≤ 180° else 360° - |final_yaw - initial_yaw|"
    axs[0, 0].text(0.98, 1.02, angle_formula, transform=axs[0, 0].transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   fontsize=9, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Time Taken Plot
    axs[0, 1].plot(trials, time_taken, marker='^', linestyle='-', color='green', label='Time Taken (s)')
    axs[0, 1].set_xlabel('Trial Number')
    axs[0, 1].set_ylabel('Time Taken (s)')

    axs[0, 1].grid(True)
    axs[0, 1].legend(loc='upper left')
    time_formula = "Time Taken = final_time - initial_time"
    axs[0, 1].text(0.98, 1.02, time_formula, transform=axs[0, 1].transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   fontsize=9, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Distance Traveled Plot
    axs[1, 0].plot(trials, distance_traveled, marker='d', linestyle='-', color='magenta', label='Distance (m)')
    axs[1, 0].set_xlabel('Trial Number')
    axs[1, 0].set_ylabel('Distance Traveled (m)')
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc='upper left')
    distance_formula = "Distance = Σ (Euclidean distance between consecutive positions)"
    axs[1, 0].text(0.98, 1.02, distance_formula, transform=axs[1, 0].transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   fontsize=9, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Max Jerk Plot
    axs[1, 1].plot(trials, max_jerk, marker='s', linestyle='-', color='red', label='Max Jerk (m/s³)')
    axs[1, 1].set_xlabel('Trial Number')
    axs[1, 1].set_ylabel('Max Jerk (m/s³)')
    axs[1, 1].set_title(f'Max Jerk Across Trials\n(mean ≈ {mean_jerk:.2f} m/s³)')
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='upper left')
    jerk_formula = "Max Jerk = max( sqrt((Δax)² + (Δay)² + (Δaz)²) / Δt )"
    axs[1, 1].text(0.98, 1.02, jerk_formula, transform=axs[1, 1].transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   fontsize=9, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.savefig(OUTPUT_DIR / output_file, dpi=300)
    plt.close()
    print(f"Key Metrics Overview plot saved to '{OUTPUT_DIR / output_file}'.")

def plot_correlation_matrix(metrics_df, output_file='correlation_matrix.png'):
    plt.figure(figsize=(8, 6), constrained_layout=True)
    corr = metrics_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Metrics")
    plt.savefig(OUTPUT_DIR / output_file, dpi=300)
    plt.close()
    print(f"Correlation matrix plot saved to '{OUTPUT_DIR / output_file}'.")

def main(test_mode=False, num_files=None):
    """
    Main function to process CSV files and compute success ratios.

    Args:
        test_mode (bool, optional): Whether to run in test mode (single CSV). Defaults to False.
        num_files (int, optional): Number of CSV files to process. Defaults to None (process all).
    """
    # Retrieve all CSV file paths
    file_paths = sorted(LOGS_DIR.glob('*.csv'))

    # Handle test mode and limit number of files if specified
    if test_mode:
        file_paths = file_paths[:1]
        print(f"Running in test mode. Processing only: {file_paths[0]}\n")
    elif num_files is not None:
        file_paths = file_paths[:num_files]
        print(f"Processing the first {len(file_paths)} out of {num_files} specified CSV files.\n")
    else:
        print(f"Found {len(file_paths)} CSV files. Processing all files.\n")

    if not file_paths:
        raise ValueError("No CSV files found in the specified directory.")

    all_metrics = []

    # Create plots directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each CSV file
    for idx, file_path in enumerate(file_paths, start=1):
        print(f"Processing file {idx}/{len(file_paths)}: {file_path}")
        try:
            df = pd.read_csv(file_path)
            required_columns = ['timestamp', 'x', 'y', 'z', 'vx', 'vy', 'vz', 
                                'ax', 'ay', 'az', 'roll', 'pitch', 'yaw']
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {file_path.name}: Missing required columns.\n")
                continue
            df = normalize_time(df)
            metrics = calculate_metrics(df)
            success_ratio, score_components = compute_success_ratio(metrics)
            metrics['success_ratio'] = success_ratio
            all_metrics.append(metrics)
            print(f"File '{file_path.name}' processed. Success Ratio: {success_ratio * 100:.2f}%\n")
        except Exception as e:
            print(f"Error processing '{file_path.name}': {e}\n")

    if not all_metrics:
        raise ValueError("No valid metrics to process. Please check the input files.")

    # Save aggregated metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    aggregated_csv_path = LOGS_DIR / 'aggregated_success_metrics.csv'
    metrics_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated success metrics saved to '{aggregated_csv_path}'.\n")

    # Display summary statistics
    print("Summary Statistics:")
    print(metrics_df.describe())
    print("\n")

    # Generate Plots
    plot_success_ratios(all_metrics)
    plot_key_metrics(all_metrics)
    plot_correlation_matrix(metrics_df)

if __name__ == '__main__':
    # Example usage:
    # To process all CSVs: main(test_mode=False, num_files=None)
    # To process first 10 CSVs: main(test_mode=False, num_files=10)
    # To run in test mode (single CSV): main(test_mode=True, num_files=1)
    main(test_mode=False, num_files=600)