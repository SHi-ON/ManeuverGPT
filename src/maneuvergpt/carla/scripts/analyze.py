import logging
import math
import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

# Set matplotlib backend for better PDF support
plt.style.use('default')

LOGS_DIR = pathlib.Path('src/maneuvergpt/carla/logs/j_turn')

# CloseUp by Lukas Keney
color_palette = ['#393449', '#d39493', '#79c3b0', '#5658c9', '#925165']


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[: truncate - 1] + '\u2026') if len(name) > truncate else name


def world_to_body(v, yaw_deg):
    """Rotate CARLA world-frame velocity into the vehicle body frame."""
    yaw_rad = math.radians(yaw_deg)
    v_lon = v.x * math.cos(yaw_rad) + v.y * math.sin(yaw_rad)
    v_lat = -v.x * math.sin(yaw_rad) + v.y * math.cos(yaw_rad)
    v_alt = v.z  # No rotation applied if assuming flat terrain
    return v_lon, v_lat, v_alt


def transform_to_body_frame(df):
    """Transform world-frame to body-frame velocities using yaw angle."""
    df = df.copy()

    # Create a simple velocity vector class for compatibility with world_to_body function
    class SimpleVector:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    # Transform each row
    v_longitudinal = []
    v_latitudinal = []
    v_altitudinal = []

    for _, row in df.iterrows():
        v = SimpleVector(row['vx'], row['vy'], row['vz'])
        v_lon, v_lat, v_alt = world_to_body(v, row['yaw'])
        v_longitudinal.append(v_lon)
        v_latitudinal.append(v_lat)
        v_altitudinal.append(v_alt)

    # Replace world-frame velocities with body-frame velocities
    df['vx'] = v_longitudinal  # Longitudinal velocity (forward/backward)
    df['vy'] = v_latitudinal  # Lateral velocity (left/right)
    df['vz'] = v_altitudinal  # Vertical velocity (up/down, if applicable)

    return df


def normalize_time(df):
    """
    Normalize the 'timestamp' column to start from zero.

    :param df: DataFrame with a 'timestamp' column
    :return: DataFrame with an added 'time' column starting from zero
    """
    df = df.copy()
    df['time'] = (
        df['timestamp'] - df['timestamp'].min()
    ) / 1000.0  # Convert to seconds
    return df


def calculate_rotational_velocity(df):
    """
    Calculate rotational velocity ('yaw_rate') from the 'yaw' changes over time.
    Deprecated: Use 'yaw_rate' column directly, if available.
    """
    warnings.warn(
        message='calculate_rotational_velocity is deprecated and'
        ' will be removed in a future release.',
        category=DeprecationWarning,
        stacklevel=2,
    )

    df = df.copy()
    df['yaw_rate'] = df['yaw'].diff() / df['time'].diff()
    df['yaw_rate'].fillna(0, inplace=True)  # Handle NaN for the first row
    return df


def interpolate_to_common_time(df, common_time, velocity_columns):
    """
    Interpolate velocity columns to a common time base.

    :param df: DataFrame with 'time' and velocity columns
    :param common_time: Numpy array of common time points
    :param velocity_columns: List of velocity column names to interpolate
    :return: Dictionary with interpolated velocity data
    """
    interpolated_data = {}
    for col in velocity_columns:
        interpolated_data[col] = np.interp(common_time, df['time'], df[col])
    return interpolated_data


def calculate_statistics(interpolated_data, velocity_columns):
    """
    Calculate mean and 95% confidence intervals for each velocity component.

    :param interpolated_data: List of dictionaries with interpolated velocity data
    :param velocity_columns: List of velocity column names
    :return: Dictionaries for mean velocities and confidence intervals
    """
    data_matrix = {col: [] for col in velocity_columns}
    for trial in interpolated_data:
        for col in velocity_columns:
            data_matrix[col].append(trial[col])

    mean_velocities = {}
    ci_velocities = {}
    n = len(interpolated_data)
    for col in velocity_columns:
        data_array = np.array(data_matrix[col])
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0, ddof=1)
        se = std / np.sqrt(n)
        ci = 1.96 * se  # 95% Confidence Interval
        mean_velocities[col] = mean
        ci_velocities[col] = ci
    return mean_velocities, ci_velocities


def smooth_data(common_time, mean_vel, ci_vel, smoothing_factor=500):
    """Apply spline smoothing to the data."""
    smooth_time = np.linspace(
        common_time.min(), common_time.max(), smoothing_factor
    )
    spline_mean = make_interp_spline(common_time, mean_vel, k=3)
    spline_ci = make_interp_spline(common_time, ci_vel, k=3)
    smooth_mean = spline_mean(smooth_time)
    smooth_ci = spline_ci(smooth_time)
    return smooth_time, smooth_mean, smooth_ci


def plot_velocity(
    common_time,
    mean_velocities,
    ci_velocities,
    output_file,
):
    """
    Plot longitudinal, lateral, and rotational velocities with confidence intervals.

    :param common_time: Numpy array of common time points
    :param mean_velocities: Dictionary with mean velocity data
    :param ci_velocities: Dictionary with confidence interval data
    :param output_file: Filename for the output PDF plot
    """
    # Define line colors using the custom palette
    line_colors = {
        'vx': color_palette[3],  # Blue (#5658c9)
        'vy': color_palette[2],  # Teal (#79c3b0)
        # 'vz': color_palette[0],
        'yaw_rate': color_palette[4],  # Burgundy (#925165)
    }

    labels = {
        'vx': {'mean': r'$\bar{v}_x$', 'ci': r'95% CI $v_x$'},
        'vy': {'mean': r'$\bar{v}_y$', 'ci': r'95% CI $v_y$'},
        # 'vz': {'mean': r'$\bar{v}_z$', 'ci': r'95% CI $v_z$'},
        'yaw_rate': {'mean': r'$\bar{\omega}$', 'ci': r'95% CI $\omega$'},
    }

    # Convert hex colors to RGB tuples for matplotlib
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    fill_colors = {
        key: hex_to_rgb(color) for key, color in line_colors.items()
    }

    # Create a matplotlib figure with high DPI for better PDF quality
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Add traces for each velocity component
    for k in mean_velocities.keys():
        # Smooth the data for better visualization
        smooth_time, smooth_mean, smooth_ci = smooth_data(
            common_time,
            mean_velocities[k],
            ci_velocities[k],
        )

        # Plot mean velocity line
        ax.plot(
            smooth_time,
            smooth_mean,
            label=labels[k]['mean'],
            color=line_colors[k],
            linewidth=1,
        )

        # Add confidence interval shading
        ax.fill_between(
            smooth_time,
            smooth_mean - smooth_ci,
            smooth_mean + smooth_ci,
            label=labels[k]['ci'],
            color=fill_colors[k],
            alpha=0.3,
        )

    # Add mathematical annotation
    ax.text(
        0.5,
        1.02,
        r'$\mathbf{95\%\ Confidence\ Interval:}$ $\Delta(v) = 1.96 \times ('
        r'\sigma / \sqrt{n})$',
        ha='center',
        va='bottom',
        transform=ax.transAxes,
        fontsize=14,
        color=color_palette[0],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
    )

    # Update layout for better aesthetics
    ax.set_title(
        'Vehicle Velocities During J-Turn Maneuver',
        fontsize=18,
        fontweight='bold',
        pad=40,
        color=color_palette[0],
    )
    ax.set_xlabel(
        'Time (s)', fontsize=14, fontweight='bold', color=color_palette[0]
    )
    ax.set_ylabel(
        'Velocity (m/s, deg/s)',
        fontsize=14,
        fontweight='bold',
        color=color_palette[0],
    )

    # Add subtitle for axis explanation
    ax.text(
        0.5,
        -0.15,
        'vx: Longitudinal (forward+), vy: Lateral (left+), yaw_rate: Yaw rate',
        ha='center',
        va='top',
        transform=ax.transAxes,
        fontsize=10,
        style='italic',
        color=color_palette[0],
    )

    # Customize legend
    legend = ax.legend(
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    # Set legend text color to #393449
    for text in legend.get_texts():
        text.set_color(color_palette[0])
    legend.get_title().set_color(color_palette[0])

    # Set tick label colors to #393449
    ax.tick_params(axis='both', which='major', colors=color_palette[0])
    ax.tick_params(axis='both', which='minor', colors=color_palette[0])

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Improve overall appearance
    plt.tight_layout()

    # Save the figure as a PDF file
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=150)

        # Add metadata to PDF
        d = pdf.infodict()
        d['Title'] = 'Vehicle Velocities During J-Turn Maneuver'
        d['Author'] = 'ManeuverGPT Paper Authors'
        d['Subject'] = 'CARLA Vehicle Dynamics Analysis'
        d['Keywords'] = 'CARLA, Vehicle Dynamics, J-Turn, Velocity Analysis'
        d['Creator'] = 'matplotlib'

    plt.close(fig)
    logging.info(f'Figure saved to {output_file}')


def main(debug=False):
    file_paths = sorted(LOGS_DIR.glob('*.csv'))
    if not file_paths:
        raise FileNotFoundError('No CSV files found.')
    logging.info(f'{len(file_paths)} CSV files found')
    if debug:
        file_paths = file_paths[:1]
        logging.debug('DeBuG MoDe, processing only the first file...')

    all_interpolated = []
    velocity_columns = [
        'vx',
        'vy',
        # 'vz',
        'yaw_rate',
    ]

    # First, normalize all DataFrames and calculate rotational velocity
    normalized_dfs = []
    for idx, file_path in tqdm(
        enumerate(file_paths, start=1),
        desc='Loading CSV files',
        total=len(file_paths),
    ):
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' not in df.columns:
                logging.warning(
                    f"Skipping {file_path}: 'timestamp' column missing!"
                )
                continue
            df = normalize_time(df)

            # Transform from world frame to body frame
            df = transform_to_body_frame(df)

            # Deprecated
            # df = calculate_rotational_velocity(df)

            # Ensure required velocity columns exist
            for col in velocity_columns:
                if col not in df.columns:
                    print(f"Skipping {file_path}: '{col}' column missing.\n")
                    break
            else:
                normalized_dfs.append(df)
                logging.debug(f'Loaded DataFrame with {len(df)} rows.\n')
        except Exception as e:
            logging.warning(f'Error reading {file_path}: {e}\n')

    if not normalized_dfs:
        raise ValueError(
            'No valid DataFrames to process. Please check the input files.'
        )

    # Determine the common time range (overlapping period)
    global_start = max(df['time'].min() for df in normalized_dfs)
    global_end = min(df['time'].max() for df in normalized_dfs)

    if global_end <= global_start:
        raise ValueError(
            'Global end time must be greater than global start time.'
        )

    # Define the common time base
    dt = 0.01  # Time step in seconds
    common_time = np.arange(global_start, global_end, dt)
    logging.info(
        f'Common time range: {common_time[0]:.2f}s to {common_time[-1]:.2f}s'
        f' with dt={dt}s.'
    )

    # Interpolate each DataFrame to the common time base
    for idx, df in enumerate(normalized_dfs, start=1):
        interpolated = interpolate_to_common_time(
            df, common_time, velocity_columns
        )
        all_interpolated.append(interpolated)

    # Calculate mean and confidence intervals
    mean_velocities, ci_velocities = calculate_statistics(
        all_interpolated, velocity_columns
    )

    # Optional: Save aggregated data to CSV for verification
    aggregated_data = pd.DataFrame(
        {
            'time': common_time,
            'mean_vx': mean_velocities['vx'],
            'ci_vx': ci_velocities['vx'],
            'mean_vy': mean_velocities['vy'],
            'ci_vy': ci_velocities['vy'],
            # 'mean_vz': mean_velocities['vz'],
            # 'ci_vz': ci_velocities['vz'],
            'mean_yaw_rate': mean_velocities['yaw_rate'],
            'ci_yaw_rate': ci_velocities['yaw_rate'],
        }
    )

    aggregated_csv_path = 'aggregated_data.csv'
    aggregated_data.to_csv(aggregated_csv_path, index=False)
    logging.info(f"Aggregated data saved to '{aggregated_csv_path}'")

    if debug:
        print('Sample of aggregated data:')
        print(aggregated_data.head())
        print('...')
        print(aggregated_data.tail())
        print('\n')

    plot_velocity(
        common_time,
        mean_velocities,
        ci_velocities,
        output_file='vehicle_velocities.pdf',
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main(debug=False)
