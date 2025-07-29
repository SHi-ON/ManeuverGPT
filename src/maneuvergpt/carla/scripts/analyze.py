import math
import pathlib
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from scipy.interpolate import make_interp_spline

# Set default renderer to save plots as HTML files
pio.renderers.default = 'browser'

LOGS_DIR = pathlib.Path('src/maneuvergpt/carla/logs/j_turn')

# Define custom color palette (CloseUp Color Palette by Lukas Keney)
custom_palette = ['#D3C94B', '#d39493', '#79c3b0', '#5658c9', '#925165']


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
    Calculate rotational velocity ('v_rot') from the 'yaw' changes over time.
    Deprecated: Use 'yaw_rate' column directly if available.

    :param df: DataFrame with 'yaw' and 'time' columns
    :return: DataFrame with an added 'v_rot' column
    """
    warnings.warn(
        message='calculate_rotational_velocity is deprecated and'
        ' will be removed in a future release.',
        category=DeprecationWarning,
        stacklevel=2,
    )
    if 'yaw_rate' in df.columns:
        df['v_rot'] = df['yaw_rate']
        return df

    df = df.copy()
    df['v_rot'] = df['yaw'].diff() / df['time'].diff()
    df['v_rot'].fillna(0, inplace=True)  # Handle NaN for the first row
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


def smooth_data(mean, common_time, smoothing_factor=500):
    """
    Apply spline smoothing to the mean data.

    :param mean: Numpy array of mean velocities
    :param common_time: Numpy array of common time points
    :param smoothing_factor: Number of points for spline interpolation
    :return: Smoothed mean velocities and corresponding time points
    """
    spl = make_interp_spline(common_time, mean, k=3)
    smooth_time = np.linspace(
        common_time.min(), common_time.max(), smoothing_factor
    )
    smooth_mean = spl(smooth_time)
    return smooth_time, smooth_mean


def plot_velocity(
    common_time,
    mean_velocities,
    ci_velocities,
    output_file='vehicle_velocities.html',
):
    """
    Plot longitudinal, lateral, and rotational velocities with confidence intervals.

    :param common_time: Numpy array of common time points
    :param mean_velocities: Dictionary with mean velocity data
    :param ci_velocities: Dictionary with confidence interval data
    :param output_file: Filename for the output HTML plot
    """
    # Define line colors using the custom palette
    line_colors = {
        'vx': custom_palette[3],  # Blue (#5658c9)
        'vy': custom_palette[2],  # Teal (#79c3b0)
        'vz': custom_palette[0],  #
        'v_rot': custom_palette[4],  # Burgundy (#925165)
    }

    # Define fill colors with RGBA format for transparency
    fill_colors = {
        'vx': f'rgba({int(custom_palette[3][1:3], 16)}, {int(custom_palette[3][3:5], 16)}, {int(custom_palette[3][5:7], 16)}, 0.3)',
        'vy': f'rgba({int(custom_palette[2][1:3], 16)}, {int(custom_palette[2][3:5], 16)}, {int(custom_palette[2][5:7], 16)}, 0.3)',
        'vz': f'rgba({int(custom_palette[0][1:3], 16)}, {int(custom_palette[0][3:5], 16)}, {int(custom_palette[0][5:7], 16)}, 0.3)',
        'v_rot': f'rgba({int(custom_palette[4][1:3], 16)}, {int(custom_palette[4][3:5], 16)}, {int(custom_palette[4][5:7], 16)}, 0.3)',
    }

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each velocity component
    for key in mean_velocities.keys():
        # Smooth the data for better visualization
        smooth_time, smooth_mean = smooth_data(
            mean_velocities[key], common_time
        )
        smooth_ci = make_interp_spline(common_time, ci_velocities[key], k=3)(
            smooth_time
        )

        # Add mean velocity line with increased width for boldness
        fig.add_trace(
            go.Scatter(
                x=smooth_time,
                y=smooth_mean,
                mode='lines',
                name=f'Mean {key}',
                line=dict(color=line_colors[key], width=2),
            )
        )

        # Add confidence interval shading
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([smooth_time, smooth_time[::-1]]),
                y=np.concatenate(
                    [smooth_mean + smooth_ci, (smooth_mean - smooth_ci)[::-1]]
                ),
                fill='toself',
                fillcolor=fill_colors[key],
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name=f'95% CI {key}',
            )
        )

    # Add mathematical annotations
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref='paper',
        yref='paper',
        text=r'<b>Confidence Interval :</b> Δ(v) = 1.96 × (σ / √n)',
        showarrow=False,
        font=dict(size=20),
        align='center',
    )

    # Update layout for better aesthetics
    fig.update_layout(
        title='Vehicle Velocities During J-Turn Maneuver (Body Frame)',
        xaxis_title='Time (s)',
        yaxis_title='Velocity (m/s, deg/s)<br><sub>vx: Longitudinal (forward+), vy: Lateral (left+), v_rot: Yaw rate</sub>',
        legend_title='Components',
        template='plotly_white',
        font=dict(size=42, weight='bold'),
        hovermode='x unified',
        legend=dict(
            font=dict(size=36, weight='bold'),
            # bordercolor=custom_palette[0],
            borderwidth=2,
        ),
    )

    # Save the figure as an HTML file and open in the browser
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f'Plot saved to {output_file} and opened in your default browser.')


def main(test_mode=False, num_files=None):
    """
    Main function to process CSV files and generate plots.

    :param test_mode: Boolean indicating whether to run in test mode (single CSV)
    :param num_files: Integer specifying the number of CSV files to process
    """
    # Debug: Check what's happening
    import os

    print(f'Current working directory: {os.getcwd()}')
    print(f'LOGS_DIR path: {LOGS_DIR}')
    print(f'LOGS_DIR absolute path: {LOGS_DIR.resolve()}')
    print(f'LOGS_DIR exists: {LOGS_DIR.exists()}')

    if LOGS_DIR.exists():
        all_files = list(LOGS_DIR.iterdir())
        csv_files = [f for f in all_files if f.suffix == '.csv']
        print(f'Total files in directory: {len(all_files)}')
        print(f'CSV files found: {len(csv_files)}')
        if csv_files:
            print(f'First few CSV files: {csv_files[:3]}')
    else:
        print(f'Directory does not exist!')
        # Let's check the parent directories
        parent = LOGS_DIR.parent
        print(f'Parent directory ({parent}) exists: {parent.exists()}')
        if parent.exists():
            print(f'Parent contents: {list(parent.iterdir())}')

    # Retrieve all CSV file paths
    file_paths = sorted(LOGS_DIR.glob('*.csv'))
    print(f"glob('*.csv') returned {len(file_paths)} files")

    # Handle test mode and limit number of files if specified
    if test_mode:
        file_paths = file_paths[:1]
        print(f'Running in test mode. Processing only: {file_paths[0]}\n')
    elif num_files is not None:
        original_count = len(file_paths)
        file_paths = file_paths[:num_files]
        print(
            f'Found {original_count} CSV files. Processing first {len(file_paths)} files.\n'
        )
    else:
        print(f'Found {len(file_paths)} CSV files. Processing all files.\n')

    if not file_paths:
        raise ValueError('No CSV files found in the specified directory.')

    all_interpolated = []
    velocity_columns = [
        'vx',
        'vy',
        'vz',
        'v_rot',
    ]  # Body frame: vx=longitudinal, vy=lateral, v_rot=yaw_rate

    # First, normalize all DataFrames and calculate rotational velocity
    normalized_dfs = []
    for idx, file_path in enumerate(file_paths, start=1):
        print(f'Reading file {idx}/{len(file_paths)}: {file_path}')
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' not in df.columns:
                print(f"Skipping {file_path}: 'timestamp' column missing.\n")
                continue
            df = normalize_time(df)

            # Transform from world frame to body frame
            df = transform_to_body_frame(df)

            df = calculate_rotational_velocity(df)

            # Ensure required velocity columns exist
            for col in velocity_columns:
                if col not in df.columns:
                    print(f"Skipping {file_path}: '{col}' column missing.\n")
                    break
            else:
                normalized_dfs.append(df)
                print(f'Loaded DataFrame with {len(df)} rows.\n')
        except Exception as e:
            print(f'Error reading {file_path}: {e}\n')

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
    print(
        f'Common time range: {common_time[0]:.2f}s to {common_time[-1]:.2f}s with dt={dt}s.\n'
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
            'mean_vz': mean_velocities['vz'],
            'ci_vz': ci_velocities['vz'],
            'mean_v_rot': mean_velocities['v_rot'],
            'ci_v_rot': ci_velocities['v_rot'],
        }
    )

    aggregated_csv_path = 'aggregated_data.csv'
    aggregated_data.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated data saved to '{aggregated_csv_path}'.\n")

    # Display sample of aggregated data
    print('Sample of aggregated data:')
    print(aggregated_data.head())
    print(aggregated_data.tail())
    print('\n')

    # Plot the velocities with confidence intervals
    plot_velocity(common_time, mean_velocities, ci_velocities)


if __name__ == '__main__':
    # To process all CSVs: main(test_mode=False, num_files=None)
    main(test_mode=False, num_files=100)
