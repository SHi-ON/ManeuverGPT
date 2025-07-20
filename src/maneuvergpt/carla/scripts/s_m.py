import os
import pathlib

import numpy as np
import pandas as pd
from bokeh.io.export import export_svg
from bokeh.layouts import column, gridplot
from bokeh.models import (
    Button,
    ColorBar,
    CustomJS,
    Label,
    LinearColorMapper,
    Span,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, output_file, save
from selenium import webdriver
from selenium.webdriver.firefox.service import Service

# Specify the path to geckodriver using Service
geckodriver_path = '/snap/bin/geckodriver'
service = Service(geckodriver_path)

# Create the Firefox driver using the service
driver = webdriver.Firefox(service=service)

# -----------------------------
# Configuration thresholds and weights
# -----------------------------
JERK_THRESHOLD = 2.0  # m/s³ (lower jerk is better)
TIME_THRESHOLD = 5.0  # seconds (shorter maneuver is better)
IDEAL_DISTANCE = 10.0  # meters (ideal traveled distance)
ROLL_THRESHOLD = 5.0  # degrees (lower roll is better)
PITCH_THRESHOLD = 5.0  # degrees (lower pitch is better)
YAWRATE_THRESHOLD = 30.0  # deg/s (both max and avg)

ANGLE_WEIGHT = 0.50  # Angle component (based on error)
JERK_WEIGHT = 0.0  # Jerk is very important
TIME_WEIGHT = 0.0
DIST_WEIGHT = 0.0
ROLL_WEIGHT = 0.15
PITCH_WEIGHT = 0.0
YAWRATE_MAX_WEIGHT = 0.15
YAWRATE_AVG_WEIGHT = 0.25

# -----------------------------
# File paths and directories
# -----------------------------

LOGS_DIR = pathlib.Path(__file__).parent / 'logs/single'

OUTPUT_DIR = LOGS_DIR / 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Download Button Utility (Alternate Workaround)
# -----------------------------
def add_download_button(plot, filename):
    """
    Wrap the given Bokeh plot in a layout that includes a download button.
    The button's callback waits for the plot's canvas to be ready,
    then retrieves the underlying canvas element, converts it to a PNG data URL,
    and triggers a download using a temporary anchor element.
    """
    button = Button(
        label=f'Download {filename}', button_type='success', width=200
    )
    button.js_on_click(
        CustomJS(
            args=dict(plot=plot, filename=filename),
            code="""
        // Poll until the canvas is ready.
        function waitForCanvas(callback) {
            if (plot.canvas_view && plot.canvas_view.ctx && plot.canvas_view.ctx.canvas) {
                callback();
            } else {
                setTimeout(function(){ waitForCanvas(callback); }, 100);
            }
        }
        waitForCanvas(function(){
            var canvas = plot.canvas_view.ctx.canvas;
            var dataURL = canvas.toDataURL("image/png");
            var link = document.createElement('a');
            link.href = dataURL;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    """,
        )
    )
    return column(plot, button)


# -----------------------------
# Utility Smoothing Function (Simple Moving Average)
# -----------------------------
def smooth_data(data, window_size=5):
    data = np.array(data)
    if len(data) < window_size:
        return data
    window = np.ones(window_size) / float(window_size)
    return np.convolve(data, window, mode='same')


# -----------------------------
# Time normalization
# -----------------------------
def normalize_time(df):
    """
    Convert the 'timestamp' column into a 'time' column (seconds since start).
    """
    df = df.copy()
    df['time'] = (df['timestamp'] - df['timestamp'].min()) / 1000.0
    return df


# -----------------------------
# Compute Metrics from the DataFrame
# -----------------------------
def calculate_metrics(df):
    """
    Compute raw metrics with improved jerk calculation
    """
    df = df.copy()

    # ---- Angle Error in Degrees ----
    initial_yaw = df.iloc[0]['pitch']
    final_yaw = df.iloc[-1]['pitch']
    angle_diff = abs(final_yaw - initial_yaw)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    angle_error = abs(angle_diff - 180)  # Error in degrees

    # ---- Vehicle Stability Metrics ----
    max_roll = df['roll'].abs().max()
    avg_roll = df['roll'].abs().mean()
    max_pitch = df['pitch'].abs().max()
    avg_pitch = df['pitch'].abs().mean()

    # ---- Dynamic Motion Metrics ----
    # Yaw rate in degrees/second
    df['yaw_rate'] = df['yaw'].diff() / df['time'].diff()
    yaw_rate_max = df['yaw_rate'].abs().max()
    yaw_rate_avg = df['yaw_rate'].abs().mean()

    # Angular acceleration (deg/s²)
    df['yaw_acceleration'] = df['yaw_rate'].diff() / df['time'].diff()
    max_yaw_accel = df['yaw_acceleration'].abs().max()

    # ---- Execution Metrics ----
    time_taken = df['time'].iloc[-1] - df['time'].iloc[0]

    # Calculate path efficiency
    df['delta_x'] = df['x'].diff()
    df['delta_y'] = df['y'].diff()
    df['delta_z'] = df['z'].diff()
    df['delta_distance'] = np.sqrt(
        df['delta_x'] ** 2 + df['delta_y'] ** 2 + df['delta_z'] ** 2
    )
    distance_traveled = df['delta_distance'].sum()

    # Path smoothness using curvature
    df['curvature'] = np.abs(df['yaw_rate']) / np.sqrt(
        df['vx'] ** 2 + df['vy'] ** 2 + 1e-6
    )
    avg_curvature = df['curvature'].mean()

    # ---- Improved Jerk Calculation ----
    # Calculate acceleration changes
    df['ax_diff'] = df['ax'].diff()
    df['ay_diff'] = df['ay'].diff()
    df['az_diff'] = df['az'].diff()

    # Calculate jerk with proper time difference
    time_diff = df['time'].diff()
    df['jerk'] = np.sqrt(
        df['ax_diff'] ** 2 + df['ay_diff'] ** 2 + df['az_diff'] ** 2
    ) / (time_diff + 1e-6)

    # Apply rolling mean to smooth out spikes
    window_size = 5  # Adjust based on sampling rate
    df['jerk_smoothed'] = (
        df['jerk']
        .rolling(window=window_size, center=True)
        .mean()
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    # Remove outliers (values beyond 3 standard deviations)
    jerk_mean = df['jerk_smoothed'].mean()
    jerk_std = df['jerk_smoothed'].std()
    df['jerk_cleaned'] = df['jerk_smoothed'].clip(
        lower=jerk_mean - 3 * jerk_std, upper=jerk_mean + 3 * jerk_std
    )

    avg_jerk = df['jerk_cleaned'].mean()
    max_jerk = df['jerk_cleaned'].max()

    # ---- Speed Profile ----
    df['speed'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2 + df['vz'] ** 2)
    max_speed = df['speed'].max()
    avg_speed = df['speed'].mean()
    speed_variation = df['speed'].std() / (
        avg_speed + 1e-6
    )  # Coefficient of variation

    return {
        'angle_error_deg': angle_error,
        'time_taken': time_taken,
        'distance_traveled': distance_traveled,
        'max_jerk': max_jerk,
        'avg_jerk': avg_jerk,
        'max_roll_deg': max_roll,
        'avg_roll_deg': avg_roll,
        'max_pitch_deg': max_pitch,
        'avg_pitch_deg': avg_pitch,
        'yaw_rate_max_dps': yaw_rate_max,
        'yaw_rate_avg_dps': yaw_rate_avg,
        'max_yaw_accel': max_yaw_accel,
        'path_smoothness': avg_curvature,
        'speed_max': max_speed,
        'speed_avg': avg_speed,
        'speed_consistency': speed_variation,
    }


# -----------------------------
# Compute Overall Success Ratio
# -----------------------------
def compute_success_ratio(metrics):
    """
    Compute success scores using raw values and more intuitive thresholds.
    Higher scores for better performance (smaller errors).
    """
    # Thresholds
    ANGLE_ERROR_THRESHOLD = 7.0  # degrees from 180° (max acceptable error)
    ANGLE_ERROR_IDEAL = 2.0  # degrees (target error)
    AVG_JERK_THRESHOLD = 2.0  # m/s³
    AVG_JERK_IDEAL = 1.0  # m/s³
    TIME_THRESHOLD = 5.0  # seconds (max acceptable)
    TIME_IDEAL = 3.5  # seconds (target)
    ROLL_THRESHOLD = 10.0  # degrees
    PITCH_THRESHOLD = 10.0  # degrees
    YAW_RATE_THRESHOLD = 45.0  # deg/s

    # Compute individual scores (0 to 1, higher is better)
    # Angle score: 1.0 for perfect (0° error), 0.0 for threshold or worse
    angle_error = metrics['angle_error_deg']
    if angle_error <= ANGLE_ERROR_IDEAL:
        angle_score = 1.0
    else:
        angle_score = max(
            0.0,
            1.0
            - (
                (angle_error - ANGLE_ERROR_IDEAL)
                / (ANGLE_ERROR_THRESHOLD - ANGLE_ERROR_IDEAL)
            ),
        )

    # Jerk score: 1.0 for ideal or better, 0.0 for threshold or worse
    avg_jerk = metrics['avg_jerk']
    if avg_jerk <= AVG_JERK_IDEAL:
        jerk_score = 1.0
    else:
        jerk_score = max(
            0.0,
            1.0
            - (
                (avg_jerk - AVG_JERK_IDEAL)
                / (AVG_JERK_THRESHOLD - AVG_JERK_IDEAL)
            ),
        )

    # Time score: 1.0 for ideal or better, 0.0 for threshold or worse
    time_taken = metrics['time_taken']
    if time_taken <= TIME_IDEAL:
        time_score = 1.0
    else:
        time_score = max(
            0.0,
            1.0 - ((time_taken - TIME_IDEAL) / (TIME_THRESHOLD - TIME_IDEAL)),
        )

    # Other scores (linear scaling)
    roll_score = max(0.0, 1.0 - (metrics['avg_roll_deg'] / ROLL_THRESHOLD))
    pitch_score = max(0.0, 1.0 - (metrics['avg_pitch_deg'] / PITCH_THRESHOLD))
    yaw_rate_score = max(
        0.0, 1.0 - (metrics['yaw_rate_max_dps'] / YAW_RATE_THRESHOLD)
    )

    # Updated weights (total = 1.0)
    weights = {
        'angle': 0.40,  # Increased weight for angle accuracy
        'jerk': 0.25,  # Smoothness is important
        'time': 0.15,  # Reasonable completion time
        'roll': 0.07,  # Vehicle stability
        'pitch': 0.07,  # Vehicle stability
        'yaw_rate': 0.06,  # Turning behavior
    }

    # Compute weighted average
    overall_score = (
        weights['angle'] * angle_score
        + weights['jerk'] * jerk_score
        + weights['time'] * time_score
        + weights['roll'] * roll_score
        + weights['pitch'] * pitch_score
        + weights['yaw_rate'] * yaw_rate_score
    )

    # Scale the overall score to percentage
    overall_score = max(0.0, min(1.0, overall_score))

    component_scores = {
        'angle_error': angle_score,
        'jerk': jerk_score,
        'time': time_score,
        'roll': roll_score,
        'pitch': pitch_score,
        'yaw_rate': yaw_rate_score,
    }

    print('\nScore Components:')
    print(f'Angle Score: {angle_score * 100:.1f}% (Error: {angle_error:.1f}°)')
    print(
        f'Jerk Score: {jerk_score * 100:.1f}% (Avg Jerk: {avg_jerk:.2f} m/s³)'
    )
    print(f'Time Score: {time_score * 100:.1f}% (Time: {time_taken:.1f}s)')
    print(f'Roll Score: {roll_score * 100:.1f}%')
    print(f'Pitch Score: {pitch_score * 100:.1f}%')
    print(f'Yaw Rate Score: {yaw_rate_score * 100:.1f}%')
    print(f'Overall Score: {overall_score * 100:.1f}%\n')

    return overall_score, component_scores


# -----------------------------
# Plot Overall Success Ratios
# -----------------------------
def plot_success_ratios(all_metrics, download_option=False):
    overall_scores = []
    component_scores = []

    for m in all_metrics:
        overall, comps = compute_success_ratio(m)
        overall_scores.append(overall * 100)  # Convert to percentage
        component_scores.append(comps)

    trials = list(range(1, len(overall_scores) + 1))
    smooth_overall = smooth_data(overall_scores, window_size=5)

    p = figure(
        title='Overall Maneuver Success Ratio Across Trials',
        x_axis_label='Trial Number',
        y_axis_label='Overall Success Ratio (%)',
        tools='pan,wheel_zoom,box_zoom,reset,save',
        width=800,
        height=400,
    )

    # Enhanced styling
    p.title.text_font_size = '16pt'
    p.title.text_font_style = 'bold'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'

    # Plot raw data with increased visibility
    p.scatter(
        trials,
        overall_scores,
        size=4,
        color='gray',
        alpha=0.4,
        legend_label='Raw Scores',
    )

    # Plot smoothed trend with darker blue
    p.line(
        trials,
        smooth_overall,
        line_width=3,
        color='#0066CC',
        legend_label='Smoothed Overall Success',
    )

    # Add threshold and target lines with enhanced visibility
    perfect_line = Span(
        location=97,
        dimension='width',
        line_color='#008000',
        line_dash='dashed',
        line_width=3,
    )
    threshold_line = Span(
        location=65,
        dimension='width',
        line_color='#FF0000',
        line_dash='dashed',
        line_width=3,
    )

    # Add labels with backgrounds
    perfect_label = Label(
        x=50,
        y=97,
        text='Target: 97%',
        text_font_size='12pt',
        text_font_style='bold',
        background_fill_color='white',
        background_fill_alpha=0.7,
        text_color='#008000',
    )
    threshold_label = Label(
        x=50,
        y=65,
        text='Threshold: 65%',
        text_font_size='12pt',
        text_font_style='bold',
        background_fill_color='white',
        background_fill_alpha=0.7,
        text_color='#FF0000',
    )

    p.add_layout(perfect_line)
    p.add_layout(threshold_line)
    p.add_layout(perfect_label)
    p.add_layout(threshold_label)

    # Enhanced legend styling
    p.legend.label_text_font_size = '12pt'
    p.legend.background_fill_alpha = 0.7
    p.legend.border_line_width = 1
    p.legend.border_line_color = 'black'
    p.legend.click_policy = 'hide'
    p.legend.location = 'top_left'

    # Save the plot
    output_path = OUTPUT_DIR / 'maneuver_success_ratios.html'
    output_file(str(output_path), title='Maneuver Success Ratios')

    if download_option:
        p = add_download_button(p, 'overall_success_ratio.png')

    save(p)
    print(f"Overall Success Ratio plot saved to '{output_path}'.")

    # Also create a plot for individual component scores
    p_comp = figure(
        title='Component Success Scores Across Trials',
        x_axis_label='Trial Number',
        y_axis_label='Component Score (%)',
        tools='pan,wheel_zoom,box_zoom,reset,save',
        width=800,
        height=400,
    )

    colors = {
        'angle_error': '#1f77b4',
        'jerk': '#ff7f0e',
        'time': '#2ca02c',
        'roll': '#d62728',
        'pitch': '#9467bd',
        'yaw_rate': '#8c564b',
    }

    for component in [
        'angle_error',
        'jerk',
        'time',
        'roll',
        'pitch',
        'yaw_rate',
    ]:
        scores = [cs[component] * 100 for cs in component_scores]
        smooth_scores = smooth_data(scores, window_size=5)
        p_comp.line(
            trials,
            smooth_scores,
            line_width=2,
            color=colors[component],
            legend_label=component.replace('_', ' ').title(),
        )

    p_comp.legend.click_policy = 'hide'
    p_comp.legend.location = 'top_left'

    # Save the component scores plot
    output_path_comp = OUTPUT_DIR / 'component_success_scores.html'
    output_file(str(output_path_comp), title='Component Success Scores')

    if download_option:
        p_comp = add_download_button(p_comp, 'component_scores.png')

    save(p_comp)
    print(f"Component Success Scores plot saved to '{output_path_comp}'.")

    # Print summary statistics
    print('\nSuccess Ratio Summary:')
    print(f'Average Overall Success: {np.mean(overall_scores):.1f}%')
    print(f'Best Success Rate: {np.max(overall_scores):.1f}%')
    print(f'Worst Success Rate: {np.min(overall_scores):.1f}%')
    print(f'Success Rate Std Dev: {np.std(overall_scores):.1f}%')


# -----------------------------
# Helper to Create Individual Metric Plots
# -----------------------------
def create_metric_plot(
    title_text, y_axis_label, data, threshold, ideal=None, trials_sm=None
):
    """
    Create a plot with both raw values and smoothed trend.
    """
    p = figure(
        title=title_text,
        x_axis_label='Trial',
        y_axis_label=y_axis_label,
        tools='pan,wheel_zoom,box_zoom,reset,save',
        width=800,
        height=400,
    )

    # Plot raw data with low alpha
    p.scatter(
        trials_sm,
        data,
        size=3,
        color='gray',
        alpha=0.3,
        legend_label='Raw Data',
    )

    # Plot smoothed trend
    smoothed = smooth_data(data, window_size=10)
    p.line(
        trials_sm,
        smoothed,
        line_width=2,
        color='blue',
        legend_label='Smoothed Trend',
    )

    # Add threshold line with label
    p.add_layout(
        Span(
            location=threshold,
            dimension='width',
            line_color='red',
            line_dash='dashed',
            line_width=2,
        )
    )
    p.line(
        [0],
        [0],
        line_color='red',
        line_dash='dashed',
        legend_label=f'Threshold ({threshold})',
        line_width=2,
        visible=False,
    )

    # Add ideal line with label if provided
    if ideal is not None:
        p.add_layout(
            Span(
                location=ideal,
                dimension='width',
                line_color='green',
                line_dash='dashed',
                line_width=2,
            )
        )
        p.line(
            [0],
            [0],
            line_color='green',
            line_dash='dashed',
            legend_label=f'Target ({ideal})',
            line_width=2,
            visible=False,
        )

    p.legend.click_policy = 'hide'
    return p


# -----------------------------
# Plot Key Metrics
# -----------------------------
def plot_key_metrics(all_metrics, download_option=False):
    trials = list(range(1, len(all_metrics) + 1))

    def create_new_plot(title_text, y_axis_label, data, threshold, ideal=None):
        """Helper function to create a new plot with enhanced styling"""
        p = figure(
            title=title_text,
            x_axis_label='Trial',
            y_axis_label=y_axis_label,
            tools='pan,wheel_zoom,box_zoom,reset,save',
            width=800,
            height=400,
        )

        # Enhanced styling for the plot
        p.title.text_font_size = '16pt'
        p.title.text_font_style = 'bold'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'

        # Plot raw data with slightly higher alpha
        p.scatter(
            trials,
            data,
            size=4,
            color='gray',
            alpha=0.4,
            legend_label='Raw Data',
        )

        # Plot smoothed trend with darker blue
        smoothed = smooth_data(data, window_size=10)
        p.line(
            trials,
            smoothed,
            line_width=3,
            color='#0066CC',
            legend_label='Smoothed Trend',
        )

        # Add threshold and ideal lines with enhanced visibility
        threshold_line = Span(
            location=threshold,
            dimension='width',
            line_color='#FF0000',
            line_dash='dashed',
            line_width=3,
        )
        p.add_layout(threshold_line)

        # Add threshold label with background
        threshold_label = Label(
            x=50,
            y=threshold,
            text=f'Threshold: {threshold:.1f}',
            text_font_size='12pt',
            text_font_style='bold',
            background_fill_color='white',
            background_fill_alpha=0.7,
            text_color='#FF0000',
        )
        p.add_layout(threshold_label)

        if ideal is not None:
            ideal_line = Span(
                location=ideal,
                dimension='width',
                line_color='#008000',
                line_dash='dashed',
                line_width=3,
            )
            p.add_layout(ideal_line)

            # Add ideal label with background
            ideal_label = Label(
                x=50,
                y=ideal,
                text=f'Target: {ideal:.1f}',
                text_font_size='12pt',
                text_font_style='bold',
                background_fill_color='white',
                background_fill_alpha=0.7,
                text_color='#008000',
            )
            p.add_layout(ideal_label)

        # Enhanced legend styling
        p.legend.label_text_font_size = '12pt'
        p.legend.background_fill_alpha = 0.7
        p.legend.border_line_width = 1
        p.legend.border_line_color = 'black'
        p.legend.click_policy = 'hide'

        # Add grid lines for better readability
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3

        return p

    # Prepare data
    angle_errors = [m['angle_error_deg'] for m in all_metrics]
    jerks = [m['max_jerk'] for m in all_metrics]
    jerk_mean = np.mean(jerks)
    jerk_std = np.std(jerks)
    cleaned_jerks = [min(j, jerk_mean + 2 * jerk_std) for j in jerks]
    times = [m['time_taken'] for m in all_metrics]
    distances = [m['distance_traveled'] for m in all_metrics]

    # Create plots for the grid
    p1 = create_new_plot(
        'Angle Error',
        'Error (degrees)',
        angle_errors,
        threshold=7.0,
        ideal=2.0,
    )
    p2 = create_new_plot(
        'Maximum Jerk', 'Jerk (m/s³)', cleaned_jerks, threshold=1.5, ideal=0.8
    )
    p3 = create_new_plot(
        'Time Taken', 'Time (seconds)', times, threshold=5.0, ideal=3.5
    )
    p4 = create_new_plot(
        'Distance Traveled',
        'Distance (meters)',
        distances,
        threshold=12.0,
        ideal=9.5,
    )

    # Save the grid layout
    grid = gridplot([[p1], [p2], [p3, p4]])
    output_path = OUTPUT_DIR / 'key_metrics_overview.html'
    output_file(str(output_path), title='Key Metrics Overview')
    save(grid)
    print(f"Key Metrics Overview plot saved to '{output_path}'.")

    # Save individual plots with download buttons if requested
    if download_option:
        plot_data = [
            ('angle_error', angle_errors, 10.0, 5.0, 'Error (degrees)'),
            (
                'avg_jerk',
                [m['avg_jerk'] for m in all_metrics],
                2.0,
                1.0,
                'Average Jerk (m/s³)',
            ),
            ('time_taken', times, 5.0, 3.5, 'Time (seconds)'),
            ('distance_traveled', distances, 12.0, 9.5, 'Distance (meters)'),
        ]

        for name, data, threshold, ideal, y_label in plot_data:
            # Create a new plot for download
            download_plot = create_new_plot(
                name.replace('_', ' ').title(), y_label, data, threshold, ideal
            )

            # Create a new file for each download plot
            output_path = OUTPUT_DIR / f'{name}_with_button.html'
            output_file(str(output_path))
            plot_with_button = add_download_button(
                download_plot, f'{name}.png'
            )
            save(plot_with_button)
            print(f'Saved {name} plot with download button to {output_path}')

            # Export SVG version
            svg_path = OUTPUT_DIR / f'{name}.svg'
            export_svg(download_plot, filename=str(svg_path), webdriver=driver)
            print(f'Saved {name} SVG to {svg_path}')


# -----------------------------
# Plot Correlation Matrix
# -----------------------------
def plot_correlation_matrix(metrics_df, download_option=False):
    corr = metrics_df.corr()
    columns = list(corr.columns)
    x = []
    y = []
    values = []
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            x.append(col1)
            y.append(col2)
            values.append(corr.loc[col2, col1])

    mapper = LinearColorMapper(
        palette=Viridis256, low=min(values), high=max(values)
    )
    source_df = pd.DataFrame({'x': x, 'y': y, 'value': values})

    p = figure(
        title='Correlation Matrix',
        x_range=columns,
        y_range=list(reversed(columns)),
        tools='hover,save,pan,wheel_zoom,box_zoom,reset',
        width=600,
        height=600,
    )
    p.rect(
        x='x',
        y='y',
        width=1,
        height=1,
        source=source_df,
        fill_color={'field': 'value', 'transform': mapper},
        line_color=None,
    )

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
    p.add_layout(color_bar, 'right')
    p.xaxis.major_label_orientation = np.pi / 3

    output_path = OUTPUT_DIR / 'correlation_matrix.html'
    output_file(str(output_path), title='Correlation Matrix')
    if download_option:
        p = add_download_button(p, 'correlation_matrix.png')
    save(p)
    print(f"Correlation Matrix plot saved to '{output_path}'.")


# -----------------------------
# Display and Save Summary Statistics
# -----------------------------
def display_summary_statistics(metrics_df):
    stats = metrics_df.describe()
    stats_path = OUTPUT_DIR / 'summary_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write(stats.to_string())
    print('Summary Statistics:')
    print(stats)
    print(f"Summary statistics saved to '{stats_path}'.\n")
    return stats


# -----------------------------
# Main Processing Function
# -----------------------------
def main(test_mode=False, num_files=None, download_plots=False):
    file_paths = sorted(LOGS_DIR.glob('*.csv'))

    if test_mode:
        file_paths = file_paths[:1]
        print(f'Running in test mode. Processing only: {file_paths[0].name}\n')
    elif num_files is not None:
        file_paths = file_paths[:num_files]
        print(
            f'Processing the first {len(file_paths)} CSV files out of specified {num_files}.\n'
        )
    else:
        print(f'Found {len(file_paths)} CSV files. Processing all files.\n')

    if not file_paths:
        raise ValueError('No CSV files found in the specified directory.')

    all_metrics = []
    for idx, file_path in enumerate(file_paths, start=1):
        print(f'Processing file {idx}/{len(file_paths)}: {file_path.name}')
        try:
            df = pd.read_csv(file_path)
            required_columns = [
                'timestamp',
                'x',
                'y',
                'z',
                'vx',
                'vy',
                'vz',
                'ax',
                'ay',
                'az',
                'roll',
                'pitch',
                'yaw',
            ]
            if not all(col in df.columns for col in required_columns):
                print(
                    f'  Skipping {file_path.name}: Missing required columns.\n'
                )
                continue
            df = normalize_time(df)
            metrics = calculate_metrics(df)
            overall, comps = compute_success_ratio(metrics)
            metrics['overall_success'] = overall
            metrics.update(comps)
            all_metrics.append(metrics)
            print(
                f"  File '{file_path.name}' processed. Overall Success Ratio: {overall * 100:.2f}%\n"
            )
        except Exception as e:
            print(f"  Error processing '{file_path.name}': {e}\n")

    if not all_metrics:
        raise ValueError(
            'No valid metrics to process. Please check the input files.'
        )

    metrics_df = pd.DataFrame(all_metrics)
    aggregated_csv_path = LOGS_DIR / 'aggregated_success_metrics.csv'
    metrics_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated success metrics saved to '{aggregated_csv_path}'.\n")

    display_summary_statistics(metrics_df)
    plot_success_ratios(all_metrics, download_option=download_plots)
    plot_key_metrics(all_metrics, download_option=download_plots)
    plot_correlation_matrix(metrics_df, download_option=download_plots)


if __name__ == '__main__':
    # Examples:
    #   Process all CSVs: main(test_mode=False, num_files=None, download_plots=True)
    #   Process first 600 CSVs: main(test_mode=False, num_files=600, download_plots=True)
    #   Run in test mode (single CSV): main(test_mode=True, num_files=1, download_plots=True)
    main(test_mode=False, num_files=600, download_plots=True)
