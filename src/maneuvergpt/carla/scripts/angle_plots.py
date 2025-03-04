import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io.export import export_svg
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Legend, Span, Label, HoverTool
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, show, save, output_file
from scipy.signal import savgol_filter
from selenium import webdriver
from selenium.webdriver.firefox.service import Service

geckodriver_path = "/snap/bin/geckodriver"

LOGS_DIR = pathlib.Path(__file__).parent / 'logs'

service = Service(geckodriver_path)

# Create the Firefox driver using the service
driver = webdriver.Firefox(service=service)


def load_and_process_data(directory: Path) -> list:
    """Load all CSV files from directory and extract angle errors"""
    angle_errors = []
    for csv_file in directory.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'angle_error_deg' in df.columns:
                angle_errors.append(df['angle_error_deg'].iloc[0])
            else:
                # Calculate angle error if not directly available
                initial_yaw = df['pitch'].iloc[0]
                final_yaw = df['pitch'].iloc[-1]
                angle_diff = abs(final_yaw - initial_yaw)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                angle_error = abs(angle_diff - 180)
                angle_errors.append(angle_error)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    return angle_errors


def create_comparison_plot(single_errors: list, multi_errors: list,
                           output_path: str):
    """Create an enhanced Bokeh plot comparing single and multi-agent angle errors"""
    # Create trial numbers
    single_trials = list(range(1, len(single_errors) + 1))
    multi_trials = list(range(1, len(multi_errors) + 1))

    # Create smoothed versions of the data
    # Use savgol_filter for smooth curves
    window_length = min(15, len(single_errors) - 1 if len(
        single_errors) % 2 == 0 else len(single_errors) - 2)
    if window_length < 3:
        window_length = 3

    single_smooth = savgol_filter(single_errors, window_length, 3)
    multi_smooth = savgol_filter(multi_errors, window_length, 3)

    # Create the figure with y-axis range limit
    p = figure(
        title="Angle Error Comparison: Single vs Multi-Agent System",
        x_axis_label="Trial Number",
        y_axis_label="Angle Error (degrees)",
        width=1200,
        height=800,
        y_range=(0, 40),  # Set fixed y-axis range
        tools="pan,box_zoom,wheel_zoom,reset,save",
        background_fill_color="#FFFFFF",
        border_fill_color="#FFFFFF"
    )

    # Enhanced styling
    p.title.text_font_size = '24pt'  # Larger title
    p.title.text_font_style = 'bold'
    p.title.text_color = '#000000'

    p.axis.axis_label_text_font_size = '22pt'
    p.axis.axis_label_text_font_style = 'bold'
    p.axis.major_label_text_font_size = '22pt'
    p.axis.axis_line_color = '#000000'
    p.axis.axis_line_width = 2

    # Create data sources with hover information
    single_source = ColumnDataSource({
        'x': single_trials,
        'y': single_errors,
        'y_smooth': single_smooth,
        'error': [f'{e:.2f}°' for e in single_errors],
        'type': ['Single Agent'] * len(single_errors)
    })

    multi_source = ColumnDataSource({
        'x': multi_trials,
        'y': multi_errors,
        'y_smooth': multi_smooth,
        'error': [f'{e:.2f}°' for e in multi_errors],
        'type': ['Multi Agent'] * len(multi_errors)
    })

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ('System', '@type'),
            ('Trial', '@x'),
            ('Error', '@error'),
        ],
        mode='mouse'
    )
    p.add_tools(hover)

    # Add smoothed lines with darker colors
    p.line('x', 'y_smooth', source=single_source,
           line_color='#CC0000',  # Darker red
           line_width=3,
           alpha=0.8,
           legend_label="Single Agent (Smoothed)")

    p.line('x', 'y_smooth', source=multi_source,
           line_color='#0000CC',  # Darker blue
           line_width=3,
           alpha=0.8,
           legend_label="Multi Agent (Smoothed)")

    # Add raw data points with matching darker colors
    p.scatter('x', 'y', source=single_source,
              size=6,
              color='#CC0000',  # Darker red
              alpha=0.4,
              legend_label="Single Agent (Raw)",
              marker='circle')

    p.scatter('x', 'y', source=multi_source,
              size=6,
              color='#0000CC',  # Darker blue
              alpha=0.4,
              legend_label="Multi Agent (Raw)",
              marker='triangle')

    # Add threshold lines with darker colors
    optimal_line = Span(location=3.0, dimension='width',
                        line_color='#006400',  # Dark green
                        line_dash='dashed',
                        line_width=3)
    threshold_line = Span(location=7.0, dimension='width',
                          line_color='#800080',  # Dark purple
                          line_dash='dashed',
                          line_width=3)

    p.add_layout(optimal_line)
    p.add_layout(threshold_line)

    # Add threshold labels with larger boxes and text
    for y, text, color in [(3.0, 'Optimal (3°)', '#006400'),  # Dark green
                           (7.0, 'Threshold (7°)', '#800080')]:  # Dark purple
        label = Label(
            x=50, y=y,
            text=text,
            text_font_size='22pt',  # Increased from 14pt
            text_font_style='bold',
            text_color=color,
            background_fill_color='white',
            background_fill_alpha=0.7,
            border_line_color=color,
            border_line_alpha=0.7
        )
        p.add_layout(label)

    # Calculate statistics
    single_stats = {
        'mean': np.mean(single_errors),
        'std': np.std(single_errors),
        'max': np.max(single_errors),
        'min': np.min(single_errors),
        'median': np.median(single_errors)
    }

    multi_stats = {
        'mean': np.mean(multi_errors),
        'std': np.std(multi_errors),
        'max': np.max(multi_errors),
        'min': np.min(multi_errors),
        'median': np.median(multi_errors)
    }

    # Add statistics box with larger size
    stats_label = Label(
        x=50,
        y=700,  # Fixed y position
        text=f"""
        Single Agent Stats:
        Mean: {single_stats['mean']:.2f}° | Median: {single_stats['median']:.2f}°
        Std: {single_stats['std']:.2f}° | Range: [{single_stats['min']:.2f}° - {single_stats['max']:.2f}°]

        Multi Agent Stats:
        Mean: {multi_stats['mean']:.2f}° | Median: {multi_stats['median']:.2f}°
        Std: {multi_stats['std']:.2f}° | Range: [{multi_stats['min']:.2f}° - {multi_stats['max']:.2f}°]
        """,
        text_font_size='22pt',  # Increased from 12pt
        text_font_style='bold',
        background_fill_color='white',
        background_fill_alpha=0.7,
        border_line_color='black',
        border_line_alpha=0.7
    )
    p.add_layout(stats_label)

    # Update legend to reflect smoothing
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    p.legend.background_fill_color = "white"
    p.legend.background_fill_alpha = 0.8
    p.legend.border_line_width = 2
    p.legend.border_line_color = "black"
    p.legend.label_text_font_size = '22pt'
    p.legend.label_text_font_style = 'bold'

    # Add smoothing note with larger text
    smoothing_note = Label(
        x=50,
        y=650,  # Position below stats
        text=f"Note: Lines smoothed using Savitzky-Golay filter (window: {window_length}, polynomial order: 3)",
        text_font_size='22pt',  # Increased from 10pt
        text_font_style='italic',
        background_fill_color='white',
        background_fill_alpha=0.7
    )
    p.add_layout(smoothing_note)

    # Add grid with better visibility
    p.grid.grid_line_color = 'gray'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1

    # Save the plot
    output_file(output_path)
    save(p)
    print(f"Enhanced plot saved to {output_path}")
    svg_path = LOGS_DIR / 'angle_error_comparison.svg'
    export_svg(p, filename=str(svg_path), webdriver=driver)
    print(f"Saved angle_error_comparison.svg to {svg_path}")


def main():
    # Define paths
    single_dir = LOGS_DIR / 'single'
    multi_dir = LOGS_DIR / 'multi'
    output_path = LOGS_DIR / 'angle_error_comparison.html'

    # Load data
    print("Loading single agent data...")
    single_errors = load_and_process_data(single_dir)
    print(f"Loaded {len(single_errors)} single agent trials")

    print("\nLoading multi agent data...")
    multi_errors = load_and_process_data(multi_dir)
    print(f"Loaded {len(multi_errors)} multi agent trials")

    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(single_errors, multi_errors, output_path)


if __name__ == "__main__":
    main()
