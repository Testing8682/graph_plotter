import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Function to plot the graph
def plot_graph(coordinates, line_names):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('white')

    max_x = 0
    max_y = 0

    for i, line_data in enumerate(coordinates):
        x_coords = line_data['X']
        y_coords = line_data['Y']
        x_tols = line_data['X Tol']
        y_tols = line_data['Y Tol']

        max_x = max(max_x, max(x_coords))
        max_y = max(max_y, max(y_coords))

        # Plot the main line
        plt.plot(x_coords, y_coords, color='black', linewidth=1)

        if i == 0:
            plt.plot([x_coords[0], x_coords[0]], [0, y_coords[0]], color='black', linewidth=1)

        # Define spacing for label padding
        x_base_offset = -0.05 * max_y
        x_spacing = 0.03 * max_y
        y_base_offset = -5
        y_spacing = 4

        xticks_set = set(np.round(plt.xticks()[0], 1))
        yticks_set = set(np.round(plt.yticks()[0], 1))

        for j, (x, y, x_tol, y_tol) in enumerate(zip(x_coords, y_coords, x_tols, y_tols)):
            # Grid lines
            line_1, = plt.plot([x, x], [0, y], 'k--', linestyle='--', linewidth=0.8)
            line_1.set_dashes([3, 2, 3, 2, 10, 2])
            line_2, = plt.plot([0, x], [y, y], 'k--', linestyle='--', linewidth=0.8)
            line_2.set_dashes([3, 2, 3, 2, 10, 2])

            # X tolerance label (if not already on x-axis)
            if x_tol > 0 and round(x, 1) not in xticks_set:
                x_label_text = f'{x:.1f}±{x_tol:.1f}'
                x_y_position = x_base_offset - j * x_spacing
                plt.text(x, x_y_position, x_label_text,
                         ha='center', va='top', fontsize=9, color='black', rotation=90)

            # Y tolerance label (if not already on y-axis)
            if y_tol > 0 and round(y, 1) not in yticks_set:
                y_label_text = f'{y:.1f}±{y_tol:.1f}'
                y_x_position = y_base_offset - j * y_spacing
                plt.text(y_x_position, y, y_label_text,
                         ha='right', va='center', fontsize=9, color='black')

        # Line label (angled above midpoint)
        dx = x_coords[-1] - x_coords[0]
        dy = y_coords[-1] - y_coords[0]
        angle = np.degrees(np.arctan2(dy, dx))
        mid_x = np.mean(x_coords)
        mid_y = np.mean(y_coords)
        length = np.hypot(dx, dy)

        if length != 0:
            normal_x = -dy / length
            normal_y = dx / length
            offset = 0.03 * max_y
            offset_x = normal_x * offset
            offset_y = normal_y * offset
        else:
            offset_x = offset_y = 0

        plt.text(mid_x + offset_x, mid_y + offset_y, line_names[i],
                 ha='center', va='center', fontsize=12, color='black',
                 rotation=angle - 2, rotation_mode='anchor')

    # Axes & Grid Styling
    plt.xlabel('INPUT FORCE --> Kg', labelpad=70, fontsize=15)
    plt.ylabel('OUTPUT PRESSURE Bar --> bar', labelpad=90, fontsize=15)
    plt.title('PERFORMANCE CHARACTERISTICS\nTOLERANCE BAND TO BE FINALISED AFTER DATA\nGENERATION ON LARGE NO. OF SAMPLES',
              pad=20, fontsize=10)

    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xticks(np.arange(0, max_x + 11, 10))
    plt.yticks(np.arange(0, max_y + 11, 10))
    plt.grid()

    st.pyplot(plt)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf

# Streamlit interface
st.title('Graph Plotter')

num_lines = st.number_input('How many lines are there in the graph?', min_value=1, step=1)

line_names = []
coordinates = []

for i in range(num_lines):
    line_name = st.text_input(f'Enter the name for line {i+1}:')
    line_names.append(line_name)

    num_coords = st.number_input(f'How many coordinates are there for line {i+1}?', min_value=1, step=1)

    df = pd.DataFrame({
        'X': [0.0] * num_coords,
        'Y': [0.0] * num_coords,
        'X Tol': [0.0] * num_coords,
        'Y Tol': [0.0] * num_coords
    })

    st.write(f'Enter coordinates and tolerance for line {i+1}:')
    edited_df = st.data_editor(df, num_rows='dynamic', key=f'df_{i}')

    line_data = {
        'X': edited_df['X'].tolist(),
        'Y': edited_df['Y'].tolist(),
        'X Tol': edited_df['X Tol'].tolist(),
        'Y Tol': edited_df['Y Tol'].tolist()
    }
    coordinates.append(line_data)

if st.button('Generate Graph'):
    buf = plot_graph(coordinates, line_names)
    st.download_button(
        label='Download Graph',
        data=buf,
        file_name='graph_with_tolerance.png',
        mime='image/png'
    )
