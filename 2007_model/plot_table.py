import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_lines_and_regression(data):
    """
    Given a table with four columns:
      - 'A_x' and 'A_y' represent point A.
      - 'B_x' and 'B_y' represent point B.
    This function:
      1. Draws a line connecting each pair (A to B).
      2. Plots point A in blue and point B in red.
      3. Computes and plots a linear regression line through the B points.
    """
    fig, ax = plt.subplots()
    
    # Loop over each row: plot line and endpoints.
    for i, row in data.iterrows():
        A = (row['A_x'], row['A_y'])
        B = (row['B_x'], row['B_y'])
        # Draw a line between A and B.
        ax.plot([A[0], B[0]], [A[1], B[1]], color='gray', linestyle='--', linewidth=0.8)
        # Plot point A in blue.
        ax.plot(A[0], A[1], marker='o', color='blue', markersize=6)
        # Plot point B in red.
        ax.plot(B[0], B[1], marker='o', color='red', markersize=6)
    
    # Extract B points for regression.
    B_x = data['B_x'].values
    B_y = data['B_y'].values

    # Compute linear regression (least squares) using numpy.polyfit.
    # Fit a line: B_y = m * B_x + c
    m, c = np.polyfit(B_x, B_y, 1)
    
    # Prepare x values for plotting the regression line.
    x_vals = np.linspace(B_x.min(), B_x.max(), 100)
    y_vals = m * x_vals + c
    
    # Plot the regression line.
    ax.plot(x_vals, y_vals, color='purple', linewidth=2, label=f'Regression: y={m:.2f}x+{c:.2f}')
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Lines between points A and B with B regression line")
    ax.legend()
    plt.show()

# Example data.
data = pd.DataFrame({
    'B_x': [0, 1, 2, 3, 4, 5, -3, -2, -3, 0, 2, 3, 4, 5],
    'B_y': [-1, 1, 3, 5, 7, 9, -7, -5, -1, -1, 3, 5, 7, 3],
    'A_x': [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 7.5, 6.5, 5.5, 4.5, 2.5, 1.5, 0.5, 0.5],
    'A_y': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.5]
})

plot_lines_and_regression(data)
