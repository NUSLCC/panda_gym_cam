import numpy as np
import matplotlib.pyplot as plt

# Define corner points
corners = np.array([[0, 0], [1, 1], [2, -1], [3, 1], [4, 0]])

# Number of points between corners for interpolation
num_interp_points = 100

# Interpolate points between corners
interpolated_points = np.empty((0, 2))
for i in range(len(corners) - 1):
    start_point = corners[i]
    end_point = corners[i + 1]
    interpolated_segment = np.linspace(start_point, end_point, num_interp_points, endpoint=False)
    interpolated_points = np.vstack((interpolated_points, interpolated_segment))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], label='Zigzag Straight Line')
plt.scatter(corners[:, 0], corners[:, 1], color='red', label='Corner Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Zigzag Straight Line between Corner Points')
plt.legend()
plt.grid(True)
plt.show()
