"""
Constants like size of road, number of grid points, etc. is defined here.

Unit is in metres.

 Y
 /\
 |
 |
 |    ===================================================================
 |                                                ---------------
 |                                                |    CAR C    |
 |                                                ---------------
 |    -------------------------------------------------------------------
 |        ---------------               ---------------
 |        |    CAR A    |               |    CAR B    |
 |        ---------------               ---------------
 |    ===================================================================
 |
 |
 --------------------------------------------------------------------------------------> X

"""
import numpy as np


# Size/Width
visibility_distance = 1000
x_min = 0
x_max = visibility_distance
lane_width = 3.2
y_min = 0
y_max = lane_width*2

car_length = 5
car_width = 2

n_x_points = 150
n_y_points = 10
# TODO. How to balance grid accuracy, vehicle movement etc.
# The X points are spaced by tan(dx). Assuming height of 25m, might be too large
# X_points = np.tan(np.linspace(0, np.arctan(x_max/20), n_x_points))*20
X_points = np.linspace(x_min, x_max, n_x_points)
# TODO. Current formula considering x_min = 0
# X_points = np.cumsum(np.linspace(x_min, 20, 20))
Y_points = np.linspace(y_min, y_max, n_y_points)
grid_y, grid_x = np.meshgrid(Y_points, X_points)
# print(X_points)

# Velocity, Acceleration
vx_max = 30  # metres/sec
vx_min = 0  # TODO. Magnitude.
vy_max = 1  # TODO
vy_min = -1

ax_max = 5  # metres/sec/sec
ax_min = -5
ay_max = 1
ay_min = -1

# Minimum following distance. Measured in sec.
t_min = 3

# Incoming Vehicles. Poisson Distribution Parameter
poisson = 0.01

# Visualize
visualize = True