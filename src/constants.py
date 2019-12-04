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


x_max = 100
x_min = 0
y_max = 1
y_min = 0

n_x_points = 101
n_y_points = 2

assert (x_max - x_min) % (n_x_points - 1) == 0
X_points = np.arange(n_x_points)
Y_points = np.array([0,1])
x_gap = X_points[1] - X_points[0]
y_gap = Y_points[1] - Y_points[0]
grid_y, grid_x = np.meshgrid(Y_points, X_points)

# Velocity, Acceleration
vx_max = 2
vx_min = 0  # TODO. Magnitude.
vy_max = 1  # TODO
vy_min = -1

ax_max = 1  # metres/sec/sec
ax_min = -1
ay_max = 1
ay_min = -1
n_x_actions = int(ax_max-ax_min) + 1
assert n_x_actions == 3
n_y_actions = int(ay_max-ay_min) + 1
assert n_y_actions == 3
X_actions = np.linspace(ax_min, ax_max, n_x_actions)
Y_actions = np.linspace(ay_min, ay_max, n_y_actions)

# Minimum following distance. Measured in sec/5.
t_min = 1

# Incoming Vehicles. Poisson Distribution Parameter
poisson = 0.05

# Visualize
visualize = True

# Belief
prob_unif_blur = 3e-3

# Cost weights
vel_weight = 1
acc_weight = 1
lane_weight = 1
collision_weight = 100
discount = 0.999
exploration = 0.01

# Training.
batch = 32  # Batching while training.