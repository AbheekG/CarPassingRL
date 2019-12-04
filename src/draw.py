from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from . import graphics
from . import constants as ck


scale = 5

def plot3d(X, Y, Z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()


def coord_to_pt(x, y):
	"""
	Takes input (x,y) co-ordinates and converts it to a point
	in the grid.
	"""
	# assert ck.x_min <= x and x <= ck.x_max, x
	# assert ck.y_min <= y and y <= ck.y_max, y
	
	# Normalize x to [0, inf]
	x = (x - ck.x_min)*scale
	# Normalize y to [0, 1]
	y = 1 - (y - ck.y_min) / (ck.y_max - ck.y_min)
	return pp1(x,y)


def pp1(x,y):
	"""
	One point perspetive plot of x and y.
	We have to change axis: x->z, y->x, z->y
	"""
	# Changing axis. x->z, y->x, z->y
	x_orig = y
	y_orig = 0
	z_orig = x

	# Now we work with new co-ordinate system.
	x0, y0 = 0.5, 0.9  # Perspective points on picture plane.
	d = 25  # -d is the point of viewing on Z-axis.

	x_norm = x0 + (x_orig - x0) * d / (z_orig + d)
	y_norm = y0 + (y_orig - y0) * d / (z_orig + d)
	return graphics.Point(x_norm, y_norm)


def color_bw(x):
	"""x is in [0,1]."""
	x = int(np.floor(x*256))
	return graphics.color_rgb(x,x,x)


class Window:
	def __init__(self):
		self.win = None
		self.cars = []
		self.belief = []
		# self.move_x = 1.1  # Move X to right for belief drawing

	def setup(self, force=False):
		if ck.visualize or force:
			self.win = graphics.GraphWin('Pass', 600, 500)
			self.win.setCoords(0,0,1.2,1)

	def destroy(self, force=False):
		if ck.visualize or force:
			self.win.close()

	def draw(self, state):
		self.draw_grid(state, False)
		self.draw_cars(state)
		# self.draw_belief(state)

	def draw_cars(self, state):
		# Draw my car.
		self.draw_car(state.our_car, "blue", "blue")

		# Draw other cars.
		for car in state.forward_cars:
			assert car.vx >= 0
			color = "green"
			self.draw_car(car, color, color)

		for car in state.backward_cars:
			assert car.vx <= 0
			color = "red"
			self.draw_car(car, color, color)

	def draw_grid(self, state, belief=False):
		# Draw the grid points.
		for i, x in enumerate(ck.X_points):
			for j, y in enumerate(ck.Y_points):
				pt = coord_to_pt(x,y)
				if belief:
					pt.move(self.move_x, 0)
					pt.setFill(color_bw(state.belief.prob[i][j]))
					self.belief.append(pt)
				pt.draw(self.win)

			if not belief:
				pt = coord_to_pt(x, ck.Y_points[0])
				pt.move(0.1,0)
				text = graphics.Text(pt, "%.2f m" % (x*scale))
				text.setSize(5)
				text.draw(self.win)

		# Draw road boundaries.
		# Right line
		linr = graphics.Line(coord_to_pt(ck.x_min, ck.y_min), coord_to_pt(ck.x_max, ck.y_min))
		# Left line
		linl = graphics.Line(coord_to_pt(ck.x_min, ck.y_max), coord_to_pt(ck.x_max, ck.y_max))
		# Center line
		y_mid = (ck.y_min + ck.y_max)/2
		linc = graphics.Line(coord_to_pt(ck.x_min, y_mid), coord_to_pt(ck.x_max, y_mid))
		# linc.setFill("yellow")
		for lin in [linr, linl, linc]:
			if belief:
				lin.move(self.move_x, 0)
			lin.draw(self.win)

		# # Random line test
		# XX = np.linspace(0, 1000, 100)
		# YY = np.linspace(0, 5, 100)
		# # YY = np.linspace(0, 5, len(XX))
		# for i in range(len(XX)):
		# 	x_norm, y_norm = coord_to_pt(XX[i], YY[i])
		# 	pt = graphics.Point(x_norm, y_norm)
		# 	pt.setFill("red")
		# 	pt.draw(self.win)

	def draw_belief(self, state):
		# self.draw_grid(state, belief=True)
		# Plot
		grid_y, grid_x = np.meshgrid(ck.Y_points, ck.X_points)
		plot3d(grid_x, grid_y, state.belief.prob)
		
	def clear_cars(self):
		for car in self.cars:
			car.undraw()
			del car
		self.cars.clear()

	def clear_belief(self):
		# TODO
		plt.close()
		# for b in self.belief:
		# 	b.undraw()
		# 	del b
		# self.belief.clear()

	def draw_car(self, car, color="blue", border="blue"):
		xl = car.x
		yl = car.y + (ck.y_min + ck.y_max)/12
		# TODO. Hack done here.
		if yl > (ck.y_min + ck.y_max)/2:
			yl = 7*(ck.y_min + ck.y_max)/12
		length = ck.x_gap
		breadth = ck.y_gap/3
		ll = coord_to_pt(xl, yl)
		lh = coord_to_pt(xl, yl+breadth)
		hh = coord_to_pt(xl+length, yl+breadth)
		hl = coord_to_pt(xl+length, yl)
		car = graphics.Polygon(ll, lh, hh, hl)
		car.setFill(color)
		car.setOutline(border)
		car.draw(self.win)
		self.cars.append(car)
