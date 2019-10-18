import numpy as np

from . import graphics
from . import constants as ck


def coord_to_pt(x, y):
	"""
	Takes input (x,y) co-ordinates and converts it to a point
	in the grid.
	"""
	assert ck.x_min <= x and x <= ck.x_max, x
	assert ck.y_min <= y and y <= ck.y_max, y
	
	# Normalize x to [0, inf]
	x = x - ck.x_min
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


class Window:
	def __init__(self):
		self.win = None
		self.cars = []

	def setup(self, force=False):
		if ck.visualize or force:
			self.win = graphics.GraphWin('Pass', 500, 500)
			self.win.setCoords(0,0,1.2,1)

	def destroy(self, force=False):
		if ck.visualize or force:
			self.win.close()

	def draw(self, state):
		self.draw_grid(state)
		self.draw_cars(state)

	def draw_cars(self, state):
		# Draw my car.
		self.draw_car(state.x_car, state.y_car, state.length_car, state.width_car)

		# Draw other cars.
		for i in range(len(state.x_others)):
			if state.vx_others[i] >= 0:
				color = "green"
			else:
				color = "red"
			self.draw_car(state.x_others[i], state.y_others[i], state.length_others[i], 
						  state.width_others[i], color, color)

	def draw_grid(self, state):

		# Draw the grid points.
		for x in ck.X_points:
			for y in ck.Y_points:
				pt = coord_to_pt(x,y) 
				pt.draw(self.win)

			pt = coord_to_pt(x, ck.Y_points[0])
			pt.move(0.1,0)
			text = graphics.Text(pt, "%.4f m" % x)
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

	def clear_cars(self):
		for car in self.cars:
			car.undraw()
			del car
		self.cars.clear()
	
	def draw_car(self, xl, yl, length, breadth, color="blue", border="blue"):
		ll = coord_to_pt(xl, yl)
		lh = coord_to_pt(xl, yl+breadth)
		hh = coord_to_pt(xl+length, yl+breadth)
		hl = coord_to_pt(xl+length, yl)
		car = graphics.Polygon(ll, lh, hh, hl)
		car.setFill(color)
		car.setOutline(border)
		car.draw(self.win)
		self.cars.append(car)