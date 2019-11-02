import numpy as np

from . import constants as ck


# velocity cost
def cost_vel(state):
	"""
	Low cost for high velocity.
	"""
	return min(state.our_car.vx_max - state.our_car.vx, 0)


# acceleration cost
def cost_acc(state):
	"""
	High cost for high magnitude for acceleration.
	"""
	return abs(state.our_car.ax) + abs(state.our_car.ay)

# position cost
def cost_lane(state):
	"""
	Cost for not moving in right lane.
	"""
	return abs( state.our_car.y + state.our_car.w/2 - 
		(state.our_car.y_min + (state.our_car.y_max-state.our_car.y_min)*0.25) )

def _point_in_rect(x, y, l, b, px, py):
	return (x <= px and px <= x + l and y <= py and py <= y + b)

def _rect_in_rect(rect1, rect2):
	"""Check whether rect 2 is in rect1."""
	
	return (_point_in_rect(rect1.x, rect1.y, rect1.l, rect1.w, rect2.x, rect2.y) or
		_point_in_rect(rect1.x, rect1.y, rect1.l, rect1.w, rect2.x + rect2.l, rect2.y) or
		_point_in_rect(rect1.x, rect1.y, rect1.l, rect1.w, rect2.x, rect2.y + rect2.w) or
		_point_in_rect(rect1.x, rect1.y, rect1.l, rect1.w, rect2.x + rect2.l, rect2.y + rect2.w)
	)

# collision cost
def cost_collision(state):
	for car in (state.forward_cars + state.backward_cars):
		if _rect_in_rect(state.our_car, car) or _rect_in_rect(car, state.our_car):
			return 1.0
	return 0.0
