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

def _point_in_rect(x_min, y_min, x_max, y_max, px, py):
	return (x_min <= px and px <= x_max and y_min <= py and py <= y_max)

def _get_boundary_points(car, dt):
	def _get_min_max(x, l, vxdt):
		x_min = x
		x_min = min(x_min, x_min + vxdt)
		x_max = x + l
		x_max = max(x_max, x_max + vxdt)
		return x_min, x_max

	x_min, x_max = _get_min_max(car.x, car.l, car.vx*dt)
	y_min, y_max = _get_min_max(car.y, car.w, car.vy*dt)

	return x_min, x_max, y_min, y_max

def _car_collide(car1, car2, dt):
	"""Check whether rect 2 is in rect1, also consider the velocity"""
	x1_min, x1_max, y1_min, y1_max = _get_boundary_points(car1, dt)
	x2_min, x2_max, y2_min, y2_max = _get_boundary_points(car2, dt)
	
	return (_point_in_rect(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min) or
		_point_in_rect(x1_min, y1_min, x1_max, y1_max, x2_min, y2_max) or
		_point_in_rect(x1_min, y1_min, x1_max, y1_max, x2_max, y2_min) or
		_point_in_rect(x1_min, y1_min, x1_max, y1_max, x2_max, y2_max)
	)

# collision cost
def cost_collision(state, dt):
	for car in (state.forward_cars + state.backward_cars):
		if _car_collide(state.our_car, car, dt) or _car_collide(car, state.our_car, dt):
			return 1.0
	return 0.0
