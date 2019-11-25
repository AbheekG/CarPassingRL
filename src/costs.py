import numpy as np

from . import constants as ck


# velocity cost
def cost_vel(state):
	"""
	Low cost for high velocity.
	"""
	return min(ck.vx_max - state.our_car.vx, 0)


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
	# Invalid position.
	if state.our_car.x < ck.x_min or state.our_car.x > ck.x_max:
		return 1000
	elif state.our_car.y < ck.y_min or state.our_car.y > ck.y_max:
		return 1000
	elif state.our_car.vx < ck.vx_min or state.our_car.vx > ck.vx_max:
		return 1000
	elif state.our_car.vy < ck.vy_min or state.our_car.vy > ck.vy_max:
		return 1000
		
	return abs(state.our_car.y)


def _car_collide(our_car, car):
	if abs(our_car.y - car.y) < ck.y_gap/2: # In same lane
		if our_car.vx - car.vx >= abs(car.x - our_car.x):
			return True
	return False


# collision cost
def cost_collision(state):
	for car in (state.forward_cars + state.backward_cars):
		if _car_collide(state.our_car, car):
			return 1.0
	return 0.0
