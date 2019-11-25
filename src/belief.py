import numpy as np
import scipy.stats as stats

from . import constants as ck
from . import draw


class Belief:
	def __init__(self):
		# Probability of another car or object being present at the grid location.
		self.prob = np.ones((ck.n_x_points, ck.n_y_points)) / ck.n_x_points / ck.n_y_points

	@staticmethod
	def _get_seeing_prob(x, y, our_car):
		"""
		Gives the probability of seeing a vehicle.
		For our lane, we see the vehicle with 
		"""
		dist_x = np.abs(x - our_car.x)

		if abs(y - our_car.y) < ck.y_gap/2:
			return 1 / (1 + dist_x)
		else:
			return 0.5 / (1 + dist_x)		

	@staticmethod
	def _sensor_measurement(our_car, forward_cars, backward_cars):
		"""
		The new sensor measurements.
		"""
		prob = np.zeros((ck.n_x_points, ck.n_y_points))

		# Right lane
		if our_car.y < ck.y_gap/2:
			if len(forward_cars) > 1: assert forward_cars[-1].x < forward_cars[-2].x
			if forward_cars:
				car = forward_cars[-1]
				if True: # np.random.uniform() < Belief._get_seeing_prob(car.x, car.y, our_car):
					prob[min(int(car.x / ck.x_gap), prob.shape[0]-1)][int(car.y / ck.y_gap)] = 1
			for car in backward_cars:
				# print("Backward", int(car.x / ck.x_gap), int(car.y / ck.y_gap), Belief._get_seeing_prob(car.x, car.y, our_car))
				if True: # np.random.uniform() < Belief._get_seeing_prob(car.x, car.y, our_car):
					prob[min(int(car.x / ck.x_gap), prob.shape[0]-1)][int(car.y / ck.y_gap)] = 1
		
		# draw.plot3d(ck.grid_x, ck.grid_y, prob)
		
		n_cars = prob.sum()
		# print(n_cars)
		prob += 1e-3
		prob = prob / prob.sum() * n_cars
		
		return prob

	def _sensor_update(self, our_car, cars):
		prob_new = Belief._sensor_measurement(our_car, cars)

		# TODO: Check whether the following trick is sound
		# w_new = stats.norm(0, v_sig_new).pdf(0)
		# w_old = stats.norm(0, self.v_sig).pdf(0)
		# w_sum = w_old + w_new
		# w_new /= w_sum
		# w_old /= w_sum
		"""
		Making assumption that total probability sums up to number of cars.
		"""
		n_cars = prob_new.sum()
		prob = self.prob * prob_new
		prob = prob / prob.sum() * n_cars

		# Adding some prob to all points to avoid 0.
		prob += 1e-3
		prob = prob / prob.sum() * n_cars

		return prob

	# @staticmethod
	# def _grid_to_index(grid_x, grid_y):
	# 	"""
	# 	Rounds the values to nearest index of X and Y.
	# 	"""
	# 	# np.clip(grid_x,ck.x_min,ck.x_max) changed to grid_x.
	# 	# Because it was accumulating mass at the boundary.
	# 	idx_x = np.mod(np.round((grid_x - ck.x_min) / ck.x_gap).astype(int), ck.n_x_points)
	# 	idx_y = np.mod(np.round((grid_y - ck.y_min) / ck.y_gap).astype(int), ck.n_y_points)

	# 	return idx_x, idx_y

	# def _action_update(self, our_car, dt):
	# 	prob = np.zeros((ck.n_x_points, ck.n_y_points))
	# 	vx_mu = np.zeros((ck.n_x_points, ck.n_y_points))
	# 	vy_mu = np.zeros((ck.n_x_points, ck.n_y_points))

	# 	# The new index point. Move the grid by velocity.
	# 	idx_x, idx_y = self._grid_to_index(ck.grid_x + dt*self.vx_mu, ck.grid_y + dt*self.vy_mu)
	# 	# print(idx_x, idx_y)
	# 	np.add.at(prob, (idx_x, idx_y), self.prob)
	# 	np.add.at(vx_mu, (idx_x, idx_y), self.vx_mu)
	# 	np.add.at(vy_mu, (idx_x, idx_y), self.vy_mu)

	# 	# TODO: Gaussian blur with different sigma for different X and Y.
	# 	# Only doing for the nearby 3x3 points.
	# 	# _, sig_x, sig_y = self._get_sig(grid_x, grid_y, our_car)

	# 	# Adding some prob to all points to avoid 0.
	# 	n_car = prob.sum()
	# 	prob += ck.prob_unif_blur
	# 	prob = prob / prob.sum() * n_car

	# 	return prob, vx_mu, vy_mu

	def step(self, our_car, forward_cars, backward_cars):
		"""
		Update the belief.
		"""
		# self.prob = self._action_update(our_car)
		self.prob = self._sensor_measurement(our_car, forward_cars, backward_cars)
		# self.prob, self.vx_mu, self.vy_mu = self._sensor_measurement(our_car, cars)