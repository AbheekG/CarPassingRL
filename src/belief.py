import numpy as np
import scipy.stats as stats

from . import constants as ck
from . import draw


class Belief:
	def __init__(self):
		# Probability of another car or object being present at the grid location.
		self.prob = np.ones((ck.n_x_points, ck.n_y_points)) / ck.n_x_points / ck.n_y_points
		self.vx_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		self.vy_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		# self.sig = np.zeros((ck.n_x_points, ck.n_y_points)) + (
		# 	ck.vx_max - ck.vx_min + ck.vy_max - ck.vy_min)

		# Only for testing.
		# self.prob = np.arange(ck.n_x_points*ck.n_y_points).reshape((ck.n_x_points, ck.n_y_points)) / ck.n_x_points / ck.n_y_points
		# self.vx_mu = np.ones((ck.n_x_points, ck.n_y_points)) * ck.x_gap
		# self.vy_mu = np.ones((ck.n_x_points, ck.n_y_points)) * ck.y_gap

	@staticmethod
	def _get_sig(x, y, our_car):
		"""
		Considers distance between our and other car, and 
		our velocity.

		Takes the distance of the vechile and returns the 
		multiplicative variance in it's observation.
		"""
		dist_x = np.abs(x - our_car.x)
		dist_y = np.abs(y - our_car.y)
		dist = np.sqrt(np.square(dist_x) + np.square(dist_y))

		dist_x_max = np.abs(ck.x_max - ck.x_min)
		dist_y_max = np.abs(ck.y_max - ck.y_min)
		dist_max = np.sqrt(np.square(dist_x_max) + np.square(dist_y_max))
		
		assert dist <= dist_max + 1  # 1 for FP error

		vel = np.sqrt(np.square(our_car.vx) + np.square(our_car.vy))
		vel_x_max = np.abs(ck.vx_max - ck.vx_min)
		vel_y_max = np.abs(ck.vy_max - ck.vy_min)
		vel_max = np.sqrt(np.square(vel_x_max) + np.square(vel_y_max))

		sig_x = (dist_x/(dist_x_max+1) + 2*np.abs(our_car.vx)/(vel_x_max+1))*0.1
		sig_y = (dist_y/(dist_y_max+1) + 2*np.abs(our_car.vy)/(vel_y_max+1))*0.1
		sig = (dist/(dist_max+1) + 2*vel/(vel_max+1))*0.1

		return sig, sig_x, sig_y  # 2*10% error in measurement.

	@staticmethod
	def _sensor_measurement(our_car, cars):
		"""
		The new sensor measurements.
		"""
		prob = np.zeros((ck.n_x_points, ck.n_y_points))
		vx_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		vy_mu = np.zeros((ck.n_x_points, ck.n_y_points))

		# We get position of arbitrary number of cars
		Weights = []
		VX = []
		VY = []

		for car in cars:
			# Get the variance in observation. Center of other car.
			_, sig_x, sig_y = Belief._get_sig(car.x + car.l/2, car.y + car.w/2, our_car)
			# Perturb the actual values (add error in observation). Multiplicative.
			x = car.x + car.l/2 + (ck.x_max-ck.x_min)*np.random.normal(0, sig_x)
			y = car.y + car.w/2 + (ck.y_max-ck.y_min)*np.random.normal(0, sig_y)
			vx = car.vx + (ck.vx_max-ck.vx_min)*np.random.normal(0, sig_x)
			vy = car.vy + (ck.vy_max-ck.vy_min)*np.random.normal(0, sig_y)

			# Find the weights. TODO: Might be wrong. The sigmas are not scaled.
			# weights_x = stats.norm(0, np.abs(ck.grid_x-our_car.x)*sig_x + 0.1).pdf(np.abs(ck.grid_x - x))
			# weights_y = stats.norm(0, np.abs(ck.grid_y-our_car.y)*sig_y + 0.1).pdf(np.abs(ck.grid_y - y))
			# The one below might be correct.
			weights_x = stats.norm(x, np.abs(x-our_car.x)*sig_x + 0.1).pdf(ck.grid_x)
			weights_y = stats.norm(y, np.abs(y-our_car.y)*sig_y + 0.1).pdf(ck.grid_y)
			weights = weights_x * weights_y
			weights += ck.prob_unif_blur
			weights /= weights.sum()

			assert not np.isnan(weights).any()
			assert weights.all() >= 0
			# TODO: Why the spikes in plot for low y? Probabily floating pt error.
			# print(x, y); draw.plot3d(ck.grid_x, ck.grid_y, weights)

			prob += weights
			# Append the weights. We will add the velocity according to the weight.
			Weights.append(weights)
			VX.append(vx)
			VY.append(vy)

		# Adding some prob to all points to avoid 0.
		n_car = prob.sum()
		prob += 1e-3
		prob = prob / prob.sum() * n_car

		# Averaging the velocity contribution according to weights for each point.
		Weights = np.array(Weights)
		Weights /= Weights.sum(0)
		vx_mu = (np.array(VX).reshape((len(VX),1,1)) * Weights).sum(0)
		vy_mu = (np.array(VY).reshape((len(VY),1,1)) * Weights).sum(0)

		# draw.plot3d(ck.grid_x, ck.grid_y, prob)
		# draw.plot3d(ck.grid_x, ck.grid_y, vx_mu)
		# draw.plot3d(ck.grid_x, ck.grid_y, vy_mu)

		return prob, vx_mu, vy_mu

	def _sensor_update(self, our_car, cars):
		prob_new, vx_mu_new, vy_mu_new = Belief._sensor_measurement(our_car, cars)

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

		# Find the VX by averaging depending on probabilities. TODO. Might be incorrect.
		Probs = np.array([prob_new, self.prob])
		Probs /= Probs.sum(0)
		vx_mu = (np.array([vx_mu_new, self.vx_mu]) * Probs).sum(0)
		vy_mu = (np.array([vy_mu_new, self.vy_mu]) * Probs).sum(0)
		# draw.plot3d(ck.grid_x, ck.grid_y, prob)
		# draw.plot3d(ck.grid_x, ck.grid_y, vx_mu)
		# draw.plot3d(ck.grid_x, ck.grid_y, vy_mu)

		return prob, vx_mu, vy_mu

	@staticmethod
	def _grid_to_index(grid_x, grid_y):
		"""
		Rounds the values to nearest index of X and Y.
		"""
		# np.clip(grid_x,ck.x_min,ck.x_max) changed to grid_x.
		# Because it was accumulating mass at the boundary.
		idx_x = np.mod(np.round((grid_x - ck.x_min) / ck.x_gap).astype(int), ck.n_x_points)
		idx_y = np.mod(np.round((grid_y - ck.y_min) / ck.y_gap).astype(int), ck.n_y_points)

		return idx_x, idx_y

	def _action_update(self, our_car, dt):
		prob = np.zeros((ck.n_x_points, ck.n_y_points))
		vx_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		vy_mu = np.zeros((ck.n_x_points, ck.n_y_points))

		# The new index point. Move the grid by velocity.
		idx_x, idx_y = self._grid_to_index(ck.grid_x + dt*self.vx_mu, ck.grid_y + dt*self.vy_mu)
		# print(idx_x, idx_y)
		np.add.at(prob, (idx_x, idx_y), self.prob)
		np.add.at(vx_mu, (idx_x, idx_y), self.vx_mu)
		np.add.at(vy_mu, (idx_x, idx_y), self.vy_mu)

		# TODO: Gaussian blur with different sigma for different X and Y.
		# Only doing for the nearby 3x3 points.
		# _, sig_x, sig_y = self._get_sig(grid_x, grid_y, our_car)

		# Adding some prob to all points to avoid 0.
		n_car = prob.sum()
		prob += ck.prob_unif_blur
		prob = prob / prob.sum() * n_car

		return prob, vx_mu, vy_mu

	def _action_update_old(self, our_car, dt):
		prob = np.zeros((ck.n_x_points, ck.n_y_points))
		vx_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		vy_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		v_sig = np.zeros((ck.n_x_points, ck.n_y_points))

		ck.grid_x = np.zeros((ck.n_x_points, ck.n_y_points)) + ck.X_points.reshape((ck.n_x_points,1))
		ck.grid_y = np.zeros((ck.n_x_points, ck.n_y_points)) + ck.Y_points.reshape((1,ck.n_y_points))

		# TODO: Really Inefficient. Convert to matrix product.
		for i in range(ck.n_x_points):
			for j in range(ck.n_y_points):
				# Find the weights by moving point by vx*dt and vy*dt
				dist = np.sqrt( np.square(ck.grid_x - (ck.X_points[i] + self.vx_mu[i][j]*dt)) + 
					np.square(ck.grid_y - (ck.Y_points[j] + self.vy_mu[i][j]*dt)) )
				weights = stats.norm(0, self.v_sig[i][j]).pdf(dist)
				# print(i*ck.n_y_points+j, weights.sum())
				weights /= weights.sum()

				prob += self.prob[i][j] * weights
				vx_mu += self.vx_mu[i][j] * weights
				vy_mu += self.vy_mu[i][j] * weights
				v_sig += self.v_sig[i][j] * weights

		return prob, vx_mu, vy_mu, v_sig

	def step(self, our_car, cars, dt):
		"""
		Update the belief.
		"""
		self.prob, self.vx_mu, self.vy_mu = self._action_update(our_car, dt)
		self.prob, self.vx_mu, self.vy_mu = self._sensor_update(our_car, cars)
		# self.prob, self.vx_mu, self.vy_mu = self._sensor_measurement(our_car, cars)