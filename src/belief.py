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
		self.v_sig = np.zeros((ck.n_x_points, ck.n_y_points)) + (
			ck.vx_max - ck.vx_min + ck.vy_max - ck.vy_min)

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

		WX = []
		WY = []
		VX = []
		VY = []

		for car in cars:
			# Get the variance in observation.
			_, sig_x, sig_y = Belief._get_sig(car.x, car.y, our_car)
			# Perturb the actual values (add error in observation). Multiplicative.
			x = car.x + (ck.x_max-ck.x_min)*np.random.normal(1, sig_x)
			y = car.y + (ck.y_max-ck.y_min)*np.random.normal(1, sig_y)
			vx = car.vx + (ck.vx_max-ck.vx_min)*np.random.normal(1, sig_x)
			vy = car.vy + (ck.vy_max-ck.vy_min)*np.random.normal(1, sig_y)

			# Find the weights
			weights_x = stats.norm(0, np.abs(ck.grid_x-our_car.x)*sig_x + 0.1).pdf(np.abs(ck.grid_x - x))
			weights_y = stats.norm(0, np.abs(ck.grid_y-our_car.y)*sig_y + 0.1).pdf(np.abs(ck.grid_y - y))
			weights = weights_x * weights_y
			for w in [weights, weights_x, weights_y]:
				w /= w.sum()

			assert not np.isnan(weights).any()
			assert weights.all() >= 0
			# TODO: Why the spikes in plot for low y? Probabily floating pt error.
			print(x, y); draw.plot3d(ck.grid_x, ck.grid_y, weights_x)
			print(x, y); draw.plot3d(ck.grid_x, ck.grid_y, weights_y)
			print(x, y); draw.plot3d(ck.grid_x, ck.grid_y, weights)

			prob += weights
			WX.append(weights_x)
			WY.append(weights_y)
			VX.append(vx)
			VY.append(vy)



		return prob, vx_mu, vy_mu

	def _sensor_update(self, our_car, cars):
		prob_new, vx_mu_new, vy_mu_new, v_sig_new = Belief._sensor_measurement(our_car, cars)
		# TODO: Check whether the following trick is sound
		w_new = stats.norm(0, v_sig_new).pdf(0)
		w_old = stats.norm(0, self.v_sig).pdf(0)
		w_sum = w_old + w_new
		w_new /= w_sum
		w_old /= w_sum

		prob = prob_new * w_new + self.prob * w_old
		vx_mu = vx_mu_new * w_new + self.vx_mu * w_old
		vy_mu = vy_mu_new * w_new + self.vy_mu * w_old
		v_sig = v_sig_new * w_new + self.v_sig * w_old

		return prob, vx_mu, vy_mu, v_sig

	def _action_update(self, dt):
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
		# self.prob, self.vx_mu, self.vy_mu, self.v_sig = self._action_update(dt)
		self.prob, self.vx_mu, self.vy_mu, self.v_sig = self._sensor_measurement(our_car, cars)