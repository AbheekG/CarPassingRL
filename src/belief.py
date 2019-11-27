import numpy as np
import scipy.stats as stats

from . import constants as ck
from . import draw


class Belief:
	def __init__(self):
		# The two cars in front.
		self.car_f1 = {"x": 0, "sig": 1000}
		self.car_f2 = {"x": 0, "sig": 1000}
		self.car_b1 = {"x": 0, "sig": 1000}

	def __repr__(self):
		return "Forw 1: %s\nForw 2: %s\nBack 1: %s\n" % (self.car_f1, self.car_f2, self.car_b1)

	# @staticmethod
	# def _get_seeing_prob(x, y, our_car):
	# 	"""
	# 	Gives the probability of seeing a vehicle.
	# 	For our lane, we see the vehicle with 
	# 	"""
	# 	dist_x = np.abs(x - our_car.x)

	# 	if abs(y - our_car.y) < ck.y_gap/2:
	# 		return 1 / (1 + dist_x)
	# 	else:
	# 		return 0.5 / (1 + dist_x)		

	@staticmethod
	def _sensor_measurement(our_car, forward_cars, backward_cars):
		"""
		The new sensor measurements.
		"""
		if len(forward_cars) > 0:
			car_f1 = {"x": forward_cars[-1].x, "sig": 0}
		else:
			car_f1 = {"x": ck.x_max, "sig": 0}

		if len(forward_cars) > 1:
			car_f2 = {"x": forward_cars[-2].x, "sig": 0}
		else:
			car_f2 = {"x": ck.x_max, "sig": 0}

		if len(backward_cars) > 0:
			car_b1 = {"x": backward_cars[0].x, "sig": 0}
		else:
			car_b1 = {"x": ck.x_max, "sig": 0}
		
		return car_f1, car_f2, car_b1

	def _sensor_update(self, our_car, forward_cars, backward_cars):
		car_f1, car_f2, car_b1 = Belief._sensor_measurement(our_car, forward_cars, backward_cars)

		def _update(new, old):
			new["sig"] += 1e-3
			old["sig"] += 1e-3
			return {
				"x": (new["x"]/new["sig"] + old["x"]/old["sig"]) / (1/new["sig"] + 1/old["sig"]),
				"sig": 2/(1/new["sig"] + 1/old["sig"])
			}

		car_f1 = _update(car_f1, self.car_f1)
		car_f2 = _update(car_f2, self.car_f2)
		car_b1 = _update(car_b1, self.car_b1)

		return car_f1, car_f2, car_b1


	def _action_update(self, our_car):
		def _update(car, sign=1):
			car["x"] += sign * (ck.vx_min + ck.vx_max)/2
			car["sig"] += car["x"]

		return _update(car_f1), _update(car_f2), _update(car_b1, sign=-1)


	def step(self, our_car, forward_cars, backward_cars):
		"""
		Update the belief.
		"""
		# self.prob = self._action_update(our_car)
		self.car_f1, self.car_f2, self.car_b1 = self._sensor_measurement(our_car, forward_cars, backward_cars)
		# self.prob, self.vx_mu, self.vy_mu = self._sensor_measurement(our_car, cars)