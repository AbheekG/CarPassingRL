import numpy as np
import scipy.stats as stats

from . import constants as ck
# from . import draw


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

		if len(forward_cars) > 1 and abs(forward_cars[-2].y - our_car.y) > ck.y_gap/2:
			# Second condition for our car in other lane as this car.
			car_f2 = {"x": forward_cars[-2].x, "sig": 0}
		else:
			car_f2 = {"x": ck.x_max, "sig": ck.x_max}

		if len(backward_cars) > 0 and (
			len(forward_cars) == 0 or 
			abs(backward_cars[0].y - our_car.y) < ck.y_gap/2 or
			abs(backward_cars[0].x - our_car.x) < 4*abs(forward_cars[-1].x - our_car.x)):
			# Second condition: we see car in other lane if we are in other lane or it is close
			car_b1 = {"x": backward_cars[0].x, "sig": 0}
		else:
			car_b1 = {"x": ck.x_max, "sig": ck.x_max}

		# Add error
		car_f1["sig"] = np.sqrt(car_f1["x"])/5  # Low error for this guy.
		car_f2["sig"] = np.sqrt(car_f2["x"])
		car_b1["sig"] = np.sqrt(car_b1["x"])

		for car in [car_f1, car_f2, car_b1]:
			# print(car["x"], car["sig"], end=' || ')
			car["x"] += np.random.normal(0, car["sig"]**2)
			# print(car["x"])
		
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
		
		car_b1 = self.car_b1
		car_b1["x"] -= (ck.vx_min + ck.vx_max)
		# If negative x, then decrease confidence.
		if car_b1["x"] < 0:
			car_b1["sig"] *= 2

		return self.car_f1, self.car_f2, car_b1

	def step(self, our_car, forward_cars, backward_cars):
		"""
		Update the belief.
		"""
		self.car_f1, self.car_f2, self.car_b1 = self._action_update(our_car)
		# self.car_f1, self.car_f2, self.car_b1 = self._sensor_measurement(our_car, forward_cars, backward_cars)
		self.car_f1, self.car_f2, self.car_b1 = self._sensor_update(our_car, forward_cars, backward_cars)