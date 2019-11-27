from . import constants as ck
import numpy as np

class Control:
	def __init__(self):
		self.mode = "follow"  # follow/pass
		self.timer = 0  # Num of timesteps remaining for pass

	def __repr__(self):
		return "Mode = %s, Timer = %s\n" % (self.mode, self.timer)

	@staticmethod
	def _check_pass(our_car, belief):
		min_follow_dist = ck.t_min * ck.vx_max + ck.x_gap
		if belief.car_f2["x"] - belief.car_f1["x"] < 2*min_follow_dist:
			return False
		if 3*(belief.car_f1["x"] - our_car.x + min_follow_dist) + 3*min_follow_dist > (
			belief.car_b1["x"] - our_car.x):
			return False

		return True

	def action(self, our_car, belief):
		# print(our_car)
		# print(belief)
		# print(self)
		
		# Fix previous velocity change in y
		ay = 0
		if abs(our_car.vy) > 0.1:
			ay = -np.sign(our_car.vy) * ck.ay_max

		# Pass mode
		if self.mode == "pass":
			if self.timer > 1:
				ax = ck.ax_max
			elif self.timer > 0:
				ax = ck.ax_min
				ay = ck.ay_min
			else:
				ax = 0
				self.mode = "follow"
				self.timer += ck.x_gap
			self.timer -= ck.x_gap
		else:
			# Follow mode
			assert self.mode == "follow"
			assert self.timer == 0
			# Check for passing condition
			min_follow_dist = ck.t_min * ck.vx_max + ck.x_gap
			if self._check_pass(our_car, belief):
				ax = ck.ax_max
				ay = ck.ay_max
				self.mode = "pass"
				self.timer = np.ceil((belief.car_f1["x"] - our_car.x + min_follow_dist) / 
					((ck.vx_min + ck.vx_max)/2))
			else:
				if belief.car_f1["x"] - our_car.x > min_follow_dist:
					ax = ck.ax_max
				else:
					if our_car.vx > (0.6*ck.vx_max + 0.4*ck.vx_min):
						ax = ck.ax_min
					else:
						ax = 0

		# if ay == ck.ay_min: print("Hi2")
		return ax, ay