from . import constants as ck
import numpy as np

min_follow_dist = ck.t_min * ck.vx_max + ck.x_gap


class Control:
	def __init__(self, dist=ck.t_min*ck.vx_max + ck.x_gap, expl=0, stab=100):
		"""
		dist: Distance behind front vehicle where maneuver starts
		expl: Time steps to explore
		stab: Time steps to wait at start to make things stable
		"""
		self.mode = "follow"  # follow/stab/expl/pass/done
		self.pass_left = 0  # Num of timesteps remaining for pass

		self.dist = dist
		self.stab = stab

		self.expl_left = 0
		self.expl_total = expl

		self.ax_set = 0
		self.ay_set = 0

	def __repr__(self):
		out = "mode = %s, pass_left = %s\n" % (self.mode, self.pass_left)
		out += "dist = %s, expl_left = %s, expl_total = %s, stab = %s\n" % (self.dist, self.expl_left, self.expl_total, self.stab)
		return out

	@staticmethod
	def _check_pass(our_car, belief):
		if belief.car_f2["x"] - belief.car_f1["x"] < 2*min_follow_dist:
			return False
		if 3*(belief.car_f1["x"] - our_car.x + min_follow_dist) + 3*min_follow_dist > (
			belief.car_b1["x"] - our_car.x):
			return False

		return True

	def _action_stability(self, our_car, belief):
		self.mode = "stab"
		assert self.ay_set == 0
		# Set the right distance
		if np.abs(belief.car_f1["x"] - our_car.x - self.dist) > 1.5*ck.x_gap:
			self.ax_set = 1
			ax = ck.ax_max * np.sign(belief.car_f1["x"] - our_car.x - self.dist)
			ay = 0
		else:
			ax = 0; ay = 0
			if self.ax_set == 1:
				ax = -our_car.ax
				self.ax_set = 2
			elif self.ax_set == 2:
				ax = 0
				self.ax_set = 0
			else:
				assert self.ax_set == 0
				assert our_car.ax == 0 and our_car.ay == 0
				self.stab -= 1
		return ax, ay

	def _action_exploration(self, our_car, belief):
		self.mode = "expl"
		assert self.ax_set == 0
		ax = 0
		# Already in an exploration maneuver.
		if self.expl_left > 0:
			ay = ck.ay_max
			self.ay_set = 1
			self.expl_left -= 1
		else:
			if self.ay_set == 1 or self.ay_set == 2:
				ay = ck.ay_min
				self.ay_set += 1
			elif self.ay_set == 3:
				ay = ck.ay_max
				self.ay_set = 4
			elif self.ay_set == 4:
				ay = 0
				self.ay_set = 0
			else:
				assert self.ay_set == 0
				assert our_car.ax == 0 and our_car.ay == 0
				# If there is no chance of collision, then explore.
				if belief.car_b1["x"] - our_car.x > 4*min_follow_dist:
					self.expl_left = min ( self.expl_total, np.floor( 
						(belief.car_b1["x"] - our_car.x - 4*min_follow_dist) / ck.vx_max))
					self.expl_total -= self.expl_left
					if self.expl_left == 0:
						self.mode = "follow"
						ay = 0
					else:
						ay = ck.ay_max
						self.ay_set = 1
				else:
					ay = 0

		return ax, ay

	def _action_passing(self, our_car, belief):
		ay = 0
		if np.abs(our_car.vy) > 0.01:
			ay = ck.ay_min * np.sign(our_car.vy)
		
		if self.pass_left > 0:
			ax = ck.ax_max
			self.ax_set = 1
			self.pass_left -= 1
		else:
			if self.ax_set == 1:
				ax = ck.ax_min
				ay = ck.ay_min
				self.ax_set = 2
			elif self.ax_set == 2:
				ax = 0
				self.ax_set = 0
				self.mode = "done"
			else:
				assert self.ax_set == 0
				# assert our_car.ax == 0 and our_car.ay == 0
				assert self.pass_left == 0
				if self._check_pass(our_car, belief):
					ax = ck.ax_max
					ay = ck.ay_max
					self.mode = "pass"
					self.pass_left = np.ceil((belief.car_f1["x"] - our_car.x + min_follow_dist) / 
						((ck.vx_min + ck.vx_max)/2))
				else:
					ax = 0

		return ax, ay

	def action(self, our_car, belief):
		# print(our_car)
		# print(belief)
		# print(self)

		# Stability mode. Move to the particular distance and wait for self.stab time.
		if self.stab > 0:
			return self._action_stability(our_car, belief)
		
		assert self.stab == 0
		
		# Exploration.
		if self.expl_total > 0 or self.expl_left > 0 or self.mode == "expl":
			return self._action_exploration(our_car, belief)

		assert self.expl_total == 0 and self.expl_left == 0 

		# Passing
		if self.mode != "done":
			return self._action_passing(our_car, belief)

		return 0, 0