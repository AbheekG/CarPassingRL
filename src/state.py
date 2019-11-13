import numpy as np
import torch

from . import constants as ck
from . import belief


class Car:
	def __init__(self, 
		l=None, l_noise=None, l_min=None, l_max=None,
		w=None, w_noise=None, w_min=None, w_max=None,
		x=None, x_noise=None, x_min=None, x_max=None,
		y=None, y_noise=None, y_min=None, y_max=None,
		vx=None, vx_noise=None, vx_min=None, vx_max=None,
		vy=None, vy_noise=None, vy_min=None, vy_max=None,
		ax=None, ax_noise=None, ax_min=None, ax_max=None,
		ay=None, ay_noise=None, ay_min=None, ay_max=None
		):
		"""
		The paremeters are reasonable for a backward moving car.
		"""
		# Setting default values.
		# Length.
		if l is None: l = ck.car_length
		if l_noise is None: l_noise = 0.1
		if l_min is None: l_min = l*0.5
		if l_max is None: l_max = l*1.5

		# Width
		if w is None: w = ck.car_width
		if w_noise is None: w_noise = 0.1
		if w_min is None: w_min = w*0.5
		if w_max is None: w_max = w*1.5

		# X. Default represents car at x_max
		if x is None: x = ck.x_max - l
		if x_noise is None: x_noise = 0.1
		if x_min is None: x_min = ck.x_min
		if x_max is None: x_max = ck.x_max - l

		# Y. Default backward car.
		if y is None: y = ck.y_min + (ck.y_max-ck.y_min)*0.75 - w/2
		if y_noise is None: y_noise = 0.1
		if y_min is None: y_min = ck.y_min + (ck.y_max-ck.y_min)*0.5 - w
		if y_max is None: y_max = ck.y_max - w

		# VX
		if vx is None: vx = -ck.vx_max*0.75
		if vx_noise is None: vx_noise = 0.1
		if vx_min is None: vx_min = min(sign(vx)*ck.vx_min, sign(vx)*ck.vx_max)
		if vx_max is None: vx_max = max(sign(vx)*ck.vx_min, sign(vx)*ck.vx_max)

		# VY
		if vy is None: vy = (ck.vy_min+ck.vy_max)/2
		if vy_noise is None: vy_noise = 0.1
		if vy_min is None: vy_min = ck.vy_min
		if vy_max is None: vy_max = ck.vy_max

		# AX
		if ax is None: ax = (ck.ax_min+ck.ax_max)/2
		if ax_noise is None: ax_noise = 0.1
		if ax_min is None: ax_min = ck.ax_min
		if ax_max is None: ax_max = ck.ax_max

		# AY
		if ay is None: ay = (ck.ay_min+ck.ay_max)/2
		if ay_noise is None: ay_noise = 0.1
		if ay_min is None: ay_min = ck.ay_min
		if ay_max is None: ay_max = ck.ay_max

		# Store the values, after adding noise and clipping
		for var in ["l", "w", "x", "y", "vx", "vy", "ax", "ay"]:
			exec("self.%s = np.clip(np.random.normal(%s, (%s_max-%s_min)*%s_noise), %s_min, %s_max)" % ((var,)*7))

		# Store bounds and noise.
		for var in ["l", "w", "x", "y", "vx", "vy", "ax", "ay"]:
			for sub in ["noise", "min", "max"]:
				exec("self.%s_%s = %s_%s" % ( (var, sub)*2 ))

	def __repr__(self):
		out = "Car: {\n"
		for var in ["l", "w", "x", "y", "vx", "vy", "ax", "ay"]:
			out += "%s = %s, noise=%s, low=%s, high=%s\n" % (
				var, eval("self.%s" % (var)), eval("self.%s_noise" % (var)),
				eval("self.%s_min" % (var)), eval("self.%s_max" % (var)))
		out += "}\n"
		return out

	def step(self, dx=0, dy=0, dvx=0, dvy=0, dax=0, day=0):
		for var in ["x", "y", "vx", "vy", "ax", "ay"]:
			exec("self.%s = np.clip(self.%s + d%s, self.%s_min, self.%s_max)" % ( (var,)*5 ))

class State:
	def __init__(self, nn):
		self.our_car = self._init_our_car()
		self.forward_cars, self.backward_cars = self._init_other_cars()

		# Probability of another car or object being present at the grid location.
		self.belief = belief.Belief()

		# The neural network to predict action.
		self.nn = nn

	def _init_our_car(self):
		"""
		Setup the initial state for our car.
		"""
		# TODO: Variable car length for our car.
		car = Car(
			l = ck.car_length, l_noise = 0,
			w = ck.car_width, w_noise = 0,
			x = ck.x_min, x_noise = 0,
			y = ck.x_min +(ck.y_min+ck.y_max)*0.25, y_noise = (ck.y_max-ck.y_min)*0.1,
				y_min = ck.y_min, y_max = ck.y_max-ck.car_width,
			vx = ck.vx_min+(ck.vx_max-ck.vx_min)*0.75, vx_noise=(ck.vx_max-ck.vx_min)*0.1,
				vx_min = ck.vx_min, vx_max = ck.vx_max,
			)

		return car

	def _init_other_cars(self):
		forward_cars = self._add_other_cars(self.our_car.x + self.our_car.l, True)
		forward_cars.reverse()  # To make it front to back.
		backward_cars = self._add_other_cars(ck.x_min, False)
		return forward_cars, backward_cars

	def _add_other_cars(self, x_min=0, forward=False):
		"""
		Setup the initial state for other car.

		forward: True implies on our car's lane. Backward means opposite lane.
		"""
		cars = []
		x = x_min
		x_noise = 0

		if forward:
			y = ck.y_min + (ck.y_max-ck.y_min)*0.25 - ck.car_width/2
			y_noise = 0.1
			y_min = ck.y_min
			y_max = (ck.y_min+ck.y_max)*0.5 - ck.car_width

			vx = ck.vx_max*0.7
			vx_noise = 0.1
			vx_min = ck.vx_min
			vx_max = ck.vx_max
		else:
			y = ck.y_min + (ck.y_max-ck.y_min)*0.75 - ck.car_width/2
			y_noise = 0.1
			y_min = (ck.y_min+ck.y_max)*0.5
			y_max = ck.y_max - ck.car_width

			vx = -ck.vx_max*0.7
			vx_noise = 0.1
			vx_min = -ck.vx_max
			vx_max = -ck.vx_min

		# Generate cars using given lambda (Poisson).
		while True:
			x += np.random.exponential(1/ck.poisson)  + ck.vx_max*ck.t_min # Adding minimum following distance.
			if x + ck.car_length > ck.x_max:
				break
			car = Car(x=x, x_noise=x_noise,
				y=y, y_noise=y_noise, y_min=y_min, y_max=y_max,
				vx=vx, vx_noise=vx_noise, vx_min=vx_min, vx_max=vx_max)
			x += car.l
			cars.append(car)

		return cars

	def _step_car(self, car, front_car, dt):
		# Warning: Updates the cars in-place. 
		sign = np.sign(car.vx)
		if front_car is not None and sign*(car.x + car.vx*dt + car.vx*ck.t_min) > sign*front_car.x:
			# TODO. Change, this leads to clustering of cars.
			dx = front_car.x - car.x - car.vx*ck.t_min
			vx = (front_car.vx + car.vx) / 2
			car.step(dx=dx, dy=car.vy*dt, dvx=car.ax*dt + vx - car.vx, dvy=car.ay*dt)
			front_car.step(dvx=vx - front_car.vx)
		else:
			car.step(dx=car.vx*dt, dy=car.vy*dt, dvx=car.ax*dt, dvy=car.ay*dt)

	def _step_our_car(self, dt):
		# Take Action.
		ax, ay = self._action()

		self.our_car.step(dx=self.our_car.vx*dt, dy=self.our_car.vy*dt,
			dvx=self.our_car.ax*dt, dvy=self.our_car.ay*dt,
			dax=(ax-self.our_car.ax)*dt, day=(ay-self.our_car.ay)*dt)
		self._step_car(self.our_car, None, dt)
		# TODO acceleration.

	def _step_other_cars(self, dt, perturb):
		"""
		Move other cars one time step.
		"""
		for cars in [self.forward_cars, self.backward_cars]:
			for i in range(len(cars)):
				car = cars[i]
				front_car = None
				if i > 0:
					front_car = cars[i-1]
					# assert np.sign(car.vx)*front_car.x > np.sign(
					# 	car.vx)*(car.x + car.vx*ck.t_min)
				self._step_car(car, front_car, dt)

	def _clean_old_cars(self):
		"""
		Remove cars that have gone out of grid.
		"""
		# TODO. Inefficient. Partial car removal.
		for cars in [self.backward_cars]:
			indices_to_remove = []
			for i in range(len(cars)):
				if cars[i].x <= 0:
					indices_to_remove.append(i)
			
			for i in sorted(indices_to_remove, reverse=True):
				cars.pop(i)

	def _add_new_cars(self, origin_movement):
		# Find the last location till which we searched.
		x = ck.x_min
		for car in self.backward_cars:
			assert car.vx <= 0
			x = max(x, car.x + car.l)
		assert len(self.backward_cars) == 0 or x == self.backward_cars[-1].x + self.backward_cars[-1].l

		x += origin_movement

		new_cars = self._add_other_cars(x_min=x, forward=False)
		self.backward_cars += new_cars

	def _action(self):
		belief_concat = torch.FloatTensor([self.belief.prob, self.belief.vx_mu, self.belief.vy_mu])
		action_probs = self.nn(self.our_car, belief_concat)
		action_idx = int(torch.multinomial(action_probs.flatten(), 1).item())
		# Exploration
		if np.random.uniform() < ck.exploration:
			action_idx = np.random.randint(0, ck.n_x_actions*ck.n_y_actions)
		ax = ck.X_actions[int(action_idx / ck.n_y_actions)]
		ay = ck.Y_actions[action_idx % ck.n_y_actions]
		# print(action_probs.flatten(), action_idx, ax, ay)
		return ax, ay

	def step(self, dt=0.1, perturb=0.1):
		# Update Belief.
		self.belief.step(self.our_car, self.forward_cars + self.backward_cars, dt)

		self._step_our_car(dt)
		self._step_other_cars(dt, perturb)

		# Reset origin
		origin_movement = self.our_car.x
		self.our_car.step(dx = -origin_movement)
		for car in self.forward_cars + self.backward_cars:
			car.step(dx = -origin_movement)

		self._clean_old_cars()
		
		# Add new cars.
		self._add_new_cars(origin_movement)

		# print(len(self.backward_cars))