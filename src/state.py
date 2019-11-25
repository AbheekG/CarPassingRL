import numpy as np
import torch

from . import constants as ck
from . import belief


class Car:
	def __init__(self,
		x, y, vx,
		vy = 0, ax = 0, ay = 0
		):
		"""
		The paremeters are reasonable for a backward moving car.
		"""
		# Setting default values.
		self.x = x
		self.y = y
		self.vx = vx
		self.vy = vy
		self.ax = ax
		self.ay = ay

	def __repr__(self):
		out = "Car: {\n"
		for var in ["x", "y", "vx", "vy", "ax", "ay"]:
			out += "%s = %s,\t" % (
				var, eval("self.%s" % (var)) )
		out += "}\n"
		return out

	def step(self, dx=0, dy=0, dvx=0, dvy=0, dax=0, day=0):
		for var in ["x", "y", "vx", "vy", "ax", "ay"]:
			exec("self.%s = self.%s + d%s" % ( (var,)*3 ))
			# exec("self.%s = np.clip(self.%s + d%s, ck.%s_min, ck.%s_max)" % ( (var,)*5 ))

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
			x = ck.x_min,
			y = ck.y_min,
			vx = (ck.vx_min + ck.vx_max)/2
		)

		return car

	def _init_other_cars(self):
		forward_cars = self._add_other_cars(self.our_car.x + ck.x_gap, True)
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

		if forward:
			y = ck.y_min
			vx = (ck.vx_min + ck.vx_max)/2
		else:
			y = ck.y_max
			vx = -(ck.vx_min + ck.vx_max)/2

		# Generate cars using given lambda (Poisson).
		while True:
			x += int(np.random.exponential(1/ck.poisson)) + ck.t_min*(ck.vx_min + ck.vx_max)/2 # Adding minimum following distance.
			if x > ck.x_max:
				break
			car = Car(x=x, y=y, vx=vx)
			x += ck.x_gap
			cars.append(car)

		return cars

	def _step_our_car(self):
		# Take Action.
		ax, ay = self._action()

		dx = np.clip(self.our_car.x + self.our_car.vx, ck.x_min, ck.x_max) - self.our_car.x
		dy = np.clip(self.our_car.y + self.our_car.vy, ck.y_min, ck.y_max) - self.our_car.y
		dvx = np.clip(self.our_car.vx + ax, ck.vx_min, ck.vx_max) - self.our_car.vx
		dvy = np.clip(self.our_car.vy + ay, ck.vy_min, ck.vy_max) - self.our_car.vy

		self.our_car.step(dx=dx, dy=dy, dvx=dvx, dvy=dvy,
			dax=(ax-self.our_car.ax), day=(ay-self.our_car.ay))

	def _step_other_cars(self):
		"""
		Move other cars one time step.
		"""
		for cars in [self.forward_cars, self.backward_cars]:
			for car in cars:
				car.step(dx=car.vx, dy=car.vy, dvx=car.ax, dvy=car.ay)

	def _clean_old_cars(self):
		"""
		Remove cars that have gone out of grid.
		"""
		# TODO. Inefficient. Partial car removal.
		for cars in [self.forward_cars, self.backward_cars]:
			indices_to_remove = []
			for i in range(len(cars)):
				if cars[i].x < ck.x_min: # TODO. Removing forward or cars[i].x > ck.x_max:
					indices_to_remove.append(i)
			
			for i in sorted(indices_to_remove, reverse=True):
				cars.pop(i)

	def _add_new_cars(self, origin_movement):
		"""
		Adds some new cars after each step.
		"""
		# print(self.backward_cars[-1].vx)
		if not self.backward_cars or (
			self.backward_cars[-1].x + ck.t_min*(ck.vx_min + ck.vx_max)/2 > ck.x_max
		):
			return

		# print(origin_movement)
		# Assuming that total movement is 2*origin_movement
		# And adding a car with poisson_prob * origin_movement
		if (self.backward_cars
			and self.backward_cars[-1].x + ck.t_min*(ck.vx_min + ck.vx_max)/2 < ck.x_max 
			and np.random.uniform() < 2*origin_movement*ck.poisson):
			self.backward_cars.append(
				Car(x=ck.x_max, y=ck.y_max, vx=-(ck.vx_min + ck.vx_max)/2)
			)

	def _action(self):
		belief_concat = torch.FloatTensor([self.belief.prob])
		action_probs = self.nn(self.our_car, belief_concat)
		action_idx = int(torch.multinomial(action_probs.flatten(), 1).item())
		# Exploration
		if np.random.uniform() < ck.exploration:
			action_idx = np.random.randint(0, ck.n_x_actions*ck.n_y_actions)
		ax = ck.X_actions[int(action_idx / ck.n_y_actions)]
		ay = ck.Y_actions[action_idx % ck.n_y_actions]
		# print(action_probs.flatten(), action_idx, ax, ay)
		return ax, ay

	def step(self):
		# Update Belief.
		self.belief.step(self.our_car, self.forward_cars, self.backward_cars)

		self._step_our_car()
		self._step_other_cars()

		# Reset origin
		origin_movement = self.our_car.x
		self.our_car.step(dx = -origin_movement)
		for car in self.forward_cars + self.backward_cars:
			car.step(dx = -origin_movement)

		self._clean_old_cars()
		
		# Add new cars.
		self._add_new_cars(origin_movement)
