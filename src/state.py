import numpy as np

from . import constants as ck


class State:
	def __init__(self):
		self._init_our_car()
		self._init_other_cars()

		# Probability of another car or object being present at the grid location.
		self.grid_prob = np.ones((ck.n_x_points, ck.n_y_points)) / ck.n_x_points / ck.n_y_points
		self.grid_vx_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		self.grid_vx_sig = np.ones((ck.n_x_points, ck.n_y_points)) * 30
		self.grid_vy_mu = np.zeros((ck.n_x_points, ck.n_y_points))
		self.grid_vy_sig = np.ones((ck.n_x_points, ck.n_y_points)) * 1

	def __repr__(self):
		out = "Our Car:\n"
		for var in ["x", "y", "length", "width", "vx", "vy", "ax", "ay"]:
			out += "self.%s_car = %s\n" % (var, eval("self.%s_car" % (var)))

		out += "Other Cars:\n"
		for var in ["x", "y", "length", "width", "vx", "vy"]:
			out += "self.%s_others = %s\n" % (var, eval("self.%s_others" % (var)))
		return out

	def _init_our_car(self):
		"""
		Setup the initial state for our car.
		"""
		self.length_car = ck.car_length
		self.width_car = ck.car_width
		# TODO: Variable (our) car length
		# self.length_car = np.clip(ck.car_length*np.random.normal(1, perturb),
		# 	ck.car_length*0.5, ck.car_length*1.5)
		# self.width_car = np.clip(ck.car_width*np.random.normal(1, perturb),
		# 	ck.car_width*0.5, ck.car_width*1.5)

		self.x_car = ck.x_min
		self.y_car = np.clip(np.random.normal((ck.y_min+ck.y_max)*0.1, (ck.y_max-ck.y_min)*0.1),
							 ck.y_min, ck.y_max - self.width_car)
		self.vx_car = np.clip(np.random.normal((ck.vx_min+ck.vx_max)*0.75, (ck.vx_max-ck.vx_min)*0.2),
							  ck.vx_min, ck.vx_max)
		self.vy_car = np.clip(np.random.normal((ck.vy_min+ck.vy_max)/2, (ck.vy_max-ck.vy_min)*0.2),
							  ck.vy_min, ck.vy_max)
		self.ax_car = np.clip(np.random.normal((ck.ax_min+ck.ax_max)/2, (ck.ax_max-ck.ax_min)*0.2),
							  ck.ax_min, ck.ax_max)
		self.ay_car = np.clip(np.random.normal((ck.ay_min+ck.ay_max)/2, (ck.ay_max-ck.ay_min)*0.2),
							  ck.ay_min, ck.ay_max)

	def _init_other_cars(self):
		"""
		Setup the initial state for other car.
		"""
		self.x_others = []
		self.y_others = []
		self.length_others = []
		self.width_others = []
		self.vx_others = []
		self.vy_others = []

		# Generate cars using given lambda (Poisson).
		# Other lane
		# My lane
		x = self.x_car + self.length_car
		while True:
			x += np.random.exponential(1/ck.poisson)
			if x + ck.car_length > ck.x_max:
				break
			x, _, length, _, _, _ = self._new_car(x=x, y=ck.y_max*0.25-ck.car_width/2, vx=ck.vx_max*0.7)
			x += length

		x = ck.x_min
		while True:
			x += np.random.exponential(1/ck.poisson)
			if x + ck.car_length > ck.x_max:
				break
			x, _, length, _, _, _ = self._new_car(x=x, vx=ck.vx_min*0.7)
			x += length

	def _new_car(self, x=ck.x_max-ck.car_length, y=ck.y_max*0.75-ck.car_width/2, 
		length=ck.car_length, width=ck.car_width,
		vx=(ck.vx_min+ck.vx_max)/2, vy=(ck.vy_min+ck.vy_max)/2, perturb=0.1):
		"""
		Create a new car with given parameters and perturbations.
		"""
		if perturb is not None:
			length = np.clip(length*np.random.normal(1, perturb), length*0.5, length*1.5)
			width = np.clip(width*np.random.normal(1, perturb), width*0.5, width*1.5)
			x = np.clip(x*np.random.normal(1, perturb), max(x*0.8,ck.x_min), min(x*1.2,ck.x_max-length))
			y = np.clip(y*np.random.normal(1, perturb), max(y*0.8,ck.y_min), min(y*1.2,ck.y_max-width))
			vx = np.clip(vx*np.random.normal(1, perturb), ck.vx_min, ck.vx_max)
			vy = np.clip(vy*np.random.normal(1, perturb), ck.vy_min, ck.vy_max)

		for var in ["x", "y", "length", "width", "vx", "vy"]:
			eval("self.%s_others.append(%s)" % (var, var))

		assert length <= ck.x_max - ck.x_min
		assert width <= ck.y_max - ck.y_min
		return x, y, length, width, vx, vy

	def _step_my_car(self, dt, perturb):
		# Our car
		# TODO: Min following distance.
		self.x_car += self.vx_car*dt
		self.x_car = np.clip(self.x_car, ck.x_min, ck.x_max - self.length_car)
		self.y_car += self.vy_car*dt
		self.y_car = np.clip(self.y_car, ck.y_min, ck.y_max - self.width_car)
		self.vx_car += self.ax_car*dt
		self.vx_car = np.clip(self.vx_car, ck.vx_min, ck.vx_max)
		self.vy_car += self.ay_car*dt
		self.vy_car = np.clip(self.vy_car, ck.vy_min, ck.vy_max)
		# TODO acceleration.

	def _step_other_cars(self, dt, perturb):
		"""
		Move other cars one time step.
		"""
		for i in range(len(self.x_others)):
			self.x_others[i] += self.vx_others[i]*dt
			self.y_others[i] += self.vy_others[i]*dt
			self.vx_others[i] *= np.random.normal(1, perturb)
			self.vy_others[i] *= np.random.normal(1, perturb)

	def _clean_old_cars(self):
		"""
		Remove cars that have goe out of grid.
		"""
		# TODO: Inefficient.
		indices_to_remove = []
		for i in range(len(self.x_others)):
			if self.x_others[i] + self.length_others[i] <= 0:
				indices_to_remove.append(i)
			elif self.x_others[i] < 0:
				self.length_others[i] += self.x_others[i]
				self.x_others[i] = 0

		for var in ["x", "y", "length", "width", "vx", "vy"]:
			for i in sorted(indices_to_remove, reverse=True):
				eval("self.%s_others.pop(%i)" % (var, i))

	def _add_new_cars(self, origin_movement):
		# Find the last location till which we searched.
		x = self.x_car + self.length_car
		for i in range(len(self.x_others)):
			if self.vx_others[i] < 0:
				x = max(x, self.x_others[i] + self.length_others[i])
		x += origin_movement

		while True:
			x += np.random.exponential(1/ck.poisson)
			if x + ck.car_length > ck.x_max:
				break
			x, _, length, _, _, _ = self._new_car(x=x, y=ck.y_max*0.75-ck.car_width/2, vx=ck.vx_min*0.7)
			# print(x)
			x += length

	def step(self, dt=0.1, perturb=0.1):
		self._step_my_car(dt, perturb)
		self._step_other_cars(dt, perturb)
		
		# Reset origin
		origin_movement = self.x_car
		self.x_car = 0
		for i in range(len(self.x_others)):
			self.x_others[i] -= origin_movement

		self._clean_old_cars()

		# Sanity check
		# print(sorted(self.x_others))
		# Check the number of cars in each array is consistent.
		for var in ["x", "y", "length", "width", "vx", "vy"]:
			assert len(self.x_others) == eval("len(self.%s_others)" % var), (
				len(self.x_others), eval("len(self.%s_others)" % var))
		# 

		# Add new cars.
		self._add_new_cars(origin_movement)

		# Clip values to be on road.
		for i in range(len(self.x_others)):
			self.x_others[i] = np.clip(self.x_others[i], ck.x_min, ck.x_max - self.length_others[i])
			self.y_others[i] = np.clip(self.y_others[i], ck.y_min, ck.y_max - self.width_others[i])
			self.vx_others[i] = np.clip(self.vx_others[i], ck.vx_min, ck.vx_max)
			self.vy_others[i] = np.clip(self.vy_others[i], ck.vy_min, ck.vy_max)

		for var in ["x", "y", "length", "width", "vx", "vy"]:
			eval("print(len(self.%s_others))" % var)