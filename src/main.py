import copy
import pickle
import multiprocessing
import torch

from .state import State
from . import draw
from . import costs as cost_functions
from . import neural
from . import constants as ck

# def init(model_path=None):
# 	state = State()
# 	win = draw.Window()
# 	win.setup()
# 	win.draw(state)
	
# 	return state, win

# def step(state, win):
# 	state.step(dt=1)
# 	# print(state)
# 	# input()
# 	win.clear_cars()
# 	# win.clear_belief()
# 	win.draw_cars(state)
# 	# win.draw_belief(state)
# 	# input()

# def destroy(state, win):
# 	win.destroy()

def _list_of_list_to_list(lol):
		l = list()
		for l_ in lol:
			l += l_
		return l


def train_run(model, optimizer, our_cars, beliefs, costs, batch_size):
	"""
	Takes the inputs and costs of a run, and trains the nn model.
	"""
	if isinstance(costs[0], list):
		# Batch.
		for costs_ in costs:
			for i in range(len(costs_)-1,0,-1):
				costs_[i-1] += ck.discount*costs_[i]  # Smooth the cost.
				costs_[i-1] /= (1 + ck.discount)  # Normalize

		our_cars = _list_of_list_to_list(our_cars)
		beliefs = _list_of_list_to_list(beliefs)
		costs = _list_of_list_to_list(costs)

	else:
		for i in range(len(costs)-1,0,-1):
			costs[i-1] += ck.discount*costs[i]  # Smooth the cost.
			costs[i-1] /= (1 + ck.discount)  # Normalize

	assert len(our_cars) == len(beliefs) == len(costs)
	for i in range(0, len(our_cars), batch_size):
		i_end = min(len(our_cars), i+batch_size)

		out = model(our_cars[i:i_end], beliefs[i:i_end])
		# TODO. Won't this try to minimize all things. Yes, but proportionally
		loss = (out*torch.tensor(costs[i:i_end]).view(len(costs[i:i_end]), 1)).mean()
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


def generate_run(model, time_steps, dt, show):
	"""
	Generates data from a run.
	"""
	state = State(model)

	if show:
		win = draw.Window()
		win.setup()
		win.draw(state)
	
	our_cars = list()
	beliefs = list()
	costs = list()

	for i_collision in range(time_steps):
		state.step(dt=dt)
		if show:
			win.clear_cars()
			win.draw_cars(state)

		our_car = state.our_car
		belief = torch.FloatTensor([state.belief.prob, state.belief.vx_mu, state.belief.vy_mu])
		cost = [cost_functions.cost_vel(state)*ck.vel_weight,
			cost_functions.cost_acc(state)*ck.acc_weight,
			cost_functions.cost_lane(state)*ck.lane_weight,
			cost_functions.cost_collision(state, dt)*ck.collision_weight]
		our_cars.append(copy.deepcopy(our_car))
		beliefs.append(copy.deepcopy(belief))
		costs.append(sum(cost))

		if cost[-1] > 0.1:  # Collision
			break

	if show:
		input()
		win.destroy()

	return our_cars, beliefs, costs, i_collision


def train(model, prev_epoch=0, epochs=100, time_steps=1000, dt=1, show=False,
	batch_size=ck.batch):
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	model.train()

	collision_length = []

	for epoch in range(epochs):
		our_cars, beliefs, costs, i_collision = generate_run(model, time_steps, dt, show)

		print("Epoch: %d. Collision: %d/%d" % (prev_epoch+epoch, i_collision, time_steps))
		collision_length.append(i_collision)

		# TODO. Training model with(/out) collision. May cause issues.
		train_run(model, optimizer, our_cars, beliefs, costs, batch_size)

	return collision_length


def generate_run_parallel(model, time_steps, dt,
	our_cars_batch, beliefs_batch, costs_batch, i_collision_batch):
	our_cars, beliefs, costs, i_collision = generate_run(model, time_steps, dt, False)
	our_cars_batch.append(our_cars)
	beliefs_batch.append(beliefs)
	costs_batch.append(costs)
	i_collision_batch.append(i_collision)


def train_parallel(model, prev_epoch=0, epochs=100, time_steps=1000, dt=1,
	batch_size=ck.batch):

	lock = multiprocessing.Lock()
	def generate_run_parallel(model, time_steps, dt,
		our_cars_batch, beliefs_batch, costs_batch, i_collision_batch):
		our_cars, beliefs, costs, i_collision = generate_run(model, time_steps, dt, False)
		
		lock.acquire()
		our_cars_batch.append(our_cars)
		beliefs_batch.append(beliefs)
		costs_batch.append(costs)
		i_collision_batch.append(i_collision)
		lock.release()

	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	model.train()

	collision_length = []

	for epoch in range(epochs):
		our_cars_batch = multiprocessing.Manager().list()
		beliefs_batch = multiprocessing.Manager().list()
		costs_batch = multiprocessing.Manager().list()
		i_collision_batch = multiprocessing.Manager().list()

		jobs = []
		for _ in range(batch_size):
			proc = multiprocessing.Process(target=generate_run_parallel, args=(
				model, time_steps, dt,
				our_cars_batch, beliefs_batch, costs_batch, i_collision_batch))
			jobs.append(proc)
			proc.start()

		for proc in jobs:
			proc.join()

		print(len(our_cars_batch))
		i_collision = sum(i_collision_batch)/batch_size
		print("Epoch: %d. Collision: %d/%d" % (prev_epoch+epoch, i_collision, time_steps))
		collision_length.append(i_collision)

		# TODO. Training model with(/out) collision. May cause issues.
		train_run(model, optimizer, our_cars_batch, beliefs_batch, costs_batch, batch_size)

	return collision_length


def main():
	model_path="data/model.ckpt"
	collision_path="data/collision.pkl"

	model = neural.CNN()
	try:
		model.load_state_dict(torch.load(model_path))
		collision_length = pickle.load(open(collision_path, 'rb'))
		print("Model loaded. Continue from epoch: %d " % len(collision_length))
	except:
		collision_length = []

	collision_length += train(model, prev_epoch=len(collision_length), epochs=1000)

	torch.save(model.state_dict(), model_path)
	pickle.dump(collision_length, open(collision_path, 'wb'))