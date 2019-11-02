import copy
import pickle
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


def train_run(model, optimizer, our_cars, beliefs, costs):
	"""
	Takes the inputs and costs of a run, and trains the nn model.
	"""
	# TODO: Batch.
	for i in range(len(costs)-1,0,-1):
		costs[i-1] += ck.discount*costs[i]  # Smooth the cost.
		costs[i-1] /= (1 + ck.discount)  # Normalize

	assert len(our_cars) == len(beliefs) == len(costs)
	for i in range(len(our_cars)):
		output = model(our_cars[i], beliefs[i])
		loss = (output * costs[i]).sum()
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


def train(model, prev_epoch=0, epochs=100, time_steps=1000, dt=1, show=False):
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
	model.train()

	collision_length = []

	for epoch in range(epochs):
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
				cost_functions.cost_collision(state)*ck.collision_weight]
			our_cars.append(copy.deepcopy(our_car))
			beliefs.append(copy.deepcopy(belief))
			costs.append(sum(cost))

			if cost[-1] > 0.1:  # Collision
				break

		print("Epoch: %d. Collision: %d/%d" % (prev_epoch+epoch, i_collision, time_steps))
		collision_length.append(i_collision)

		if show:
			input()
			win.destroy()

		# TODO. Training model with(/out) collision. May cause issues.
		train_run(model, optimizer, our_cars, beliefs, costs)

	return collision_length

def main():
	model_path="data/model.ckpt"
	collision_path="data/collision.pkl"

	model = neural.CNN()
	try:
		model.load_state_dict(torch.load(model_path))
		collision_length = pickle.load(open(collision_path))
	except:
		collision_length = []

	collision_length += train(model, prev_epoch=len(collision_length))

	torch.save(model.state_dict(), model_path)
	pickle.dump(collision_length, open(collision_path, 'wb'))