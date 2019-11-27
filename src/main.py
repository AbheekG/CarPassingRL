import copy
import pickle

from .state import State
from . import draw
from . import constants as ck
from . import costs as cost_functions

def init(model_path=None):
	state = State()
	win = draw.Window()
	win.setup()
	win.draw(state)
	
	return state, win

def step(state, win):
	state.step()
	# print(state)
	# input()
	win.clear_cars()
	# win.clear_belief()
	win.draw_cars(state)
	# win.draw_belief(state)
	# input()

def destroy(state, win):
	win.destroy()

def test_main():
	state, win = init()
	input()
	for _ in range(100):
		step(state, win)
		# print(state.our_car)
		print(cost_functions.cost_vel(state)*ck.vel_weight,
			cost_functions.cost_acc(state)*ck.acc_weight,
			cost_functions.cost_lane(state)*ck.lane_weight,
			cost_functions.cost_collision(state)*ck.collision_weight)
		input()
	destroy(state, win)

def main():
	# train_main()
	test_main()