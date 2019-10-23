from .state import State
from . import draw

def init():
	state = State()
	win = draw.Window()
	win.setup()
	win.draw(state)
	# input()
	return state, win

def step(state, win):
	state.step(dt=1)
	# print(state)
	# input()
	win.clear_cars()
	# win.clear_belief()
	win.draw_cars(state)
	# win.draw_belief(state)
	# input()

def destroy(state, win):
	win.destroy()