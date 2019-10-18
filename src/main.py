from .state import State
from . import draw

def init():
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
	win.draw_cars(state)

def destroy(state, win):
	win.destroy()