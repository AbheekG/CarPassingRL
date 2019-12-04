import copy
import pickle
import multiprocessing as mp

from .state import State
from . import draw
from . import control
from . import constants as ck
from . import costs as cost_functions

def init(model_path=None):
	state = State(control.Control(dist=7, expl=6, stab=50))
	win = draw.Window()
	win.setup()
	win.draw(state)
	
	return state, win

def step(state, win):
	state.step()
	win.clear_cars()
	win.draw_cars(state)

def destroy(state, win):
	win.destroy()

def test_main():
	for _ in range(2):
		state, win = init()
		# input()
		for _ in range(500):
			step(state, win)
			# print(state.our_car)
			# print(cost_functions.cost_vel(state)*ck.vel_weight,
			# 	cost_functions.cost_acc(state)*ck.acc_weight,
			# 	cost_functions.cost_lane(state)*ck.lane_weight,
			# 	cost_functions.cost_collision(state)*ck.collision_weight)
			if cost_functions.cost_collision(state) > 0:
				print("Collision!!!\n\n\n\n")
				break
			input()
		destroy(state, win)

class Stats:
	def __init__(self):
		self.success = 0
		self.crash = 0
		self.total = 0
		self.score = 0

	def __repr__(self):
		return "success = %s, crash = %s, total = %s, score = %s" % (
			self.success, self.crash, self.total, self.score)

stats = dict()
def collect_result(args):
	dist, expl, result = args
	global stats
	if dist not in stats:
		stats[dist] = dict()
	if expl not in stats[dist]:
		stats[dist][expl] = Stats()

	stats[dist][expl].score = stats[dist][expl].score*(stats[dist][expl].total/
		(stats[dist][expl].total + result.total)) + result.score*(result.total/
		(stats[dist][expl].total + result.total))
	stats[dist][expl].success += result.success
	stats[dist][expl].crash += result.crash
	stats[dist][expl].total += result.total

	print("dist = %d, expl = %d, stats = %s" % (dist, expl, stats[dist][expl]))
	with open("data/stats_parallel.txt", "wb") as fp:
		pickle.dump(stats, fp)

def stats_helper(dist, expl, stab):
	result = Stats()
	for _ in range(100):
		result.score = result.score*(
			result.total/(result.total+1))
		result.total += 1

		state = State(control.Control(dist=dist, expl=expl, stab=stab))
		for _ in range(1000):
			state.step()
			if cost_functions.cost_collision(state) > 0:
				result.crash += 1
				result.score -= ck.collision_weight/result.total
				break
			if state.control.mode == "done":
				result.success += 1
				factor = 1
				if len(state.forward_cars) > 0:
					factor = min(1, (state.forward_cars[-1].x - state.our_car.x) / control.min_follow_dist)
				result.score += factor/result.total
				break

	return dist, expl, result

def stats_main():
	global stats
	try:
		with open("data/stats_parallel.txt", "rb") as fp:
			stats = pickle.load(fp)
	except:
		print("Falied to load previous statistics.")

	stab = 100
	for dist in range(1, 50):
		for expl in range(100):
			collect_result(stats_helper(dist, expl, stab))

def stats_parallel():
	global stats
	try:
		with open("data/stats_parallel.txt", "rb") as fp:
			stats = pickle.load(fp)
	except:
		print("Falied to load previous statistics.")

	pool = mp.Pool(mp.cpu_count())
	stab = 100
	for dist in range(1, 50):
		for expl in range(100):
			if dist not in stats or expl not in stats[dist] or stats[dist].total < 100:
				pool.apply_async(stats_helper, args=(dist, expl, stab), callback=collect_result)

	pool.close()
	pool.join()
	with open("data/stats_parallel.txt", "wb") as fp:
		pickle.dump(stats, fp)

def main():
	# stats_main()
	# stats_parallel()
	test_main()