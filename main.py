from src import main


if __name__ == "__main__":
	state, win = main.init()
	for _ in range(1000):
		main.step(state, win)
	input()
	main.destroy(state, win)