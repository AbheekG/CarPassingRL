def generate_run_parallel(model, time_steps,
	our_cars_batch, beliefs_batch, costs_batch, i_collision_batch):
	our_cars, beliefs, costs, i_collision = generate_run(model, time_steps, False)
	our_cars_batch.append(our_cars)
	beliefs_batch.append(beliefs)
	costs_batch.append(costs)
	i_collision_batch.append(i_collision)


def train_parallel(model, prev_epoch=0, epochs=100, time_steps=1000,
	batch_size=ck.batch):

	lock = multiprocessing.Lock()
	def generate_run_parallel(model, time_steps,
		our_cars_batch, beliefs_batch, costs_batch, i_collision_batch):
		our_cars, beliefs, costs, i_collision = generate_run(model, time_steps, False)
		
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
				model, time_steps,
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