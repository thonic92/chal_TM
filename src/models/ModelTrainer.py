from keras.callbacks import ModelCheckpoint, CSVLogger


class ModelTrainer:

	def __init__(self, model, data_loader, epochs, verbose = 1, initial_epoch = 0, workers=1, use_multiprocessing=False, callbacks=[]):
		self.model = model
		self.data_loader = data_loader

		self.epochs = epochs
		self.verbose = verbose
		self.initial_epoch = initial_epoch
		self.workers = workers
		self.use_multiprocessing = use_multiprocessing

		self.csv_logger = CSVLogger('{}/training.log'.format(self.model.save_directory))
		self.checkpointer = ModelCheckpoint(filepath=self.model.save_directory+'/model-{epoch:02d}.hdf5', verbose = 1, period = 20)

		callbacks.extend([self.csv_logger, self.checkpointer])
		self.callbacks = callbacks

	def train(self):

		self.model.model.fit_generator(
			generator = self.data_loader.generate(),
			steps_per_epoch = self.data_loader.stepPerEpoch(), 
			epochs = self.epochs,
			verbose = self.verbose,
			initial_epoch = self.initial_epoch,
			callbacks = self.callbacks,
			workers = self.workers,
			use_multiprocessing = self.use_multiprocessing
		)

		self.model.save()


