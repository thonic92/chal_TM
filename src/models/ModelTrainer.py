from keras.callbacks import ModelCheckpoint, CSVLogger


class ModelTrainer:

	def __init__(self, model, data_loader, epochs, verbose = 1, initial_epoch = 0):
		self.model = model
		self.data_loader = data_loader

		self.epochs = epochs
		self.verbose = verbose
		self.initial_epoch = initial_epoch

		self.csv_logger = CSVLogger('{}/training.log'.format(self.model.save_directory))
		self.checkpointer = ModelCheckpoint(filepath=self.model.save_directory+'/model-{epoch:02d}.hdf5', verbose = 1, period = 5)

	def train(self):

		self.model.model.fit_generator(
			generator = self.data_loader.generate(),
			steps_per_epoch = self.data_loader.stepPerEpoch(), 
			epochs = self.epochs,
			verbose = self.verbose,
			initial_epoch = self.initial_epoch,
			callbacks = [self.checkpointer, self.csv_logger]
		)

		self.model.save()


