from keras.models import load_model

class AbstractModel:
	def __init__(self, save_directory):
		self.save_directory = save_directory

	def save(self):
		self.model.save('{}/final_model.hdf5'.format(self.save_directory))


	def load(self, which = 'final_model'):
		self.model = load_model('{}/{}.hdf5'.format(self.save_directory, which))

