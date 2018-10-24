from src.models.AbstractModel import AbstractModel
from keras.models import Sequential
from keras.layers import Dense, Lambda, Embedding
import keras.backend as K
from keras import optimizers

class Word2VecModel(AbstractModel):

	def __init__(self, save_directory, vocabulary_size, output_dim, input_length):

		super().__init__(save_directory)

		self.vocabulary_size = vocabulary_size
		self.output_dim = output_dim
		self.input_length = input_length
		self.build_model()

	def build_model(self):

		model = Sequential()
		model.add(Embedding(input_dim = self.vocabulary_size, output_dim = self.output_dim, input_length = self.input_length))
		model.add(Lambda(lambda x: K.mean(x, axis = 1), output_shape = (self.output_dim, )))
		model.add(Dense(self.vocabulary_size, activation = 'softmax'))

		adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0, amsgrad = False)

		model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['categorical_accuracy'])

		self.model = model

		return self.model
