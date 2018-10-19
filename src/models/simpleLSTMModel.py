from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM


class simpleLSTMModel:

	def __init__(self, vocabulary_size, lstm_hidden_size, batch_input_shape):

		self.vocabulary_size = vocabulary_size
		self.lstm_hidden_size = lstm_hidden_size
		self.batch_input_shape = batch_input_shape
		self.build_model()
		self.save_directory = 'models/simpleLSTMModel/version1'

	def build_model(self):
		# batch_input_length=self.batch_input_shape.

		model = Sequential()
		model.add(Embedding(input_dim = self.vocabulary_size, output_dim = self.lstm_hidden_size, batch_input_shape = self.batch_input_shape))
		model.add(LSTM(self.lstm_hidden_size, return_sequences=True))
		model.add(TimeDistributed(Dense(self.vocabulary_size)))
		model.add(Activation('softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

		self.model = model

		return self.model

	def save(self):
		self.model.save('{}/final_model.hdf5'.format(self.save_directory))


	def load(self, which = 'final_model'):
		self.model = load_model('{}/{}.hdf5'.format(self.save_directory, which))

	def checkpointer(self):
		ModelCheckpoint(filepath='models/simpleLSTMModel/version1/model-{epoch:02d}.hdf5', verbose = 1, period = 5)

			






