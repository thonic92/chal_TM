import numpy as np
from keras.utils import to_categorical

class WordDataLoader:

	def __init__(self, num_steps, skip_step, word_data, batch_size, step_per_epoch = None):

		self.num_steps = num_steps
		self.skip_step = skip_step
		self.word_data = word_data
		self.batch_size = batch_size
		self.step_per_epoch = step_per_epoch
		self.current_idx = 0

	def stepPerEpoch(self):
		if self.step_per_epoch is None:
			return len(self.word_data.token_final) // self.batch_size
		return self.step_per_epoch


class IdWordDataLoader(WordDataLoader):

	def generate(self):
		max_len = max([len(el) for el in self.word_data.token_final])

		tokens_flat = self.word_data.getFlatFinalToken()

		while True:
			X = np.zeros((self.batch_size, self.num_steps))
			Y = np.zeros((self.batch_size, self.num_steps, self.word_data.getVocabularyLength()))

			for i in range(self.batch_size):
				if self.current_idx >= len(self.word_data.token_final):
					self.current_idx = 0

				windows = tokens_flat[self.current_idx:(self.current_idx + self.num_steps + 1)]

				x = windows[:-1]
				x = [self.word_data.ref_word_to_id[word] for word in x]
				X[i, : ] = x

				y = windows[1:]
				y = [self.word_data.ref_word_to_id[word] for word in y]
				y = to_categorical(y, num_classes = self.word_data.getVocabularyLength())
				Y[i, : ] = y

				self.current_idx += self.skip_step
			yield X, Y


class OneHotIdWordDataLoader(WordDataLoader):

	def generate(self):

		max_len = max([len(el) for el in self.word_data.token_final])

		while True:
			X = np.zeros((self.batch_size, max_len, self.word_data.getVocabularyLength()))
			Y = np.zeros((self.batch_size, max_len, self.word_data.getVocabularyLength()))

			for i in range(self.batch_size):
				if self.current_idx >= len(self.word_data.token_final):
					self.current_idx = 0

				x = self.word_data.token_final[self.current_idx][:-1]
				x = [self.word_data.ref_word_to_id[word] for word in x]
				x = to_categorical(x, num_classes=self.word_data.getVocabularyLength())
				X[i, :(x.shape[0]), : ] = x

				y = self.word_data.token_final[self.current_idx][1:]
				y = [self.word_data.ref_word_to_id[word] for word in y]
				y = to_categorical(y, num_classes=self.word_data.getVocabularyLength())
				Y[i, :(y.shape[0]), : ] = y
				
				# idx_debut = self.current_idx*self.batch_size:
				# idx_fin = idx_debut + (self.batch_size) - 1

				self.current_idx += 1

			yield X, Y