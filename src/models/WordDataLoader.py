import numpy as np
from keras.utils import to_categorical

class WordDataLoader:

	def __init__(self, word_data, batch_size):
		self.word_data = word_data
		self.batch_size = batch_size
		self.current_idx = 0

	def generate(self):

		while True:
			X = list()
			Y = list()
			for i in range(self.batch_size):
				if self.current_idx >= len(self.word_data.token_final):
					self.current_idx = 0

				x = self.word_data.token_final[self.current_idx][:-1]
				x = [self.word_data.ref_word_to_id[word] for word in x]
				x = to_categorical(x, num_classes=self.word_data.getVocabularyLength())

				y = self.word_data.token_final[self.current_idx][1:]
				y = [self.word_data.ref_word_to_id[word] for word in y]
				y = to_categorical(y, num_classes=self.word_data.getVocabularyLength())
				

				# idx_debut = self.current_idx*self.batch_size:
				# idx_fin = idx_debut + (self.batch_size) - 1

				X.append(x)
				Y.append(y)

				self.current_idx += 1

			yield X, Y