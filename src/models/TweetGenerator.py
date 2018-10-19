from keras.utils import to_categorical
import numpy as np

class TweetGenerator:

	def __init__(self, model, word_data):
		self.model = model
		self.word_data = word_data

	def nextWord(self, mot_debut = None, batch_size = 1):
		"""
			from array of one hot
		"""
		if mot_debut is None:
			mot_debut = self.word_data.wordToId(self.word_data.sentence_token_start)

		mot_debut = mot_debut.reshape(1, 1, self.word_data.getVocabularyLength())

		predict = self.model.model.predict(mot_debut, batch_size = batch_size)
		predict = predict[-1][-1, :]

		word = self.word_data.probaToWord(predict)

		return word, predict

	def tweet(self, mot_debut = None, batch_size = 1, stop=244):

		sent = list()
		pred = mot_debut

		for i in range(stop):
			word, pred = self.nextWord(pred, batch_size)
			sent.append(word)
			if word == self.word_data.sentence_token_stop:
 				break

		return sent

