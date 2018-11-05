from keras.utils import to_categorical
import numpy as np

class TweetGenerator:

	def __init__(self, model, word_data):
		self.model = model
		self.word_data = word_data

	def nextWordOld(self, mot_debut = None, batch_size = 1):
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

	def nextWord(self, mot_debut = None, batch_size = 1):
		"""
			from array of one hot
		"""
		mot_debut = mot_debut.reshape(batch_size, mot_debut.size)

		predict = self.model.model.predict(mot_debut, batch_size = batch_size)
		predict = predict[-1][-1, :]

		# print(np.sort(-predict)[:10])
		idx = 0
		if np.absolute(np.sort(-predict)[1]) > 0.3:
			print(np.sort(-predict)[1])
			idx = 1
		word = self.word_data.probaToWord(predict, idx = idx)
		word_id = self.word_data.ref_word_to_id[word]

		return word, word_id, predict

	def tweet(self, mot_debut = None, batch_size = 1, stop=244, seed=675):
		# print("ok")

		sent = list()
		if mot_debut is None:
			mot_debut = np.array([self.word_data.ref_word_to_id[self.word_data.sentence_token_start]])
		else:
			for w in mot_debut:
				sent.append(self.word_data.idToWord(w))
		
		np_id_sent = mot_debut

		np.random.seed(seed)
		for i in range(stop):
			past = np.random.randint(5, 20)
			word, word_id, pred = self.nextWord(np_id_sent[-past:], batch_size)
			np_id_sent = np.append(np_id_sent, word_id)
			sent.append(word)
			if word == self.word_data.sentence_token_stop:
 				break

		return sent

	def randomTweet(self, stop):
		return self.tweet(np.array([np.random.randint(self.word_data.getVocabularyLength()-1)]), stop = stop)

