import pandas as pd
from keras.utils import to_categorical
import numpy as np

## Class pour génrer les tokens en mode flat
## - virer les tokens inutiles
## - les tweets trop similaire ? (cf thomas)

class WordData:

	def __init__(self, tokens, tweets, nb_keep):

		self.tokens = tokens
		self.tweets = tweets
		self.nb_keep = nb_keep
		self.token_keep = None
		self.UNKNOWN = '_UNKNOWN_'
		self.token_final = list()
		self.sentence_token_start = '_SENTENCE_START_'
		self.sentence_token_stop = '_SENTENCE_STOP_'
		self.ref_word_to_id = None

		self.initListToken()
		self.truncate()
		self.computeWordToId()

	def initListToken(self):

		tokens_flat = [token for x in self.tokens for token in x]
		tokens_df = pd.DataFrame(tokens_flat, columns = ['token'])

		freq = tokens_df.groupby('token')['token'].count().reset_index(name = 'count').sort_values(['count'], ascending=False).head(self.nb_keep)
		self.token_keep = freq['token'].tolist()

		self.token_keep.append(self.sentence_token_start)
		self.token_keep.append(self.sentence_token_stop)
		self.token_keep.append(self.UNKNOWN)

		return self

	def truncate(self):
		
		for i, sentence in enumerate(self.tokens):
			token_final = [self.sentence_token_start]
			for token in sentence:
				if token in self.token_keep:
					token_final.append(token)
				else:
					token_final.append(self.UNKNOWN)
			token_final.append(self.sentence_token_stop)
			self.token_final.append(token_final)

		return self
		
	def getVocabularyLength(self):
		return len(self.token_keep)

	def computeWordToId(self):
		self.ref_word_to_id = dict([(w, i) for i,w in enumerate(self.token_keep)])

		return self

	def wordToId(self, word):
		return to_categorical(self.ref_word_to_id[word], self.getVocabularyLength())


	def idToWord(self, id):
		return self.token_keep[id]

	def probaToWord(self, proba):
		if np.sum(proba) == 0:
			proba = self.wordToId(self.sentence_token_stop)
		return self.idToWord(np.argmax(proba))


	@staticmethod
	def one_hot(words, vocabulary):
		to_categorical(words, vocabulary)

