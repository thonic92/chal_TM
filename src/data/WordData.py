import pandas as pd
from keras.utils import to_categorical
import numpy as np

## Class pour génrer les tokens en mode flat
## - virer les tokens inutiles
## - les tweets trop similaire ? (cf thomas)

class WordData:

	def __init__(self, tokens, tweets, nb_keep, keep_unknown = True, start_and_stop = True):

		self.tokens = tokens
		self.tweets = tweets
		self.nb_keep = nb_keep
		self.token_keep = None
		self.UNKNOWN = '_UNKNOWN_'
		self.token_final = list()
		self.sentence_token_start = '_SENTENCE_START_'
		self.sentence_token_stop = '_SENTENCE_STOP_'
		self.ref_word_to_id = None
		self.start_and_stop = start_and_stop
		self.keep_unknown = keep_unknown

		self.initListToken()
		self.truncate()
		self.computeWordToId()

	def initListToken(self):

		tokens_flat = [token for x in self.tokens for token in x]
		tokens_df = pd.DataFrame(tokens_flat, columns = ['token'])

		freq = tokens_df.groupby('token')['token'].count().reset_index(name = 'count').sort_values(['count'], ascending=False)
		if self.nb_keep > 0:
			freq = freq.head(self.nb_keep)
		self.token_keep = freq['token'].tolist()

		if self.start_and_stop:
			self.token_keep.append(self.sentence_token_start)
			self.token_keep.append(self.sentence_token_stop)
			self.token_keep.append(self.UNKNOWN)

		return self

	def truncate(self):
		
		for i, sentence in enumerate(self.tokens):
			if self.start_and_stop:
				token_final = [self.sentence_token_start]
			else:
				token_final = []
			for token in sentence:
				if self.nb_keep > 0 :
					if token in self.token_keep:
						token_final.append(token)
					elif self.keep_unknown:
						token_final.append(self.UNKNOWN)
				else:
					token_final.append(token)
			if self.start_and_stop:
				token_final.append(self.sentence_token_stop)
			self.token_final.append(token_final)

		return self

	def getFlatFinalToken(self):
		return [token for x in self.token_final for token in x]
		
	def getVocabularyLength(self):
		return len(self.token_keep)

	def computeWordToId(self):
		self.ref_word_to_id = dict([(w, i) for i,w in enumerate(self.token_keep)])

		return self

	def wordToId(self, word):
		return to_categorical(self.ref_word_to_id[word], self.getVocabularyLength())


	def idToWord(self, id):
		return self.token_keep[id]

	def probaToWord(self, proba, idx = 0):
		if np.sum(proba) == 0:
			proba = self.wordToId(self.sentence_token_stop)
		proba_argsort = np.argsort(-proba)
		return self.idToWord(proba_argsort[idx])


	@staticmethod
	def one_hot(words, vocabulary):
		to_categorical(words, vocabulary)

