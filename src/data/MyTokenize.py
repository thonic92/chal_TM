import spacy
import pandas as pd

class MyTokenize:

	def __init__(self, tweets, hashtags, mentions, spacy_lang = 'fr'):

		self.tweets = tweets

		self.hashtags = hashtags
		self.mentions = mentions

		self.hashtags_arr = None
		self.mentions_arr = None

		self.nlp = spacy.load(spacy_lang, disable=['parser'])
		self.docs = list()

		self.initTweets()
		self.initHashtags(200)
		self.initMentions(200)

	def initTweets(self):
		self.tweets = self.tweets[['tweet_id', 'tweet_text']]
		self.tweets.loc[:, 'tweet_text'] = self.tweets['tweet_text'].str.lower()


	def initHashtags(self, threshold):
		self.hashtags_arr = self.hashtags[self.hashtags['count'] <= threshold]['hashtags_lower_case']

	def initMentions(self, threshold):
		self.mentions_arr = self.mentions[self.mentions['count'] <= threshold]['mentions_lower_case']

	def tokenizeOne(self, string):
		# commencer par les remplacements simples

		# utiliser spacy pour les plus complexes
		doc = self.nlp(string, disable=['parser'])

		# les entities
		for ent in doc.ents:
			ent.merge()

		return doc

	def processTokenize(self):
		pretty_tweet = list()
		for index, row in self.tweets.iterrows():
			
			doc = self.tokenizeOne(row['tweet_text'])

			self.docs.append(doc)
			self.tweets.loc[index, 'pretty_tweet_text'] = doc.text