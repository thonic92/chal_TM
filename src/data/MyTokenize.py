import spacy
import pandas as pd
from spacy.matcher import Matcher
from spacy.tokens import Token
# from HashtagMerger import HashtagMerger
import re
import unicodedata
import html

class MyTokenize:

	def __init__(self, tweets, hashtags, mentions, spacy_lang = 'fr'):

		# http://www.noah.org/wiki/RegEx_Python#URL_regex_pattern
		self.url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
		
		self.tweets = tweets

		self.hashtags = hashtags
		self.mentions = mentions

		self.hashtags_arr = None
		self.mentions_arr = None

		self.nlp = spacy.load(spacy_lang, disable=['tagger', 'parser'])

		self.tokens = list()

		self.initTweets()
		self.initHashtags(500)
		self.initMentions(200)


	def initTweets(self):
		"""
			modifier les tweets avant de tokeniser
		"""
		self.tweets = self.tweets[['tweet_id', 'tweet_text']]
		## lower case
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text'].str.lower()
		## rendre le html plutôt que de garder les html entities
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: html.unescape(t))
		## changer correctement l'encodage / caractères spéciaux
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: unicodedata.normalize('NFD', t).encode('ascii', 'ignore').decode('utf-8'))
		## ajouter les bons espaces après les virgules mot,mot => mot, mot
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: re.sub('(?<=\D),(?=\S)', ', ', t))
		## ajouter les bons espaces sur les parenthèses (mot)mot => (mot) mot
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: re.sub('(?<=\S)\)(?=\S)', ') ', t))
		## ajouter les bons espaces sur les hashtags mot#hashtags => mot #hastags
		## normalement on pourrait utiliser la liste des hashtags mais bof... 
		## attention aux url à la place de mot (mais rare car url minifiee sans sharp)
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: re.sub('(?<=\S)#(?=\S)', ' #', t))
		## ajouter les bons espaces sur les mentions mot@mentions => mot @hastags
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: re.sub('(?<=\S)@(?=\S)', ' @', t))
		## ajout les bons espace entre les mot et les urls motURL => mot url
		## je ne veux pas les supprimer tout de suite
		self.tweets.loc[:, 'tweet_text_init'] = self.tweets['tweet_text_init'].apply(lambda t: re.sub("(?<=\S)(?={})".format(self.url_regex), ' ', t))


	def initHashtags(self, threshold):
		self.hashtags_arr = self.hashtags[self.hashtags['count'] <= threshold]['hashtags_lower_case'].tolist()

	def initMentions(self, threshold):
		self.mentions_arr = self.mentions[self.mentions['count'] <= threshold]['mentions_lower_case'].tolist()

	def processTokenize(self):
		pretty_tweet = list()

		column_count_url = list()
		column_count_hashtag = list()
		column_count_mention = list()

		## partie token via spaCy
		## pour flagger
		## - les hashtags
		## - les mentions
		## - les urls

		for doc in self.nlp.pipe(self.tweets['tweet_text_init'], n_threads = 3, batch_size = 128, disable=['tagger', 'parser']):

			for ent in doc.ents:
				ent.merge()

			## En fonction des information que donne spaCy
			## on retraite les tokens

			token_to_return = list()
			
			count_url = 0
			count_hashtag = 0
			count_mention = 0

			for token in doc:
				## url
				if token.like_url:
					count_url +=1
					pass
				## url 2 (à voir si faut pas le faire avant)
				elif re.compile(self.url_regex).match(token.text):
					count_url +=1
					pass
				## hashtags
				elif token._.is_hashtag:
					count_hashtag +=1
					if token.text[1:] in self.hashtags_arr:
						token_to_return.append("_HASHTAG_")
					else:
						token_to_return.append(token.text)
				## mentions
				elif re.compile('^@.*').match(token.text):
					count_mention +=1
					if token.text[1:] in self.mentions_arr:
						token_to_return.append("_MENTION_")
					else:
						token_to_return.append(token.text)
				## guillemets et blancs
				elif re.compile('^("|\s)+$').match(token.text):
					pass
				else:
					token_to_return.append(token.text)

			self.tokens.append(token_to_return)
			pretty_tweet.append(' '.join(token_to_return))

			column_count_url.append(count_url)
			column_count_hashtag.append(count_hashtag)
			column_count_mention.append(count_mention)

		self.tweets.loc[:, 'pretty_tweet_text'] = pretty_tweet
		self.tweets.loc[:, 'count_url'] = column_count_url
		self.tweets.loc[:, 'count_hashtag'] = column_count_hashtag
		self.tweets.loc[:, 'count_mention'] = column_count_mention

