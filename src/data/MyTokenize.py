import spacy
import pandas as pd
from spacy.matcher import Matcher
from spacy.tokens import Token

class MyTokenize:

	def __init__(self, tweets, hashtags, mentions, spacy_lang = 'fr'):

		self.tweets = tweets

		self.hashtags = hashtags
		self.mentions = mentions

		self.hashtags_arr = None
		self.mentions_arr = None

		self.nlp = spacy.load(spacy_lang, disable=['tagger', 'parser'])

		self.tokens = list()

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

	def processTokenize(self):
		pretty_tweet = list()

		## partie token via spaCy
		## pour flagger
		## - les hashtags
		## - les mentions
		## - les ...
		hashtag_merger = HashtagMerger(self.nlp)
		self.nlp.add_pipe(hashtag_merger, last = True) 

		for doc in self.nlp.pipe(self.tweets['tweet_text'], n_threads = 3, batch_size = 128, disable=['tagger', 'parser']):

			for ent in doc.ents:
				ent.merge()

			## En fonction des information que donne spaCy
			## on retraite les tokens

			token_to_return = list()
			for token in doc:
				if token._.is_hashtag:
					if token.text in 


			self.tokens.append([token.text for token in doc])
			pretty_tweet.append(doc.text)

		tmp_tokens = list()	
		for tokens in self.tokens:
			for token in tokens:
				if token.token in self.hashtags_arr
					tmp_tokens.append("HASHTAG")


		self.tweets.loc[:, 'pretty_tweet_text'] = pretty_tweet


class HashtagMerger(object):
    def __init__(self, nlp):
        
        Token.set_extension('is_hashtag', default=None)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        for span in spans:
            span.merge()
            print(len(span))
            for token in span:
                token._.is_hashtag = True
        return doc