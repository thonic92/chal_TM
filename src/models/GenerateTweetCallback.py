from keras.callbacks import Callback
import numpy as np

class GenerateTweetCallback(Callback):

	def __init__(self, tweet_generator):
		self.tweet_generator = tweet_generator
		self.validation_data = None
		self.model = None

	def on_epoch_end(self, epoch, logs):
		if epoch % 10 == 0:
			t = self.tweet_generator.tweet(stop = 30)
			print(' '.join(t))
			print("\n")
			t = self.tweet_generator.randomTweet(stop = 30)
			print(' '.join(t))
			print("\n")
