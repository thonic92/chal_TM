# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import json
import pandas as pd

from src.data.WordData import WordData
from src.models.simpleLSTMModel import simpleLSTMModel
from src.models.WordDataLoader import IdWordDataLoader
from src.models.ModelTrainer import ModelTrainer
from src.models.TweetGenerator import TweetGenerator
from src.models.GenerateTweetCallback import GenerateTweetCallback

from keras.callbacks import LambdaCallback

@click.command()
@click.argument('goal', type=str, default = "run")
@click.argument('tokens_dir', type=click.Path(), default = 'data/interim')
@click.argument('save_dir', type=click.Path(), default = 'models/simpleLSTMModel/version2')
@click.argument('model_dir', type=click.Path(), default = 'models/simpleLSTMModel/version2')
@click.argument('nb_token_keep', type=int, default = 10000)

@click.argument('lstm_hidden_size', type=int, default = 200)
@click.argument('num_step', type=int, default = 20)
@click.argument('skip_step', type=int, default = 5)

@click.argument('batch', type=int, default = 4)
@click.argument('epochs', type=int, default = 1)
@click.argument('step_per_epoch', type=int, default = 100)

@click.argument('model_name', type=str, default = "final_model")
def main(goal, tokens_dir, save_dir, model_dir, nb_token_keep, lstm_hidden_size, num_step, skip_step, batch, epochs, step_per_epoch, model_name):

	logger = logging.getLogger(__name__)

	## ------ lecture des tokens ------
	logger.info('processTokenize')
	with open('{}/tokens_2.json'.format(tokens_dir)) as f:
		tokens = json.load(f)

	## ------ lecture des tweets.csv ------
	tweets_tokens = pd.read_csv('{}/tweets_2.csv'.format(tokens_dir))
	tokens_ss_pbm = tweets_tokens[(tweets_tokens.count_hashtag_ano <= 2)  & (tweets_tokens.count_mention_ano <= 1)].index

    ## ------ lecture des tokens à garder ------
	with open('{}/tokens_id_keep_2.json'.format(tokens_dir)) as f:
		tokens_id_keep = json.load(f)

	tokens_id_keep = list(set(tokens_id_keep) & set(tokens_ss_pbm))
	print(len(tokens_id_keep))

	## ------ Class WordData ------
	logger.info('Class WordData')
	word_data = WordData(tokens, tokens_id_keep, nb_token_keep, True, True)
	print(len(word_data.tokens))

	## ------ Data Loader ------
	logger.info('IdWordDataLoader')
	sentence_loader = IdWordDataLoader(num_step, skip_step, word_data, batch, step_per_epoch)

	## ------ LSTM model ------
	logger.info('LSTM model')
	lstm_model = simpleLSTMModel(model_dir, word_data.getVocabularyLength(), lstm_hidden_size, (None, None))

	## ------ LSTM model: summary ------
	logger.info('LSTM model: summary')
	print(lstm_model.model.summary())

	## ------ Tweet Generator - callback ------
	tweet_generator = TweetGenerator(lstm_model, word_data)
	generate_tweet_callback = GenerateTweetCallback(tweet_generator)

	if goal == 'run':

		## ------ Trainer ------
		logger.info('Trainer')
		floydhub_log_callback = LambdaCallback(on_epoch_end = lambda epoch, logs: print('{"metric": "accuracy", "value":'+ str(logs['categorical_accuracy']) +' }'+' '+'{"metric": "loss", "value":'+ str(logs['loss']) +' }'))
		trainer = ModelTrainer(model = lstm_model, data_loader = sentence_loader, epochs =  epochs, callbacks = [floydhub_log_callback, generate_tweet_callback])

		try:
			trainer.train()
		except KeyboardInterrupt:
			logger.info('Save model before exiting program...')
			trainer.model.save()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
