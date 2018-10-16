import pandas as pd
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from ReadNestedList import ReadNestedList
from MyTokenize import MyTokenize

def main():
	tweets = pd.read_csv('data/external/csv_datas_full.csv', sep = '\t', low_memory=False)

	# tweets = tweets[-tweets['tweet_text'].str.contains('(h|H)idalgo')]

	mentions = ReadNestedList(tweets, tweets['tweet_user_mentions_list'], "mentions")
	mentions.read().DF().computeGrpDF()
	tmp = mentions.grpDF.head()
	print(tmp)

	hashtags = ReadNestedList(tweets, tweets['tweet_used_hashtags_list'], "hashtags")
	hashtags.read().DF().computeGrpDF()
	tmp = hashtags.grpDF.head()
	print(tmp)

	# tweets = tweets[:100]
	
	my_tokenize = MyTokenize(tweets, hashtags.grpDF, mentions.grpDF)
	print(my_tokenize.tweets.head())

	logger = logging.getLogger(__name__)
	logger.info('processTokenize')

	my_tokenize.processTokenize()

	print(my_tokenize.tweets.head())
	# print(my_tokenize.tokens)

	my_tokenize.tweets.to_csv('data/interim/tweets.csv')
	print(my_tokenize.tweets['pretty_tweet_text'].apply(len).sum())

	logger = logging.getLogger(__name__)
	logger.info('Fin')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    logger = logging.getLogger(__name__)

    main()