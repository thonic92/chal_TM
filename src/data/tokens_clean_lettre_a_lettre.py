# -*- coding: utf-8 -*-


def main():

	## ------ lecture des tweets.csv ------
	tweets_tokens = pd.read_csv('{}/tweets_2.csv'.format(tokens_dir))
	tokens_ss_pbm = tweets_tokens[(tweets_tokens.count_hashtag_ano <= 2)  & (tweets_tokens.count_mention_ano <= 1)].index

    ## ------ lecture des tokens à garder ------
	with open('{}/tokens_id_keep_2.json'.format(tokens_dir)) as f:
		tokens_id_keep = json.load(f)

