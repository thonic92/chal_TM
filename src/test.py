import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from data.WordData import WordData
import json
from models.WordDataLoader import WordDataLoader


def main():

	with open('data/interim/tokens.json') as f:
		tokens = json.load(f)

	word_data = WordData(tokens, None, 3000)

	print(word_data.token_final[1:4])

	print(word_data.getVocabularyLength())

	# print(word_data.ref_word_to_id)

	word_data_loader = WordDataLoader(word_data, 1)

	for x,y in word_data_loader.generate():
		print(x)
		print(x.shape)
		
		print(y)
		print(y.shape)
		break

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    logger = logging.getLogger(__name__)

    main()