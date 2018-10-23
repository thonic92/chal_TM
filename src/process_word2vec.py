# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import json

from src.data.WordData import WordData
from src.models.Word2VecModel import Word2VecModel
from src.models.WordDataLoader import ContextWordDataLoader
from src.models.ModelTrainer import ModelTrainer


@click.command()
@click.argument('tokens_dir', type=click.Path(), default = 'data/interim')
@click.argument('save_dir', type=click.Path(), default = 'models/word2vec')
@click.argument('nb_token_keep', type=int, default = 100)
@click.argument('output_dim', type=int, default = 100)
@click.argument('batch', type=int, default = 4)
@click.argument('epochs', type=int, default = 1)
@click.argument('step_per_epoch', type=int, default = 10)
@click.argument('windows_size', type=int, default = 2)
def main(tokens_dir, save_dir, nb_token_keep, output_dim, batch, epochs, step_per_epoch, windows_size):

    logger = logging.getLogger(__name__)

    ## ------ lecture des tokens ------
    logger.info('processTokenize')
    with open('{}/tokens.json'.format(tokens_dir)) as f:
        tokens = json.load(f)

    ## ------ Class WordData ------
    logger.info('Class WordData')
    word_data = WordData(tokens, None, nb_token_keep)


    ## ------ ContextWordDataLoader ------
    logger.info('ContextWordDataLoader')
    contexte_loader = ContextWordDataLoader(2, word_data, batch, step_per_epoch = None)


    ## ------ Word2Vec model ------
    logger.info('Word2Vec model')
    word2vec_model = Word2VecModel(save_dir, word_data.getVocabularyLength() + 1, output_dim, 4)

    ## ------ Word2Vec model: summary ------
    logger.info('Word2Vec model: summary')
    print(word2vec_model.model.summary())

    ## ------ Trainer ------
    logger.info('Trainer')
    trainer = ModelTrainer(model = word2vec_model, data_loader = contexte_loader, epochs =  epochs)

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
