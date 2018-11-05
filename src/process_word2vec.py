# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import json

from src.data.WordData import WordData
from src.models.Word2VecModel import Word2VecModel
from src.models.WordDataLoader import ContextWordDataLoader
from src.models.ModelTrainer import ModelTrainer

from keras.callbacks import LambdaCallback

@click.command()
@click.argument('goal', type=str, default = "run")
@click.argument('tokens_dir', type=click.Path(), default = 'data/interim')
@click.argument('save_dir', type=click.Path(), default = 'models/word2vec')
@click.argument('model_dir', type=click.Path(), default = 'models/word2vec')
@click.argument('nb_token_keep', type=int, default = 100)
@click.argument('output_dim', type=int, default = 100)
@click.argument('batch', type=int, default = 4)
@click.argument('epochs', type=int, default = 1)
@click.argument('step_per_epoch', type=int, default = 100)
@click.argument('windows_size', type=int, default = 2)
@click.argument('model_name', type=str, default = "final_model")
def main(goal, tokens_dir, save_dir, model_dir, nb_token_keep, output_dim, batch, epochs, step_per_epoch, windows_size, model_name):

    logger = logging.getLogger(__name__)

    ## ------ lecture des tokens ------
    logger.info('processTokenize')
    with open('{}/lemma.json'.format(tokens_dir)) as f:
        tokens = json.load(f)

    ## ------ Class WordData ------
    logger.info('Class WordData')
    word_data = WordData(tokens, None, nb_token_keep, False, False)


    ## ------ ContextWordDataLoader ------
    logger.info('ContextWordDataLoader')
    contexte_loader = ContextWordDataLoader(windows_size, word_data, batch, step_per_epoch)


    ## ------ Word2Vec model ------
    logger.info('Word2Vec model')
    word2vec_model = Word2VecModel(model_dir, word_data.getVocabularyLength() + 1, output_dim, windows_size*2)

    ## ------ Word2Vec model: summary ------
    logger.info('Word2Vec model: summary')
    print(word2vec_model.model.summary())

    if goal == 'run':

        ## ------ Trainer ------
        logger.info('Trainer')
        floydhub_log_callback = LambdaCallback(on_epoch_end = lambda epoch, logs: print('{"metric": "accuracy", "value":'+ str(logs['categorical_accuracy']) +' }'))
        trainer = ModelTrainer(model = word2vec_model, data_loader = contexte_loader, epochs =  epochs, callbacks = [floydhub_log_callback])

        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info('Save model before exiting program...')
            trainer.model.save()

    elif goal == 'save_vectors':

        word2vec_model.load(model_name)

        f = open('{}/vectors.txt'.format(save_dir) ,'w')
        f.write('{} {}\n'.format(word_data.getVocabularyLength(), output_dim))

        vectors = word2vec_model.model.get_weights()[0]
        for word, i in word_data.ref_word_to_id.items():
            str_vec = ' '.join(map(str, list(vectors[i, :])))
            f.write('{} {}\n'.format(i, str_vec))
        f.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
