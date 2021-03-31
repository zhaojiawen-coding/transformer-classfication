import argparse
import os
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time

from tensorflow.keras.callbacks import EarlyStopping

from data_helper import data_loader
from model.text_cnn import TextCNN
from utils.metrics import micro_f1, macro_f1


def train(X_train, X_test, y_train, y_test, params, save_path):
    print("\nTrain...")
    model = build_model(params)

    # parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    # parallel_model.compile(tf.optimizers.Adam(learning_rate=args.learning_rate), loss='binary_crossentropy',
    #                        metrics=[micro_f1, macro_f1])
    # keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(args.results_dir, timestamp, "model.pdf"))
    # y_train = tf.one_hot(y_train, args.num_classes)
    # tb_callback = keras.callbacks.TensorBoard(os.path.join(args.results_dir, timestamp, 'log/'),
    #                                           histogram_freq=0.1, write_graph=True,
    #                                           write_grads=True, write_images=True,
    #                                           embeddings_freq=0.5, update_freq='batch')

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')

    history = model.fit(X_train, y_train,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        workers=params.workers,
                        use_multiprocessing=True,
                        callbacks=[early_stopping],
                        validation_data=(X_test, y_test))

    print("\nSaving model...")
    keras.models.save_model(model, save_path)
    pprint(history.history)


def build_model(params):
    if params.model == 'cnn':
        model = TextCNN(max_sequence_length=params.padding_size, max_token_num=params.vocab_size,
                        embedding_dim=params.embed_size,
                        output_dim=params.num_classes)
        model.compile(tf.optimizers.Adam(learning_rate=params.learning_rate),
                      loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])

    else:

        pass

    model.summary()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN train project.')

    parser.add_argument('--model', default='cnn')

    parser.add_argument('-t', '--test_sample_percentage', default=0.1, type=float,
                        help='The fraction of test data.(default=0.1)')
    parser.add_argument('-p', '--padding_size', default=300, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-e', '--embed_size', default=512, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=128, type=int,
                        help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float,
                        help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float,
                        help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.05, type=float,
                        help='The fraction of validation.(default=0.05)')
    parser.add_argument('--results_dir', default='results/', type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=./results/)')

    parser.add_argument('--data_path', default='data/baidu_95.csv', type=str,
                        help='data path')
    parser.add_argument('--vocab_save_dir', default='data/', type=str,
                        help='data path')

    parser.add_argument('--workers', default=32, type=int,
                        help='use worker count')

    params = parser.parse_args()
    print('Parameters:', params, '\n')

    if not os.path.exists(params.results_dir):
        os.mkdir(params.results_dir)
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(params.results_dir, timestamp))
    os.mkdir(os.path.join(params.results_dir, timestamp, 'log/'))

    X_train, X_test, y_train, y_test = data_loader(params)

    train(X_train, X_test, y_train, y_test, params, os.path.join(params.results_dir, 'TextCNN.h5'))
