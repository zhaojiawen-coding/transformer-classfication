import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

from data_helper import data_loader
from utils.metrics import micro_f1, macro_f1


def test(model, x_test, y_true):
    print("Test...")
    y_pred = model.predict(x_test)

    y_true = tf.constant(y_true, tf.float32)
    y_pred = tf.constant(y_pred, tf.float32)
    print(micro_f1(y_true, y_pred))
    print(macro_f1(y_true, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('-p', '--padding_size', default=200, type=int, help='Padding size of sentences.(default=200)')
    parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=95)')
    parser.add_argument('--results_dir', default='results/', type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=results/)')
    params = parser.parse_args()
    print('Parameters:', params)

    X_train, X_test, y_train, y_test = data_loader(params)
    print("Loading model...")
    model = load_model(os.path.join(params.results_dir, 'TextCNN.h5'),
                       custom_objects={'micro_f1': micro_f1, 'macro_f1': macro_f1})
    test(model, X_test, y_test)
