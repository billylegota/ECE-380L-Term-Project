"""train.py -- Main training code.
"""

import argparse
import logging
import os
import pickle

from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import yaml

import models
import util


def load_dataset(path: str, tone_start: int, tone_end: int, batch_size: int,
                 shuffle_buffer_size: int) -> tf.data.Dataset:
    """Load a flattened dataset from the specified HDF5 file.
    :param path: path to the flattened dataset.
    :param tone_start: start index of tones.
    :param tone_end: stop index of tones.
    :param batch_size: batch size.
    :param shuffle_buffer_size: shuffle buffer size.
    :return: dataset with batching and shuffling already applied.
    """
    num_pred_tones = tone_end - tone_start + 1

    l_ltf_1_gain = tfio.IODataset.from_hdf5(path, '/l_ltf_1_gain')
    l_ltf_2_gain = tfio.IODataset.from_hdf5(path, '/l_ltf_2_gain')
    he_ltf_gain = tfio.IODataset.from_hdf5(path, '/he_ltf_gain')
    he_ppdu_pilot_gain = tfio.IODataset.from_hdf5(path, '/he_ppdu_pilot_gain')
    rx_he_ppdu_data = tfio.IODataset.from_hdf5(path, '/rx_he_ppdu_data')
    tx_he_ppdu_data = tfio.IODataset.from_hdf5(path, '/tx_he_ppdu_data')

    if num_pred_tones != 234:
        rx_he_ppdu_data = rx_he_ppdu_data.map(
            lambda x: x[tone_start:tone_end], num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        tx_he_ppdu_data = tx_he_ppdu_data.map(
            lambda x: x[tone_start:tone_end], num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    inputs = tfio.IODataset.zip((
        l_ltf_1_gain,
        l_ltf_2_gain,
        he_ltf_gain,
        he_ppdu_pilot_gain,
        rx_he_ppdu_data
    ))

    outputs = tx_he_ppdu_data

    # TODO: https://determined.ai/blog/tf-dataset-the-bad-parts/
    #       https://github.com/determined-ai/yogadl
    # FIXME: This dataset is not shuffled and Keras will not shuffle it. Shuffle in Keras can only shuffle the
    #        batches given to it, not the elements before they are batched. This means that we will always have the same
    #        batches (albeit in a different order if we use shuffle=True) which will probably hurt generalization. The
    #        best solution is probably to have some hybrid random access / sequential access data pipeline that performs
    #        the shuffling at the random access layer not sequential access layer. That way we can avoid the issues
    #        related to tf.data.Dataset.shuffle.
    # TODO: Figure out how big the buffer should be when shuffling for good results.
    dataset = tfio.IODataset.zip((inputs, outputs)) \
        .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True) \
        .batch(batch_size=batch_size, drop_remainder=False) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', help='seed for Numpy RNG', default=42, type=int)
    parser.add_argument('--output_dir', help='output directory', default=None, type=str)

    parser.add_argument('--mcs', help='modulation and coding scheme', choices=[7, 9, 11], default=7, type=int)
    parser.add_argument('--tone_cluster', help='index of the tone cluster (one indexed)', default=1, type=int)

    parser.add_argument('--model_config', help='model config file', default='config/default.yaml', type=str)
    parser.add_argument('--load_checkpoint', help='load weights from checkpoint file', default=None, type=str)

    parser.add_argument('--train', help='training dataset', default=['default.h5'], type=str, nargs='*')
    parser.add_argument('--test', help='testing dataset', default='default.h5', type=str)
    parser.add_argument('--batch_size', help='batch size', default=1000, type=int)
    parser.add_argument('--shuffle_buffer_size', help='shuffle buffer size', default=10000, type=int)
    parser.add_argument('--epochs', help='training epochs', default=100, type=int)
    parser.add_argument('--loss_function', help='loss function', default='mean_squared_error', type=str)
    parser.add_argument('--learning_rate', help='learning rate', default=0.00001, type=float)
    parser.add_argument('--cpu', help='CPU only operation (disable GPU)', action='store_true')
    args = parser.parse_args()

    # Disable info messages from TensorFlow.
    tf.get_logger().setLevel(logging.WARNING)

    # Seed the PRNG.
    # FIXME: This (1) does not (reliably) set all the PRNG seeds and (2) is deprecated according to Numpy docs.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Disable GPU acceleration.
    if args.cpu:
        util.disable_gpu()

    # Build the model.
    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)

    model = models.MODELS[model_config['model']['name']](**model_config['model']['parameters']).build()
    adam = tf.keras.optimizers.Adam(lr=args.learning_rate)
    model.compile(optimizer=adam, loss=args.loss_function)

    # Load the model weights.
    if args.load_checkpoint is not None:
        model.load_weights(args.load_checkpoint)

    # Create output directory.
    mcs_strings = {
        7: '64_QAM',
        9: '256_QAM',
        11: '1024_QAM'
    }

    mcs_string = mcs_strings[args.mcs]

    output_dir = f'output/{mcs_string}/{model_config["model"]["name"]}/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}'
    output_dir = args.output_dir or output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset.
    num_pred_tones = model_config['model']['parameters']['num_pred_tones']
    tone_start = (args.tone_cluster - 1) * num_pred_tones
    tone_end = tone_start + num_pred_tones

    datasets = [
        load_dataset(path, tone_start, tone_end, args.batch_size, args.shuffle_buffer_size) for path in args.train
    ]

    # Training.
    result = model.fit(
        x=datasets[0],
        shuffle=False,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{output_dir}/weights_epoch_{{epoch}}.h5',
                save_weights_only=True,
                save_freq='epoch',
                save_best_only=False
            )
        ],
    )
    history = result.history

    # Save arguments.
    path = f'{output_dir}/arguments.pkl'
    with open(path, 'wb') as file:
        pickle.dump(vars(args), file)

    # Save model.
    path = f'{output_dir}/model.json'
    with open(path, 'w') as file:
        file.write(model.to_json())

    # Save weights.
    path = f'{output_dir}/weights.h5'
    model.save_weights(path)
    print('Saved model weights to disk')

    # Save history.
    path = f'{output_dir}/history.pkl'
    with open(path, 'wb') as file:
        pickle.dump(history, file)


if __name__ == '__main__':
    main()
