"""train.py -- Main training code.
"""

import argparse
import functools
import logging
import operator
import os
import pickle
import random

from datetime import datetime

import comet_ml
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tqdm
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

    rx_l_ltf_1 = tfio.IODataset.from_hdf5(path, '/rx_l_ltf_1')
    rx_l_ltf_2 = tfio.IODataset.from_hdf5(path, '/rx_l_ltf_2')
    rx_he_ltf_data = tfio.IODataset.from_hdf5(path, '/rx_he_ltf_data')
    rx_he_ltf_pilot = tfio.IODataset.from_hdf5(path, '/rx_he_ltf_pilot')
    rx_data = tfio.IODataset.from_hdf5(path, '/rx_data')
    rx_pilot = tfio.IODataset.from_hdf5(path, '/rx_pilot')
    tx_data = tfio.IODataset.from_hdf5(path, '/tx_data')
    tx_pilot = tfio.IODataset.from_hdf5(path, '/tx_pilot')

    if num_pred_tones != 234:
        rx_data = rx_data.map(lambda x: x[tone_start:tone_end], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tx_data = tx_data.map(lambda x: x[tone_start:tone_end], num_parallel_calls=tf.data.experimental.AUTOTUNE)

    inputs = tfio.IODataset.zip((
        rx_l_ltf_1,
        rx_l_ltf_2,
        rx_he_ltf_data,
        rx_he_ltf_pilot,
        rx_data,
        rx_pilot,
        tx_pilot
    ))

    outputs = tx_data

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

    parser.add_argument('--meta_algo', help='use the specified meta learning algorithm',
                        choices=[None, 'reptile', 'maml'], default=None, type=str)
    parser.add_argument('--meta_steps', help='number of meta learning steps to perform', default=10, type=int)
    parser.add_argument('--mini_batches_per_meta_step', help='number of mini-batches per meta learning step',
                        default=4, type=int)
    parser.add_argument('--meta_step_size', help='meta learning step size or range (start, stop)', default=[0.1],
                        type=float, nargs='*')
    parser.add_argument('--meta_steps_per_save', help='number of meta learning steps between saves', default=10,
                        type=int)

    parser.add_argument('--train', help='training dataset', default=['default.h5'], type=str, nargs='*')
    parser.add_argument('--test', help='testing dataset', default='default.h5', type=str)
    parser.add_argument('--batch_size', help='batch size', default=1000, type=int)
    parser.add_argument('--shuffle_buffer_size', help='shuffle buffer size', default=10000, type=int)
    parser.add_argument('--epochs', help='training epochs', default=100, type=int)
    parser.add_argument('--loss_function', help='loss function', default='mean_squared_error', type=str)
    parser.add_argument('--learning_rate', help='learning rate', default=0.00001, type=float)
    parser.add_argument('--cpu', help='CPU only operation (disable GPU)', action='store_true')
    parser.add_argument('--comet', help='enable Comet ML using given experiment name', default=None, type=str, nargs=2)
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

    # Comet ML setup.
    if args.comet:
        experiment = comet_ml.Experiment(workspace='wblount', project_name=args.comet[0])
        experiment.set_name(args.comet[1])
        experiment.set_os_packages()
        experiment.set_pip_packages()

        os.environ['COMET_GIT_DIRECTORY'] = os.path.abspath(os.path.dirname(__file__))

        experiment.log_parameters({
            'model': model_config['model']['name'],
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'loss_function': args.loss_function,
            'learning_rate': args.learning_rate,
            'model_parameters': model_config['model']['parameters']
        })

        experiment.log_others({
            'seed': args.seed,
            'mcs': args.mcs,
            'num_pred_tones': num_pred_tones,
            'tone_cluster': args.tone_cluster,
            'cpu': args.cpu,
            'training_dataset': args.train,
            'test_dataset': args.test
        })

    # Training.
    if args.meta_algo is None:
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

    elif args.meta_algo == 'reptile':
        history = {
            'dataset': []
        }

        if len(args.meta_step_size) == 1:
            meta_step_size_initial = args.meta_step_size[0]
            meta_step_size_final = args.meta_step_size[0]
        else:
            meta_step_size_initial = args.meta_step_size[0]
            meta_step_size_final = args.meta_step_size[1]

        progress = tqdm.trange(args.meta_steps)
        for i in progress:
            fraction_done = i / args.meta_steps
            current_meta_step_size = fraction_done * meta_step_size_final + (1 - fraction_done) * meta_step_size_initial

            index = random.randrange(len(datasets))
            gradients = []
            old_weights = np.array(model.get_weights())
            for _ in range(args.mini_batches_per_meta_step):
                result = model.fit(
                    x=datasets[index],
                    shuffle=False,
                    epochs=1,
                    steps_per_epoch=1,
                    verbose=0
                )
                new_weights = np.array(model.get_weights())
                gradients.append(old_weights - new_weights)
                model.set_weights(old_weights)

                for key, value in result.history.items():
                    if key in history:
                        history[key] += value
                    else:
                        history[key] = value

                history['dataset'] += [index] * args.mini_batches_per_meta_step

                loss = result.history['loss'][0]
                progress.set_description(f'Loss: {loss:.4g}', True)

            gradient = functools.reduce(operator.add, gradients)
            new_weights = old_weights - current_meta_step_size * gradient
            model.set_weights(new_weights)

            if (i + 1) % args.meta_steps_per_save == 0:
                model.save_weights(f'{output_dir}/weights_meta_step_{i + 1}.h5')
                model.save_weights('output/meta_weights_latest.h5')

    elif args.meta_algo == 'maml':
        raise NotImplementedError('MAML meta learning algorithm has not been implemented!')

    else:
        raise ValueError(f'{args.meta_algo} is not a valid meta learning algorithm.')

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
