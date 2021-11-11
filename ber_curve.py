"""ber_curve.py -- Plot BER curve for a model.
"""

import os
import pickle

import h5py
import numpy as np
import tensorflow as tf
import yaml

import models
import qam_decode
import util


def calculate_ber(model: tf.keras.Model, tone_start: int, tone_end: int, path: str):
    data = h5py.File(path, 'r')

    x = [
        np.array(data['rx_l_ltf_1']),
        np.array(data['rx_l_ltf_2']),
        np.array(data['rx_he_ltf_data']),
        np.array(data['rx_he_ltf_pilot']),
        np.array(data['rx_data'][:, tone_start:tone_end]),
        np.array(data['rx_pilot']),
        np.array(data['tx_pilot'])
    ]

    y = np.array(data['tx_data'][:, tone_start:tone_end])
    y_hat = model.predict(x)

    bits = qam_decode.decode(y, 7)
    bits_hat = qam_decode.decode(y_hat, 7)

    return np.mean(bits_hat != bits)


def calculate_ber_wbb(tone_start: int, tone_end: int, path: str):
    data = h5py.File(path, 'r')

    y = np.array(data['tx_data'][:, tone_start:tone_end])
    y_hat = np.array(data['rx_ref_data'][:, tone_start:tone_end])

    bits = qam_decode.decode(y, 7)
    bits_hat = qam_decode.decode(y_hat, 7)

    return np.mean(bits_hat != bits)


def generate_single_data(name: str, config_path: str, weights_path: str, data_channel: str):
    if os.path.exists(f'{name}.pkl'):
        with open(f'{name}.pkl', 'rb') as file:
            return pickle.load(file)

    path_format = r'D:\EE 364D\dataset\synthetic_data\channel_specific\test_indoor_{0}dB\test_indoor_{' \
                  r'0}dB_channel_{1}_flat.h5'

    with open(config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    model = models.MODELS[model_config['model']['name']](**model_config['model']['parameters']).build()
    model.load_weights(weights_path)

    paths = {
        snr: path_format.format(snr, data_channel) for snr in [10, 15, 20, 25, 30, 35, 40, 45]
    }

    snr = list(paths.keys())
    ber = []
    ber_wbb = []
    for path in paths.values():
        ber.append(calculate_ber(model, 0, 18, path))
        ber_wbb.append(calculate_ber_wbb(0, 18, path))

    result = (snr, ber, ber_wbb)

    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(result, file)

    return result


def main():
    util.disable_gpu()

    config_path = r'..\config\dense_parallel_model.yaml'
    data_channel = 'e'

    for train_percent in [1, 10]:
        if train_percent == 1:
            for epochs in [50, 100]:
                name = f'Conventional retrained from 20 epochs at 10 percent on F for {epochs} epochs at ' \
                       f'{train_percent} percent'
                weights_path = f'../output/conventional/conventional_retrain_fair_{train_percent}_percent/' \
                               f'weights_epoch_{epochs}.h5'
                generate_single_data(name, config_path, weights_path, data_channel)
        elif train_percent == 10:
            for epochs in [5, 10]:
                name = f'Conventional retrained from 20 epochs at 10 percent on F for {epochs} epochs at ' \
                       f'{train_percent} percent'
                weights_path = f'../output/conventional/conventional_retrain_fair_{train_percent}_percent/' \
                               f'weights_epoch_{epochs}.h5'
                generate_single_data(name, config_path, weights_path, data_channel)


if __name__ == '__main__':
    generate_single_data(
        name='dense_parallel_e',
        config_path='../config/dense_parallel_model.yaml',
        weights_path='../output/dense_rx_data_comparison/dense_parallel_e/weights.h5',
        data_channel='e'
    )

    generate_single_data(
        name='dense_parallel_e_slim',
        config_path='../config/dense_parallel_model_slim.yaml',
        weights_path='../output/dense_rx_data_comparison/dense_parallel_slim_e/weights.h5',
        data_channel='e'
    )
