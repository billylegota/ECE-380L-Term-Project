"""ber_curve.py -- Plot BER curve for a model.
"""

import h5py
import numpy as np
import tensorflow as tf
import yaml

import models
import qam_decode


def calculate_ber(model: tf.keras.Model, tone_start: int, tone_end: int, path: str):
    data = h5py.File(path, 'r')

    x = [
        np.array(data['l_ltf_1_gain']),
        np.array(data['l_ltf_2_gain']),
        np.array(data['he_ltf_gain']),
        np.array(data['he_ppdu_pilot_gain']),
        np.array(data['rx_he_ppdu_data'][:, tone_start:tone_end]),
    ]

    y = np.array(data['tx_he_ppdu_data'][:, tone_start:tone_end])
    y_hat = model.predict(x)

    bits = qam_decode.decode(y, 7)
    bits_hat = qam_decode.decode(y_hat, 7)

    return np.mean(bits_hat != bits)


if __name__ == '__main__':
    config_path = 'config/dense_model.yaml'
    weights_path = r'C:\Users\billy\PycharmProjects\ECE 380L Term Project' \
                   r'\output\64_QAM\DenseModel\2021-11-28\20-26-46\weights.h5'
    data_path = 'data_exploration/test_indoor_45dB_flat_engineered.h5'

    with open(config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    model = models.MODELS[model_config['model']['name']](**model_config['model']['parameters']).build()
    model.load_weights(weights_path)

    ber = calculate_ber(model, 0, 18, data_path)
    print(ber)
