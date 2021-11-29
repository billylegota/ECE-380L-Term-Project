"""model.py -- DeepWiPHY model definitions.
"""

import tensorflow as tf


Input = tf.keras.layers.InputLayer


# noinspection DuplicatedCode
class DenseModel:
    def __init__(self, layers: int, units: int, activation: str, dropout: float, num_inputs: int,
                 num_pred_tones: int, **_):
        self._layers = layers
        self._units = units
        self._activation = activation
        self._dropout = dropout
        self._num_inputs = num_inputs
        self._num_pred_tones = num_pred_tones

    def build(self) -> tf.keras.Model:
        # All inputs other than the RX HE-PPDU data tones.
        inputs = tf.keras.layers.Input(
            dtype='complex64',
            shape=(self._num_inputs,),
            name='inputs'
        )

        rx_he_ppdu_data = tf.keras.layers.Input(
            dtype='complex64',
            shape=(self._num_pred_tones,),
            name='rx_he_ppdu_data'
        )

        n = tf.concat(values=[tf.math.real(inputs), tf.math.imag(inputs)], axis=-1)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation=self._activation)(n)
            n = tf.keras.layers.Dropout(rate=self._dropout)(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        correction = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        output = tf.keras.layers.multiply([rx_he_ppdu_data, correction])

        return tf.keras.Model(
            inputs=[
                inputs,
                rx_he_ppdu_data
            ],
            outputs=output
        )


# noinspection DuplicatedCode
class ConvolutionalModel:
    def __init__(self, filters: int, kernel_size: int, layers: int, units: int, num_pred_tones: int, **_):
        self._filters = filters
        self._kernel_size = kernel_size
        self._layers = layers
        self._units = units
        self._num_pred_tones = num_pred_tones

    def build(self) -> tf.keras.Model:
        l_ltf_1_gain = tf.keras.layers.Input(
            dtype='complex64',
            shape=(52,),
            name='l_ltf_1_gain'
        )

        l_ltf_2_gain = tf.keras.layers.Input(
            dtype='complex64',
            shape=(52,),
            name='l_ltf_2_gain'
        )

        he_ltf_gain = tf.keras.layers.Input(
            dtype='complex64',
            shape=(242,),
            name='he_ltf_gain'
        )

        he_ppdu_pilot_gain = tf.keras.layers.Input(
            dtype='complex64',
            shape=(8,),
            name='he_ppdu_pilot_gain'
        )

        rx_he_ppdu_data = tf.keras.layers.Input(
            dtype='complex64',
            shape=(self._num_pred_tones,),
            name='rx_he_ppdu_data'
        )

        inputs = [
            l_ltf_1_gain,
            l_ltf_2_gain,
            he_ltf_gain,
            he_ppdu_pilot_gain,
        ]

        n = tf.concat(values=[f(x) for f in [tf.math.real, tf.math.imag] for x in inputs], axis=-1)
        n = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x))(n)
        n = tf.keras.layers.Conv1D(filters=self._filters, kernel_size=self._kernel_size, activation='elu')(n)
        n = tf.keras.layers.Flatten()(n)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation='tanh')(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        correction = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        output = tf.keras.layers.multiply([rx_he_ppdu_data, correction])

        return tf.keras.Model(
            inputs=[
                l_ltf_1_gain,
                l_ltf_2_gain,
                he_ltf_gain,
                he_ppdu_pilot_gain,
                rx_he_ppdu_data,
            ],
            outputs=output
        )


MODELS = {
    'DenseModel': DenseModel,
    'ConvolutionalModel': ConvolutionalModel,
}
