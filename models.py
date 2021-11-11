"""model.py -- DeepWiPHY model definitions.
"""

import abc

import tensorflow as tf


Input = tf.keras.layers.InputLayer


class Model(metaclass=abc.ABCMeta):
    def __init__(self, num_pred_tones: int):
        self._num_pred_tones = num_pred_tones

    def build(self) -> tf.keras.Model:
        rx_l_ltf_1 = tf.keras.layers.Input(
            dtype='complex64',
            shape=(64,),
            name='rx_l_ltf_1'
        )

        rx_l_ltf_2 = tf.keras.layers.Input(
            dtype='complex64',
            shape=(64,),
            name='rx_l_ltf_2'
        )

        rx_he_ltf_data = tf.keras.layers.Input(
            dtype='complex64',
            shape=(234,),
            name='he_ltf_data'
        )

        rx_he_ltf_pilot = tf.keras.layers.Input(
            dtype='complex64',
            shape=(8,),
            name='he_ltf_pilot'
        )

        rx_data = tf.keras.layers.Input(
            dtype='complex64',
            shape=(self._num_pred_tones,),
            name='rx_data'
        )

        rx_pilot = tf.keras.layers.Input(
            dtype='complex64',
            shape=(8,),
            name='rx_pilot'
        )

        tx_pilot = tf.keras.layers.Input(
            dtype='complex64',
            shape=(8,),
            name='tx_pilot'
        )

        inputs = [
            rx_l_ltf_1,
            rx_l_ltf_2,
            rx_he_ltf_data,
            rx_he_ltf_pilot,
            rx_data,
            rx_pilot,
            tx_pilot
        ]

        output = self._output(rx_l_ltf_1, rx_l_ltf_2, rx_he_ltf_data, rx_he_ltf_pilot, rx_data, rx_pilot, tx_pilot)

        return tf.keras.Model(inputs=inputs, outputs=output)

    @abc.abstractmethod
    def _output(self, rx_l_ltf_1: Input, rx_l_ltf_2: Input, rx_he_ltf_data: Input, rx_he_ltf_pilot: Input,
                rx_data: Input, rx_pilot: Input, tx_pilot: Input) -> tf.keras.layers.Layer:
        raise NotImplementedError


# noinspection DuplicatedCode
class DenseModel(Model):
    # TODO: Look into a better way of providing arguments here so we don't have to use **kwargs.
    def __init__(self, layers: int, units: int, activation: str, dropout: float, num_pred_tones: int, **_):
        super().__init__(num_pred_tones)
        self._layers = layers
        self._units = units
        self._activation = activation
        self._dropout = dropout
        self._num_pred_tones = num_pred_tones

    def _output(self, rx_l_ltf_1: Input, rx_l_ltf_2: Input, rx_he_ltf_data: Input, rx_he_ltf_pilot: Input,
                rx_data: Input, rx_pilot: Input, tx_pilot: Input) -> tf.keras.layers.Layer:
        inputs = [
            rx_l_ltf_1,
            rx_l_ltf_2,
            rx_he_ltf_data,
            rx_he_ltf_pilot,
            rx_data,
            rx_pilot,
            tx_pilot
        ]

        n = tf.concat(values=[f(x) for f in [tf.math.real, tf.math.imag] for x in inputs], axis=-1)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation=self._activation)(n)
            n = tf.keras.layers.Dropout(rate=self._dropout)(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        correction = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        return tf.keras.layers.multiply([rx_data, correction])


# noinspection DuplicatedCode
class DenseParallelModel(DenseModel):
    def _output(self, rx_l_ltf_1: Input, rx_l_ltf_2: Input, rx_he_ltf_data: Input, rx_he_ltf_pilot: Input,
                rx_data: Input, rx_pilot: Input, tx_pilot: Input) -> tf.keras.layers.Layer:
        inputs = [
            rx_l_ltf_1,
            rx_l_ltf_2,
            rx_he_ltf_data,
            rx_he_ltf_pilot,
            rx_data,
            rx_pilot,
            tx_pilot
        ]

        # Equalizer.
        n = tf.concat(values=[f(x) for f in [tf.math.real, tf.math.imag] for x in inputs], axis=-1)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation=self._activation)(n)
            n = tf.keras.layers.Dropout(rate=self._dropout)(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        equalizer = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        # CPE/SRO correction.
        n = tf.concat(values=[f(x) for f in [tf.math.real, tf.math.imag] for x in inputs], axis=-1)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation=self._activation)(n)
            n = tf.keras.layers.Dropout(rate=self._dropout)(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        cpe_sro_correction = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        return tf.keras.layers.multiply([rx_data, equalizer, cpe_sro_correction])


# noinspection DuplicatedCode
class DenseParallelModelSlim(DenseModel):
    def _output(self, rx_l_ltf_1: Input, rx_l_ltf_2: Input, rx_he_ltf_data: Input, rx_he_ltf_pilot: Input,
                rx_data: Input, rx_pilot: Input, tx_pilot: Input) -> tf.keras.layers.Layer:
        inputs = [
            rx_l_ltf_1,
            rx_l_ltf_2,
            rx_he_ltf_data,
            rx_he_ltf_pilot,
            rx_pilot,
            tx_pilot
        ]

        # Equalizer.
        n = tf.concat(values=[f(x) for f in [tf.math.real, tf.math.imag] for x in inputs], axis=-1)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation=self._activation)(n)
            n = tf.keras.layers.Dropout(rate=self._dropout)(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        equalizer = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        # CPE/SRO correction.
        n = tf.concat(values=[f(x) for f in [tf.math.real, tf.math.imag] for x in inputs], axis=-1)
        for i in range(self._layers):
            n = tf.keras.layers.Dense(units=self._units, activation=self._activation)(n)
            n = tf.keras.layers.Dropout(rate=self._dropout)(n)

        real = tf.keras.layers.Dense(self._num_pred_tones)(n)
        imag = tf.keras.layers.Dense(self._num_pred_tones)(n)

        cpe_sro_correction = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([real, imag])

        return tf.keras.layers.multiply([rx_data, equalizer, cpe_sro_correction])


# noinspection DuplicatedCode
class ConvolutionalModel(Model):
    def __init__(self, filters: int, kernel_size: int, layers: int, units: int, num_pred_tones: int, **_):
        super().__init__(num_pred_tones)
        self._filters = filters
        self._kernel_size = kernel_size
        self._layers = layers
        self._units = units
        self._num_pred_tones = num_pred_tones

    # noinspection DuplicatedCode
    def _output(self, rx_l_ltf_1: Input, rx_l_ltf_2: Input, rx_he_ltf_data: Input, rx_he_ltf_pilot: Input,
                rx_data: Input, rx_pilot: Input, tx_pilot: Input) -> tf.keras.layers.Layer:
        inputs = [
            rx_l_ltf_1,
            rx_l_ltf_2,
            rx_he_ltf_data,
            rx_he_ltf_pilot,
            rx_data,
            rx_pilot,
            tx_pilot
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

        return tf.keras.layers.multiply([rx_data, correction])


MODELS = {
    'DenseModel': DenseModel,
    'DenseParallelModel': DenseParallelModel,
    'DenseParallelModelSlim': DenseParallelModelSlim,
    'ConvolutionalModel': ConvolutionalModel
}
