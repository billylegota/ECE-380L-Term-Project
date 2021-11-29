"""extract_features.py -- Performs feature engineering and extraction on a flattened HDF5 dataset.
"""

import h5py
import numpy as np
import scipy.io


# noinspection DuplicatedCode
def extract_features(source: str, dest: str, constant_features_path: str = 'constant_features.mat') -> None:
    """Performs feature engineering and extraction on a flattened HDF5 dataset.
    :param source: path to original dataset.
    :param dest: path to store new dataset.
    :param constant_features_path: path to constant features.
    """
    data = h5py.File(source, 'r')
    constant_features = scipy.io.loadmat(constant_features_path, squeeze_me=True)
    constant_features = constant_features['constant']

    # L-LTF extraction.
    rx_l_ltf_1 = np.array(data['rx_l_ltf_1'])
    rx_l_ltf_2 = np.array(data['rx_l_ltf_2'])

    tx_l_ltf = constant_features['txLltfFftOut'][()]

    rx_l_ltf_1_trimmed = rx_l_ltf_1[:, tx_l_ltf != 0]
    rx_l_ltf_2_trimmed = rx_l_ltf_2[:, tx_l_ltf != 0]
    tx_l_ltf_trimmed = tx_l_ltf[tx_l_ltf != 0]

    l_ltf_1_gain = rx_l_ltf_1_trimmed / tx_l_ltf_trimmed
    l_ltf_2_gain = rx_l_ltf_2_trimmed / tx_l_ltf_trimmed

    # HE-LTF extraction.
    he_ltf_data_indices = constant_features['iMDataTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_pilot_indices = constant_features['iMPilotTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_size = 256

    rx_he_ltf_data = np.array(data['rx_he_ltf_data'])
    rx_he_ltf_pilot = np.array(data['rx_he_ltf_pilot'])
    rx_he_ltf = np.zeros((rx_he_ltf_data.shape[0], he_ltf_size), dtype=complex)
    rx_he_ltf[:, he_ltf_data_indices] = rx_he_ltf_data
    rx_he_ltf[:, he_ltf_pilot_indices] = rx_he_ltf_pilot

    tx_he_ltf = constant_features['txHeltfFftOut'][()]

    rx_he_ltf_trimmed = rx_he_ltf[:, tx_he_ltf != 0]
    tx_he_ltf_trimmed = tx_he_ltf[tx_he_ltf != 0]

    he_ltf_gain = rx_he_ltf_trimmed / tx_he_ltf_trimmed

    # Data and pilot extraction.
    rx_he_ppdu_pilot = np.array(data['rx_pilot'])
    tx_he_ppdu_pilot = np.array(data['tx_pilot'])
    he_ppdu_pilot_gain = rx_he_ppdu_pilot / tx_he_ppdu_pilot

    rx_he_ppdu_data = np.array(data['rx_data'])
    tx_he_ppdu_data = np.array(data['tx_data'])

    fields = {
        'l_ltf_1_gain': l_ltf_1_gain,
        'l_ltf_2_gain': l_ltf_2_gain,
        'he_ltf_gain': he_ltf_gain,
        'he_ppdu_pilot_gain': he_ppdu_pilot_gain,
        'rx_he_ppdu_data': rx_he_ppdu_data,
        'tx_he_ppdu_data': tx_he_ppdu_data
    }

    output = h5py.File(dest, 'w')
    for name, data in fields.items():
        output.create_dataset(name=name, data=data)


def main():
    extract_features(
        source='../data_exploration/test_indoor_45dB_flat.h5',
        dest='../data_exploration/test_indoor_45dB_flat_engineered.h5'
    )


if __name__ == '__main__':
    main()
