"""split_synthetic.py -- Split synthetic dataset into sub channel models.
"""

import os
import pickle

import h5py
import numpy as np
import tqdm


if __name__ == '__main__':
    # You should only need to modify these values.
    source = r'D:\EE 364D\dataset\synthetic_data\train_indoor.h5'
    definition_path = r'dataset\train_indoor.pkl'
    destination_format = r'D:\EE 364D\dataset\synthetic_data\train_indoor\train_indoor_channel_{0}.h5'

    with open(definition_path, 'rb') as file:
        definition = pickle.load(file)

    os.makedirs(os.path.dirname(destination_format), exist_ok=True)

    data = h5py.File(source, 'r')
    names = set(name for name, _ in definition)
    channels = {name: h5py.File(destination_format.format(name), 'w') for name in names}

    start = 0
    for name, count in tqdm.tqdm(definition, total=len(definition)):
        fields = {
            field: np.array(data[field][start:start + count]) for field in data
        }
        start += count

        for key, value in fields.items():
            if key not in channels[name]:
                channels[name].create_dataset(name=key, data=value, maxshape=(None, *value.shape[1:]), chunks=True)
            else:
                channels[name][key].resize(channels[name][key].shape[0] + value.shape[0], axis=0)
                channels[name][key][-value.shape[0]:] = value

    for snr in [10, 15, 20, 25, 30, 35, 40, 45]:
        # You should only need to modify these values.
        source = rf'D:\EE 364D\dataset\synthetic_data\test_indoor_{snr}dB.h5'
        definition_path = r'dataset\test_indoor.pkl'
        destination_format = rf'D:\EE 364D\dataset\synthetic_data\test_indoor_{snr}dB\test_indoor_{snr}dB_channel_{{0}}.h5'

        with open(definition_path, 'rb') as file:
            definition = pickle.load(file)

        os.makedirs(os.path.dirname(destination_format), exist_ok=True)

        data = h5py.File(source, 'r')
        names = set(name for name, _ in definition)
        channels = {name: h5py.File(destination_format.format(name), 'w') for name in names}

        start = 0
        for name, count in tqdm.tqdm(definition, total=len(definition)):
            fields = {
                field: np.array(data[field][start:start + count]) for field in data
            }
            start += count

            for key, value in fields.items():
                if key not in channels[name]:
                    channels[name].create_dataset(name=key, data=value, maxshape=(None, *value.shape[1:]), chunks=True)
                else:
                    channels[name][key].resize(channels[name][key].shape[0] + value.shape[0], axis=0)
                    channels[name][key][-value.shape[0]:] = value

        print(f'Processed a total of {start} packets with {len(channels)} channel models.')
