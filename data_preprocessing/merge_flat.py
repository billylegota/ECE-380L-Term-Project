"""merge_flat.py -- Merge flattened HDF5 datasets.
"""

import h5py
import numpy as np
import tqdm


def merge(sources: list, destination: str, batch_size: int = 1000, silent: bool = True) -> int:
    """Merge a number of flattened HDF5 datasets.
    :param sources: paths to files to merge.
    :param destination: path to store subsample of dataset.
    :param batch_size: batch size.
    :param silent: suppress output.
    :return: the number of total symbols.
    """
    # Open destination dataset.
    output = h5py.File(destination, 'w')

    total_symbols = 0

    fields = [
        'rx_l_ltf_1',
        'rx_l_ltf_2',
        'rx_he_ltf_data',
        'rx_he_ltf_pilot',
        'rx_data',
        'rx_pilot',
        'tx_data',
        'tx_pilot'
    ]

    for source in sources:
        # Open source dataset.
        data = h5py.File(source, 'r')

        print(f'Processing {source}:')

        # Create create / grow fields.
        for field in fields:
            if field not in output:
                output.create_dataset(field, shape=data[field].shape, maxshape=(None, *data[field].shape[1:]),
                                      dtype=data[field].dtype)  # , chunks=(batch_size, *data[field].shape[1:]))
            else:
                output[field].resize(output[field].shape[0] + data[field].shape[0], axis=0)

            iterator = zip(
                np.split(np.arange(-data[field].shape[0], 0), data[field].shape[0] // batch_size),
                np.split(np.arange(data[field].shape[0]), data[field].shape[0] // batch_size)
            )

            for i, j in iterator if silent else tqdm.tqdm(iterator, total=data[field].shape[0] // batch_size):
                output[field][i] = data[field][j]

            print(f'\tMerged {data[field].shape[0]} elements from {field}.')

        total_symbols += data[fields[0]].shape[0]

    if not silent:
        print(f'Processed {total_symbols} symbols.')

    return total_symbols


if __name__ == '__main__':
    basedir = r'D:\EE 364D\dataset\synthetic_data\channel_specific\train_indoor'

    merge(
        sources=[rf'{basedir}\train_indoor_channel_{channel_model}_flat.h5' for channel_model in 'abcdef'],
        destination=rf'{basedir}\subsampled\train_indoor_merged_flat.h5',
        batch_size=100,
        silent=False
    )
