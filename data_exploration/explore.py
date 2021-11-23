import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import complex_pca


def plot_pca_variance_curve(x: np.ndarray, title: str = 'PCA -- Variance Explained Curve') -> None:
    pca = complex_pca.ComplexPCA(n_components=x.shape[1])
    pca.fit(x)

    plt.figure()
    plt.plot(range(1, x.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_) / np.sum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Proportion of Variance Captured')
    plt.title(title)
    plt.grid(True)


# noinspection DuplicatedCode
def load_and_transform_data(data_path: str, constant_features_path: str = None) -> np.ndarray:
    if constant_features_path is None:
        constant_features_path = '../data_preprocessing/constant_features.mat'

    # Load dataset and constant features.
    data = h5py.File(data_path, 'r')
    constant_features = scipy.io.loadmat(constant_features_path, squeeze_me=True)
    constant_features = constant_features['constant']

    # L-LTF extraction.
    rx_l_ltf_1 = np.array(data['rx_l_ltf_1'])
    rx_l_ltf_2 = np.array(data['rx_l_ltf_2'])

    tx_l_ltf = constant_features['txLltfFftOut'][()]

    rx_l_ltf_1_trimmed = rx_l_ltf_1[:, tx_l_ltf != 0]
    rx_l_ltf_2_trimmed = rx_l_ltf_2[:, tx_l_ltf != 0]
    tx_l_ltf_trimmed = tx_l_ltf[tx_l_ltf != 0]

    l_ltf_1_trimmed_gain = rx_l_ltf_1_trimmed / tx_l_ltf_trimmed
    l_ltf_2_trimmed_gain = rx_l_ltf_2_trimmed / tx_l_ltf_trimmed

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

    he_ltf_trimmed_gain = rx_he_ltf_trimmed / tx_he_ltf_trimmed

    # Combine data.
    return np.hstack([
        he_ltf_trimmed_gain,
        l_ltf_1_trimmed_gain,
        l_ltf_2_trimmed_gain
    ])


def main() -> None:
    use_real_data = False
    data_path = 'flat_train_real_data_42dB.h5' if use_real_data else 'test_indoor_45dB_flat.h5'
    constant_features_path = '../data_preprocessing/constant_features.mat'

    data = h5py.File(data_path, 'r')
    constant_features = scipy.io.loadmat(constant_features_path, squeeze_me=True)
    constant_features = constant_features['constant']

    # Number of data points to use.
    n = 1000

    # L-LTF extraction.
    l_ltf_size = 64

    rx_l_ltf_1 = np.array(data['rx_l_ltf_1'][0:n - 1, :])
    rx_l_ltf_2 = np.array(data['rx_l_ltf_2'][0:n - 1, :])

    tx_l_ltf = constant_features['txLltfFftOut'][()]

    rx_l_ltf_1_trimmed = rx_l_ltf_1[:, tx_l_ltf != 0]
    rx_l_ltf_2_trimmed = rx_l_ltf_2[:, tx_l_ltf != 0]
    tx_l_ltf_trimmed = tx_l_ltf[tx_l_ltf != 0]

    l_ltf_1_trimmed_gain = rx_l_ltf_1_trimmed / tx_l_ltf_trimmed
    l_ltf_2_trimmed_gain = rx_l_ltf_2_trimmed / tx_l_ltf_trimmed

    # HE-LTF extraction.
    he_ltf_data_indices = constant_features['iMDataTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_pilot_indices = constant_features['iMPilotTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_size = 256

    rx_he_ltf_data = np.array(data['rx_he_ltf_data'][0:n - 1, :])
    rx_he_ltf_pilot = np.array(data['rx_he_ltf_pilot'][0:n - 1, :])
    rx_he_ltf = np.zeros((rx_he_ltf_data.shape[0], he_ltf_size), dtype=complex)
    rx_he_ltf[:, he_ltf_data_indices] = rx_he_ltf_data
    rx_he_ltf[:, he_ltf_pilot_indices] = rx_he_ltf_pilot

    tx_he_ltf = constant_features['txHeltfFftOut'][()]

    rx_he_ltf_trimmed = rx_he_ltf[:, tx_he_ltf != 0]
    tx_he_ltf_trimmed = tx_he_ltf[tx_he_ltf != 0]

    he_ltf_trimmed_gain = rx_he_ltf_trimmed / tx_he_ltf_trimmed

    # Frequency domain.
    f_rx_he_ltf = np.linspace(0, 1, he_ltf_size)
    f_rx_he_ltf_trimmed = f_rx_he_ltf[tx_he_ltf != 0]

    f_l_ltf = np.linspace(0, 1, l_ltf_size)
    f_l_ltf_trimmed = f_l_ltf[tx_l_ltf != 0]

    # Channel instance to use.
    i = 1

    # Make plots.
    plot_constellation = False
    plot_magnitude = True
    plot_phase = True
    plot_pca = False
    plot_mean_magnitude = False

    if plot_constellation:
        plt.figure()
        plt.scatter(np.real(he_ltf_trimmed_gain[i, :]), np.imag(he_ltf_trimmed_gain[i, :]))
        plt.scatter(np.real(l_ltf_1_trimmed_gain[i, :]), np.imag(l_ltf_1_trimmed_gain[i, :]))
        plt.scatter(np.real(l_ltf_2_trimmed_gain[i, :]), np.imag(l_ltf_2_trimmed_gain[i, :]))
        plt.xlabel('In-phase Component')
        plt.ylabel('Quadrature Component')
        plt.title('Channel Gain Constellation')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2'])
        plt.grid()

    if plot_magnitude:
        plt.figure()
        plt.scatter(f_rx_he_ltf_trimmed, 20 * np.log10(np.abs(he_ltf_trimmed_gain[i, :])))
        plt.scatter(f_l_ltf_trimmed, 20 * np.log10(np.abs(l_ltf_1_trimmed_gain[i, :])))
        plt.scatter(f_l_ltf_trimmed, 20 * np.log10(np.abs(l_ltf_2_trimmed_gain[i, :])))
        plt.xlabel(r'$f$ (normalized)')
        plt.ylabel(r'$|H|^2$ (dB)')
        plt.title('Channel Gain')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2'])
        plt.grid()

    if plot_phase:
        plt.figure()
        plt.scatter(f_rx_he_ltf_trimmed, np.unwrap(np.angle(he_ltf_trimmed_gain[i, :])) / np.pi)
        plt.scatter(f_l_ltf_trimmed, np.unwrap(np.angle(l_ltf_1_trimmed_gain[i, :])) / np.pi)
        plt.scatter(f_l_ltf_trimmed, np.unwrap(np.angle(l_ltf_2_trimmed_gain[i, :])) / np.pi)
        plt.xlabel(r'$f$ (normalized)')
        plt.ylabel(r'$\angle H$ ($\times \pi$)')
        plt.title('Channel Phase')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2'])
        plt.grid()

    if plot_pca:
        plot_pca_variance_curve(he_ltf_trimmed_gain, 'HE-LTF Trimmed Gain')
        plot_pca_variance_curve(rx_he_ltf, 'HE-LTF Raw')
        plot_pca_variance_curve(l_ltf_1_trimmed_gain, 'L-LTF-1 Trimmed Gain')
        plot_pca_variance_curve(rx_l_ltf_1, 'L-LTF-1 Raw')
        plot_pca_variance_curve(l_ltf_2_trimmed_gain, 'L-LTF-2 Trimmed Gain')
        plot_pca_variance_curve(rx_l_ltf_2, 'L-LTF-2 Raw')
        plot_pca_variance_curve(np.hstack([  # TODO: integrate this.
            he_ltf_trimmed_gain,
            l_ltf_1_trimmed_gain,
            l_ltf_2_trimmed_gain
        ]), 'All data')

    if plot_mean_magnitude:
        plt.figure()
        x = f_rx_he_ltf_trimmed
        y = np.mean(np.abs(he_ltf_trimmed_gain), axis=0)
        s = np.std(np.abs(he_ltf_trimmed_gain), axis=0)
        plt.plot(x, 20 * np.log10(y))
        plt.fill_between(x, 20 * np.log10(y - s), 20 * np.log10(y + s), alpha=0.5)
        plt.xlabel(r'$f$ (normalized)')
        plt.ylabel(r'$|H|^2$ (dB)')
        plt.title('Mean Channel Gain')
        plt.legend([r'$\mu$', r'$\pm\sigma$'])
        plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
