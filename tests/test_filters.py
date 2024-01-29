import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from meegkit.utils.filters import (
    NonResOscillator,
    ResOscillator,
    locking_based_phase,
)


def test_all_alg(show=True):
    """Replicate the figure from the paper."""
    # Model data, all three algorithms
    npt = 100000
    fs = 100
    s  = generate_multi_comp_data_fr_mod(npt, fs)  # Generate test data
    dt = 1 / fs
    time = np.arange(npt) * dt

    # Plot the test signal's Fourier spectrum
    # f, ax = plt.subplots(1, 1)
    # ax.psd(s, Fs=fs, NFFT=2018*4, noverlap=fs)
    # ax.set_title("Test signal's Fourier spectrum")
    # plt.show()


    # The test signal s and its Hilbert amplitude ah (red); one can see that
    # ah does not represent a good envelope for s. On the contrary, the
    # Hilbert-based phase estimation yields good results, and therefore we take
    # it for the ground truth.
    ht_ampl = np.abs(hilbert(s))  # Hilbert amplitude
    ht_phase = np.angle(hilbert(s))  # Hilbert phase

    lb_phase = locking_based_phase(s, dt, npt)
    lb_phi_dif = phase_difference(ht_phase, lb_phase)

    osc = NonResOscillator(fs, 1.1)
    nr_phase, nr_ampl = osc.transform(s)
    nr_phi_dif = phase_difference(ht_phase, nr_phase)

    osc = ResOscillator(fs, 1.1)
    r_phase, r_ampl = osc.transform(s)
    r_phi_dif = phase_difference(ht_phase, r_phase)

    # Panels (b, c, d) show the difference between the
    # Hilbert phase and causally estimated phases (phi_L, phi_N, phi_R) are
    # obtained by means of the locking-based technique, non-resonant and
    # resonant oscillator, respectively). These panels demonstrate that the
    # output of the developed causal algorithms is very close to the HT-phase.
    # Notice that in (c) we show phi_H - phi_N modulo 2pi, since the phase
    # difference is not bounded : within the first 20 time units the phase
    # difference decreases to -14pi, until it saturates and oscillates around
    # (phi_H - phi_N) = 0.
    f, ax = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(12, 8))
    ax[0, 0].plot(time, s, time, ht_phase, lw=.75)
    ax[0, 0].set_ylabel(r"$s,\phi_H$")
    # ax[0, 0].set_ylim([-2, 3])
    ax[0, 0].set_title("Signal and its Hilbert phase")

    ax[1, 0].plot(time, lb_phi_dif, lw=.75)
    ax[1, 0].axhline(0, color="k", ls=":", zorder=-1)
    ax[1, 0].set_ylabel(r"$\phi_H - \phi_L$")
    ax[1, 0].set_ylim([-np.pi, np.pi])
    ax[1, 0].set_title("Phase locking approach")

    ax[2, 0].plot(time, nr_phi_dif, lw=.75)
    ax[2, 0].axhline(0, color="k", ls=":", zorder=-1)
    ax[2, 0].set_ylabel(r"$\phi_H - \phi_N$")
    ax[2, 0].set_ylim([-np.pi, np.pi])
    ax[2, 0].set_title("Nonresonant oscillator")

    ax[3, 0].plot(time, r_phi_dif, lw=.75)
    ax[3, 0].axhline(0, color="k", ls=":", zorder=-1)
    ax[3, 0].set_ylim([-np.pi, np.pi])
    ax[3, 0].set_ylabel("$\phi_H - \phi_R$")
    ax[3, 0].set_xlabel("Time")
    ax[3, 0].set_title("Resonant oscillator")

    ax[0, 1].plot(time, s, time, ht_ampl, lw=.75)
    ax[0, 1].set_ylabel(r"$s,a_H$")
    ax[0, 1].set_title("Signal and its Hilbert amplitude")

    ax[1, 1].axis("off")

    ax[2, 1].plot(time, s, time, nr_ampl, lw=.75)
    ax[2, 1].set_ylabel(r"$s,a_N$")
    ax[2, 1].set_title("Amplitudes")
    ax[2, 1].set_title("Nonresonant oscillator")

    ax[3, 1].plot(time, s, time, r_ampl, lw=.75)
    ax[3, 1].set_xlabel("Time")
    ax[3, 1].set_ylabel(r"$s,a_R$")
    ax[3, 1].set_title("Resonant oscillator")
    plt.suptitle("Amplitude (right) and phase (left) estimation algorithms")
    plt.tight_layout()
    plt.show()

def phase_difference(phi1, phi2):
    cos_phi_dif = np.cos(phi1) * np.cos(phi2) + np.sin(phi1) * np.sin(phi2)
    sin_phi_dif = np.sin(phi1) * np.cos(phi2) - np.cos(phi1) * np.sin(phi2)
    phi_dif = np.arctan2(sin_phi_dif, cos_phi_dif)

    return phi_dif

def generate_multi_comp_data_fr_mod(npt=40000, fs=100):
    """Generate multi-component data with frequency modulation.

    Returns
    -------
    s : np.ndarray
        Generated data.

    """
    dt = 1 / fs
    t = np.arange(1, npt + 1) * dt
    omega1 = np.sqrt(2) / 30  # Frequency of the first component (slow)
    omega2 = np.sqrt(5) / 60  # Frequency of the second component (fast)
    amp = 1 + 0.95 * np.cos(omega1 * t)
    p = t + 5 * np.sin(omega2 * t)
    s = np.cos(p) + 0.2 * np.cos(2 * p + np.pi / 6) + 0.1 * np.cos(3 * p + np.pi / 3)
    s *= amp

    return s


def generate_noisy_signal(npt=40000, fs=100, noise=0.1):
    """Generate multi-component data with frequency modulation.

    Returns
    -------
    s : np.ndarray
        Generated data.

    """
    rng = np.default_rng(42)
    dt = 1 / fs
    # t = np.arange(1, npt + 1) * dt
    s = generate_multi_comp_data_fr_mod(npt, fs)
    s += rng.random(npt) * 0.1

    return s


if __name__ == "__main__":
    # Run the model_data_all_alg function
    test_all_alg()
