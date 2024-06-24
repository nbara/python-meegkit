
"""Real-time phase and amplitude estimation using resonant oscillators.

Notes
-----
While the original paper [1]_ only describes the algorithm for a single
channel, this implementation can handle multiple channels. However, the
algorithm is not optimized for this purpose. Therefore, this code is likely to
be slow for large input arrays (n_channels >> 10), since an individual
oscillator is instantiated for each channel.

.. [1] Rosenblum, M., Pikovsky, A., Kühn, A.A. et al. Real-time estimation of
       phase and amplitude with application to neural data. Sci Rep 11, 18037
       (2021). https://doi.org/10.1038/s41598-021-97560-5
"""
import numpy as np
from scipy.fftpack import fft, fftshift, ifft, ifftshift, next_fast_len
from scipy.signal import butter, freqz

from meegkit.utils.buffer import Buffer


class Device:
    """Measurement device.

    This class implements a measurement device, as describes by Rosenblum et al.

    Parameters
    ----------
    om0 : float
        Oscillator frequency (estimation).
    dt : float
        Sampling interval.
    damping : float
        Damping parameter.
    type : str in {"integrator", "oscillator"}
        Type of the device.
    mu : float
        Integrator parameter (only for type="integrator"). Should be >> target
        frequency of interest.

    """

    def __init__(self, om0, dt, damping, type, mu=500):
        self.om0 = om0
        self.dt = dt
        self.damping = damping
        self.type = type
        self.init_coefs(om0, dt, damping)

        # For the integrator
        if type == "integrator":
            self.mu = mu
            self.edelmu = np.exp(-self.dt / self.mu)

        self.state = dict(x=0, y=0)


    def init_coefs(self, om0, dt, damping):
        """Initialize coefficients for solving oscillator's equations."""
        self.c1, self.c2, self.c3, self.enuDel, self.ealDel, self.eta = \
            init_coefs(om0, dt, damping)
        self.om0 = om0

    def step(self, sprev, s, snew):
        """Perform one step of the oscillator equations.

        Parameters
        ----------
        x : float
            State variable x.
        y : float
            State variable y.
        sprev : float
            Previous measurement.
        s : float
            Current measurement.
        snew : float
            New measurement.

        Returns
        -------
        x : float
            Updated state variable x.
        y : float
            Updated state variable y.
        """
        if self.type == "oscillator":
            x = self.state["x"]
            y = self.state["y"]
            x, y = one_step_oscillator(x, y, self.damping, self.eta, self.enuDel,
                                       self.ealDel, self.c1, self.c2, self.c3,
                                       sprev, s, snew)
            self.state["x"] = x
            self.state["y"] = y

        elif self.type == "integrator":
            x = self.state["x"]
            x = one_step_integrator(x, self.edelmu, self.mu, self.dt,
                                    sprev, s, snew)
            self.state["x"] = x

        return self


class NonResOscillator:
    """Real-time measurement of phase and amplitude using non-resonant oscillator.

    This estimator relies on the resonance effect. The measuring “device”
    consists of two linear damped oscillators. The oscillators' frequency is
    much larger than the frequency of the signal, i.e., the system is far from
    resonance. We choose the damping parameters to ensure that (i) the phase of
    the first linear oscillator equals that of the input and that (ii)
    amplitude of the second one and the input relate by a known constant
    multiplicator. The technique yields both phase and amplitude of the input
    signal.

    This estimator includes an automated frequency-tuning algorithm to adjust
    to the a priori unknown signal frequency.

    For a detailed description, refer to [1]_.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    nu : float
        Rough estimate of the main frequency of interest.

    References
    ----------
    .. [1] Rosenblum, M., Pikovsky, A., Kühn, A.A. et al. Real-time estimation
        of phase and amplitude with application to neural data. Sci Rep 11,
        18037 (2021). https://doi.org/10.1038/s41598-021-97560-5
    """

    def __init__(self, fs=250, nu=1.1, alpha_a=6.0, alpha_p=0.2, update_factor=5):

        # Parameters of the measurement "devices"
        self.dt = 1 / fs  # Sampling interval
        self.nu = nu  # Rough estimate of the tremor frequency
        self.om0 = 5 * nu  # Oscillator frequency (estimation)
        self.alpha_a = alpha_a  # Damping parameter for the "amplitude device"
        self.gamma_a = alpha_a / 2
        self.alpha_p = alpha_p  # Damping parameter for the "phase device"
        self.gamma_p = alpha_p / 2
        self.factor = np.sqrt((self.om0 ** 2 - nu ** 2) ** 2 + (self.alpha_a * nu) ** 2)

        # Update parameters, and precomputed quantities
        self.memory = round(2 * np.pi / self.om0 / self.dt)
        self.update_point = 2 * self.memory
        self.update_step = round(self.memory / update_factor)
        self._tbuf = np.arange(self.memory) * self.dt
        self._Sx = np.sum(self._tbuf)
        self._denom = self.memory * np.sum(self._tbuf ** 2) - self._Sx ** 2

        # Buffer to store past input values
        self.n_channels = None
        self.buffer = None

    def _set_devices(self, n_channels):
        # Set up the phase and amplitude "devices"
        self.adevice = [Device(self.om0, self.dt, self.gamma_a, "oscillator")
                        for _ in range(n_channels)]
        self.pdevice = [Device(self.om0, self.dt, self.gamma_p, "oscillator")
                        for _ in range(n_channels)]


    def transform(self, X):
        """Transform the input signal into phase and amplitude estimates.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_channels)
            The input signal to be transformed.

        Returns
        -------
        phase : ndarray, shape=(n_samples, n_channels)
            Current phase estimate.
        ampl : ndarray, shape=(n_samples, n_channels)
            Current amplitude estimate.
        """
        if self.buffer is None:
            self.n_channels = 1 if X.ndim < 2 else X.shape[1]
            self.buffer = Buffer(self.memory, self.n_channels)
            self.phase_buffer = Buffer(self.memory, self.n_channels)
            self._set_devices(self.n_channels)

        n_samples = X.shape[0]
        phase = np.zeros((n_samples, self.n_channels))
        amp = np.zeros((n_samples, self.n_channels))

        for k in range(n_samples):
            self.buffer.push(X[k])

            if self.buffer.counter < 3:
                continue # Skip the first two samples

            # Amplitude estimation
            spp, sp, s = self.buffer.view(3)[:, 0]

            for ch in range(self.n_channels):
                self.adevice[ch].step(spp, sp, s)
                z = self.adevice[ch].state["y"] / self.nu
                amp[k, ch] = self.factor * \
                    np.sqrt(z ** 2 + self.adevice[ch].state["x"] ** 2)

                # Phase estimation
                self.pdevice[ch].step(spp, sp, s)
                z = self.pdevice[ch].state["y"] / self.nu
                phase[k, ch] = np.arctan2(-z, self.pdevice[ch].state["x"])

            self.phase_buffer.push(phase[k])

            if k > self.update_point:
                tmpphase = self.phase_buffer.view(self.memory)
                for ch in range(self.n_channels):
                    tmp = np.unwrap(tmpphase[:, ch])
                    self.nu = (self.memory * np.sum(self._tbuf * tmp) -
                               self._Sx * np.sum(tmp)) / self._denom
                self.update_point += self.update_step

        return phase, amp


class ResOscillator:
    """Real-time measurement of phase and amplitude using a resonant oscillator.

    In this version, the corresponding “device” consists of a linear oscillator
    in resonance with the measured signal and of an integrating unit. It also
    provides both phase and amplitude of the input signal. The method exploits
    the known relation between the resonant oscillator's phase and amplitude
    and those of the input. Additionally, the resonant oscillator acts as a
    bandpass filter for experimental data.

    This filter also includes a frequency-adaptation algorithm, and removes
    low-frequency trend (baseline fluctuations). For a detailed description,
    refer to [1]_.

    Parameters
    ----------
    npt : int
        Number of points to be measured and processed
    fs : float
        Sampling frequency in Hz.
    nu : float
        Rough estimate of the tremor frequency.
    freq_adaptation : bool
        Whether to use the frequency adaptation algorithm (default=True).

    Returns
    -------
    Dphase : numpy.ndarray
        Array containing the phase values.
    Dampl : numpy.ndarray
        Array containing the amplitude values.

    References
    ----------
    .. [1] Rosenblum, M., Pikovsky, A., Kühn, A.A. et al. Real-time estimation
       of phase and amplitude with application to neural data. Sci Rep 11, 18037
       (2021). https://doi.org/10.1038/s41598-021-97560-5
    """

    def __init__(self, fs=1000, nu=4.5, update_factor=5, freq_adaptation=True,
                 assume_centered=False):

        # Parameters of the measurement "device"
        self.dt = 1 / fs  # Sampling interval
        self.om0 = nu  # Angular frequency
        self.alpha = 0.3 * self.om0
        self.gamma = self.alpha / 2

        # Parameters of adaptation algorithm
        nperiods = 1  # Number of previous periods for frequency correction
        npt_period = round(2 * np.pi / self.om0 / self.dt)  # Number of points per period
        self.memory = nperiods * npt_period  # M points for frequency correction buffer
        self.update_factor = update_factor  # Number of frequency updates per period
        self.update_step = round(npt_period / self.update_factor)
        self.updatepoint = 2 * self.memory

        # Precomputed quantities for linear fit for frequency adaptation
        self._tbuf = np.arange(1, self.memory + 1) * self.dt
        self._Sx = np.sum(self._tbuf)
        self._denom = self.memory * np.sum(self._tbuf ** 2) - self._Sx ** 2

        # Buffer to store past input values
        self.n_channels = None
        self.buffer = None
        self.runav = 0.  # Initial guess for the dc-component
        self.freq_adaptation = freq_adaptation
        self.assume_centered = assume_centered

    def _set_devices(self, n_channels):
        # Set up the phase and amplitude "devices"
        self.osc = [Device(self.om0, self.dt, self.gamma, "oscillator")
                    for _ in range(n_channels)]
        self.int = [Device(self.om0, self.dt, self.gamma, "integrator")
                    for _ in range(n_channels)]

    def transform(self, X):
        """Transform the input signal into phase and amplitude estimates.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels)
            The input signal to be transformed.

        Returns
        -------
        phase : float
            Current phase estimate.
        ampl : float
            Current amplitude estimate.
        """
        if self.buffer is None:
            self.n_channels = 1 if X.ndim < 2 else X.shape[1]
            self.buffer = Buffer(self.memory, self.n_channels)
            self.phase_buffer = Buffer(self.memory, self.n_channels)
            self._demean = np.zeros((3, self.n_channels)) # past demeaned signal values
            self._oscbuf = np.zeros((3, self.n_channels)) # past integrator values
            self._set_devices(self.n_channels)

        n_samples = X.shape[0]
        phase = np.zeros((n_samples, self.n_channels))
        ampl = np.zeros((n_samples, self.n_channels))

        for k in range(n_samples):
            self.buffer.push(X[k])
            self._demean = np.roll(self._demean, -1, axis=0)
            self._demean[-1] = self.buffer.view(1) - self.runav  # Baseline correction

            if self.buffer.counter < 3:
                continue # Skip the first two samples

            for ch in range(self.n_channels):
                # Perform one step of the oscillator equations
                self.osc[ch].step(self._demean[-3, ch],
                                  self._demean[-2, ch],
                                  self._demean[-1, ch])

                self._oscbuf[:, ch] = np.roll(self._oscbuf[:, ch], -1, axis=0)
                self._oscbuf[-1, ch] = self.osc[ch].state["y"]

                self.int[ch].step(self._oscbuf[-3, ch],
                                  self._oscbuf[-2, ch],
                                  self._oscbuf[-1, ch])

                # New phase and amplitude values
                v = self.int[ch].mu * self.int[ch].state["x"] * self.osc[ch].om0

                phase[k, ch] = np.arctan2(v, np.real(self.osc[ch].state["y"]))
                ampl[k, ch] = self.alpha * np.sqrt(self.osc[ch].state["y"] ** 2 + v ** 2)

            self.phase_buffer.push(phase[k])

            if self.freq_adaptation and k > self.updatepoint:
                tmpphase = self.phase_buffer.view(self.memory)
                # Recompute integration parameters for the new frequency
                for ch in range(self.n_channels):
                    tmp = np.unwrap(tmpphase[:, ch])
                    # Frequency via linear fit of the phases in the buffer
                    om0 = (self.memory * np.sum(self._tbuf * tmp) -
                        self._Sx * np.sum(tmp)) / self._denom
                    self.osc[ch].init_coefs(om0, self.dt, self.gamma)

                # Update running average
                if self.assume_centered is False:
                    self.runav = np.mean(self.buffer.view(self.memory), axis=0,
                                         keepdims=True)
                self.updatepoint += self.update_step  # Point for the next update

        return phase, ampl


def locking_based_phase(s, dt, npt):
    """Compute the locking-based phase.

    This technique exploits the ideas from the synchronization theory. It is
    well-known that a force s(t) acting on a limit-cycle oscillator can entrain
    it. It means that the oscillator's frequency becomes equal to that of the
    force, and their phases differ by a constant. Thus, the phase of the locked
    limit-cycle oscillator will correspond to the phase of the signal. For our
    purposes, it is helpful to use the simplest oscillator model, the so-called
    phase oscillator. To ensure the phase-locking to the force, we have to
    adjust the oscillator's frequency to the signal's frequency. We assume that
    we do not know the latter a priori, but can only roughly estimate it. We
    propose a simple approach that starts with this estimate and automatically
    tunes the “device's” frequency to ensure the locking and thus provides the
    instantaneous phase.

    Note that the amplitude is not determined with this approach. Technically,
    the algorithm reduces to solving differential equation incorporating
    measured data given at discrete time points; for details, see [1]_.

    Parameters
    ----------
    s: array-like
        Input signal.
    dt: float
        Time step.
    npt: int
        Number of points.

    Returns
    -------
    lb_phase: ndarray
        Locking-based phase.

    References
    ----------
    .. [1] Rosenblum, M., Pikovsky, A., Kühn, A.A. et al. Real-time estimation
       of phase and amplitude with application to neural data. Sci Rep 11, 18037
       (2021). https://doi.org/10.1038/s41598-021-97560-5
    """
    lb_phase = np.zeros(npt)

    # Parameters of the measurement device
    epsilon = 0.8  # Amplitude of the oscillator
    K = 0.5  # Frequency adaptation coefficient
    n_rk_steps = 5  # Number of Runge-Kutta steps per time step

    # Parameters of the frequency adaptation algorithm
    nu_est = 1.1  # initial guess for the frequency
    update_factor = 20  # number of frequency updates per period

    # Buffer for frequency estimation
    npbuf = round(2 * np.pi / nu_est / dt) # number of samples for frequency estimation
    update_point = 2 * npbuf
    update_step = round(npbuf / update_factor)
    tbuf = np.arange(npbuf) * dt
    sx = np.sum(tbuf)
    denom = npbuf * np.sum(tbuf ** 2) - sx ** 2

    for k in range(2, npt):
        lb_phase[k] = rk(lb_phase[k - 1], dt, n_rk_steps, nu_est, epsilon,
                         s[k], s[k - 1], s[k])

        if k > update_point:
            buffer = lb_phase[k - npbuf:k]
            nu_quasi = (npbuf * np.sum(tbuf * buffer) - sx * np.sum(buffer)) / denom
            nu_est = nu_est + K * (nu_quasi - nu_est)
            update_point += update_step

    return lb_phase


def rk(phi, dt, n_steps, omega, epsilon, s_prev, s, s_new):
    """Runge-Kutta method for phase calculation.

    Parameters
    ----------
    phi : float
        Initial phase value.
    dt : float
        Time step size.
    n_steps : int
        Number of steps to iterate.
    omega : float
        Angular frequency.
    epsilon : float
        Amplitude of the oscillation.
    s_prev : float
        Previous phase value.
    s : float
        Current phase value.
    s_new : float
        New phase value.

    Returns
    -------
    phi : float
        The calculated phase value.

    """
    h = dt / n_steps
    b = (s_new - s_prev) / dt / 2
    c = (s_prev - 2 * s + s_new) / dt ** 2
    hh = h / 2.
    h6 = h / 6.
    t = 0

    for _ in range(n_steps):
        th = t + hh
        dy0 = omega - epsilon * (s + b * t + c * t ** 2) * np.sin(phi)
        phit = phi + hh * dy0
        dyt = omega - epsilon * (s + b * th + c * th ** 2) * np.sin(phit)
        phit = phi + hh * dyt
        dym = omega - epsilon * (s + b * th + c * th ** 2) * np.sin(phit)
        phit = phi + h * dym
        dym = dym + dyt
        t = t + h
        dyt = omega - epsilon * (s + b * t + c * t ** 2) * np.sin(phit)
        phi = phi + h6 * (dy0 + 2 * dym + dyt)

    return phi


def init_coefs(om0, dt, alpha):
    """Compute coefficients for solving oscillator's equations.

    Parameters
    ----------
    om0 : float
        Oscillator frequency (estimation).
    dt : float
        Sampling interval.
    alpha : float
        Half of the damping.

    Returns
    -------
    C1 : float
        Coefficient C1.
    C2 : float
        Coefficient C2.
    C3 : float
        Coefficient C3.
    eetadel : complex
        Exponential term for amplitude device.
    ealdel : float
        Exponential term for amplitude device.
    eta : float
        Square root of the difference of oscillator frequency squared and
        damping squared.

    """
    # alpha is the half of the damping: x'' + 2 * alpha * x' + om0^2 * x = input
    eta2 = om0**2 - alpha**2
    eta2 = complex(eta2) if eta2 < 0 else eta2
    eta = np.sqrt(eta2)  # replicate Matlab behavior
    eetadel = np.exp(1j * eta * dt)
    a = 1. / eetadel
    ealdel = np.exp(alpha * dt)

    I1 = 1j * (a - 1) / eta
    I2 = (a * (1 + 1j * eta * dt) - 1) / eta2
    I3 = (a * (dt * eta * (2 + 1j * dt * eta) - 2 * 1j) + 2 * 1j) / eta2 / eta

    C1 = (I3 - I2 * dt) / (2 * dt**2 * ealdel)
    C2 = I1 - I3 / dt**2
    C3 = ealdel * (I2 * dt + I3) / (2 * dt**2)

    return C1, C2, C3, eetadel, ealdel, eta


def one_step_oscillator(x, xd, alpha, eta, eeta_del, eal_del, C1, C2, C3, spp, sp, s):
        """Perform one step of the oscillator equations.

        Parameters
        ----------
        x : float
            State variable x.
        xd : float
            Derivative of state variable x.
        alpha : float
            Damping parameter.
        eta : float
            Square root of the difference of oscillator frequency squared and
            damping squared.
        eeta_del : complex
            Exponential term for time step.
        eal_del : float
            Exponential term for time step.
        C1 : float
            Coefficient C1.
        C2 : float
            Coefficient C2.
        C3 : float
            Coefficient C3.
        spp : float
            Second orevious measurement.
        sp : float
            Previous measurement.
        s : float
            Current measurement.

        Returns
        -------
        x : float
            Updated state variable x.
        xd : float
            Updated derivative of state variable x.
        """
        A = x - 1j * (xd + alpha * x + C1 * spp + C2 * sp + C3 * s) / eta
        d = A * eeta_del
        y = np.real(d)
        yd = - eta * np.imag(d)
        x = y / eal_del
        xd = (yd - alpha * y) / eal_del

        return x, xd


def one_step_integrator(z, edelmu, mu, dt, spp, sp, s):
    """Perform one step of the integrator in the oscillator equations.

    Parameters
    ----------
    z : float
        State variable z.
    edelmu : float
        Exponential term for integrator.
    mu : float
        Integrator parameter.
    dt : float
        Sampling interval.
    spp : float
        Second previous measurement.
    sp : float
        Previous measurement.
    s : float
        Current measurement.

    Returns
    -------
    z : float
        Updated state variable z.
    """
    a = sp
    b = (s - spp) / 2 / dt
    c = (spp - 2 * sp + s) / 2 / dt ** 2
    d = -a + b * mu - 2 * c * (mu ** 2)
    C0 = z + d
    z = C0 * edelmu - d + b * dt - 2 * c * mu * dt + c * dt ** 2
    return z


class ECHT:
    """Endpoint Corrected Hilbert Transform (ECHT).

    See [1]_ for details.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_channels)
        Time domain signal.
    l_freq : float | None
        Low-cutoff frequency of a bandpass causal filter. If None, the data is
        only low-passed.
    h_freq : float | None
        High-cutoff frequency of a bandpass causal filter. If None, the data is
        only high-passed.
    sfreq : float
        Sampling rate of time domain signal.
    n : int, optional
        Length of analytic signal. If None, it defaults to the length of X.
    filt_order : int, optional
        Order of the filter. Default is 2.

    Notes
    -----
    One common implementation of the Hilbert Transform uses a DFT (aka FFT)
    as part of its computation. Inherent to the DFT is the assumption that
    a finite sample of a signal is replicated infinitely in time, effectively
    abutting the end of a sample with its replicated start. If the start and
    end of the sample are not continuous with each other, distortions are
    introduced by the DFT. Echt effectively smooths out this 'discontinuity'
    by selectively deforming the start of the sample. It is hence most suited
    for real-time applications in which the point/s of interest is/are the
    most recent one/s (i.e. last) in the sample window.

    We found that a filter bandwidth (BW=h_freq-l_freq) of up to half the
    signal's central frequency works well.

    References
    ----------
    .. [1] Schreglmann, S. R., Wang, D., Peach, R. L., Li, J., Zhang, X.,
        Latorre, A., ... & Grossman, N. (2021). Non-invasive suppression of
        essential tremor via phase-locked disruption of its temporal coherence.
        Nature communications, 12(1), 363.

    Examples
    --------
    >>> f0 = 2
    >>> filt_BW = f0 / 2
    >>> N = 1000
    >>> sfreq = N / (2 * np.pi)
    >>> t = np.arange(-2 * np.pi, 0, 1 / sfreq)
    >>> X = np.cos(2 * np.pi * f0 * t - np.pi / 4)
    >>> l_freq = f0 - filt_BW / 2
    >>> h_freq = f0 + filt_BW / 2
    >>> Xf = echt(X, l_freq, h_freq, sfreq)
    """

    def __init__(self, l_freq, h_freq, sfreq, n_fft=None, filt_order=2):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.n_fft = n_fft
        self.filt_order = filt_order

        # attributes
        self.h_ = None
        self.coef_ = None

    def fit(self, X, y=None):
        """Fit the ECHT transform to the input signal."""
        if self.n_fft is None:
            self.n_fft = next_fast_len(X.shape[0])

        # Set the amplitude of the negative components of the FFT to zero and
        # multiply the amplitudes of the positive components, apart from the
        # zero-frequency component (DC) and Nyquist components, by 2.
        #
        # If the signal has an even number of elements n, the frequency components
        # are:
        # - n/2-1 negative elements,
        # - one DC element,
        # - n/2-1 positive elements and
        # - one Nyquist element (in order).
        #
        # If the signal has an odd number of elements n, the frequency components
        # are:
        # - (n-1)/2 negative elements,
        # - one DC element,
        # - (n-1)/2 positive elements (in order).
        # - no positive element corresponding to the Nyquist frequency.
        self.h_ = np.zeros(self.n_fft)
        self.h_[0] = 1
        self.h_[1:(self.n_fft // 2) + 1] = 2
        if self.n_fft % 2 == 0:
            self.h_[self.n_fft // 2] = 1

        # The frequency response vector is computed using freqz from the filter's
        # impulse response function, computed by butter function, and the user
        # defined low-cutoff frequency, high-cutoff frequency and sampling rate.
        Wn = [self.l_freq / (self.sfreq / 2), self.h_freq / (self.sfreq / 2)]
        b, a = butter(self.filt_order, Wn, btype="band")
        T = 1 / self.sfreq * self.n_fft
        filt_freq = np.ceil(np.arange(-self.n_fft / 2, self.n_fft / 2) / T)

        self.coef_ = freqz(b, a, filt_freq, fs=self.sfreq)[1]
        self.coef_ = self.coef_[:, None]

        return self

    def transform(self, X):
        """Apply the ECHT transform to the input signal.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_channels)
            The input signal to be transformed.

        Returns
        -------
        Xf : ndarray, shape=(n_samples, n_channels)
            The transformed signal (complex-valued).

        """
        if not np.isrealobj(X):
            X = np.real(X)

        # if not fitted
        if self.h_ is None or self.coef_ is None:
            self.fit(X)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        # Compute the FFT of the signal.
        Xf = fft(X, self.n_fft, axis=0)

        # In contrast to :meth:`scipy.signal.hilbert()`, the code then
        # multiplies the array by a frequency response vector of a causal
        # bandpass filter.
        Xf = Xf * self.h_[:, None]

        # The array is arranged, using fft_shift function, so that the zero-frequency
        # component is at the center of the array, before the multiplication, and
        # rearranged back so that the zero-frequency component is at the left of the
        # array using ifft_shift(). Finally, the IFFT is computed.
        Xf = fftshift(Xf)
        Xf = Xf * self.coef_
        Xf = ifftshift(Xf)
        Xf = ifft(Xf, axis=0)

        return Xf

    def fit_transform(self, X, y=None):
        """Fit the ECHT transform to the input signal and transform it."""
        return self.fit(X).transform(X)