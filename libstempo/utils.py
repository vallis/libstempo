"""Utility functions for noise models."""

import numpy as np


def quantize_fast(times, flags, dt=1.0):
    isort = np.argsort(times)

    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt and flags[i] != "":
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])

    # only keep epochs with 2 or more TOAs
    bucket_ind = [ind for ind in bucket_ind if len(ind) >= 2]

    avetoas = np.array([np.mean(times[l]) for l in bucket_ind], "d")
    U = np.zeros((len(times), len(bucket_ind)), "d")
    for i, l in enumerate(bucket_ind):
        U[l, i] = 1

    return avetoas, U


def create_fourier_design_matrix(t, nmodes, freq=False, Tspan=None, logf=False, fmin=None, fmax=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency

    :return: F: fourier design matrix
    :return: f: Sampling frequencies (if freq=True)
    """

    N = len(t)
    F = np.zeros((N, 2 * nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1 / T, nmodes / T, nmodes)
    if logf:
        f = np.logspace(np.log10(1 / T), np.log10(nmodes / T), nmodes)

    Ffreqs = np.zeros(2 * nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    F[:, ::2] = np.sin(2 * np.pi * t[:, None] * f[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * t[:, None] * f[None, :])

    if freq:
        return F, Ffreqs
    else:
        return F


def powerlaw(f, log10_A=-16, gamma=5):
    """Power-law PSD.

    :param f: Sampling frequencies
    :param log10_A: log10 of red noise Amplitude [GW units]
    :param gamma: Spectral index of red noise process
    """

    fyr = 1 / 3.16e7
    return (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * fyr ** (gamma - 3) * f ** (-gamma)
