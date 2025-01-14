# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:47:54 2022

@author: marca
"""

import warnings

import numpy as np
from scipy import fft as sp_fft
from scipy.signal import _signaltools
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal.windows.windows import get_window

"""
Just a simplified version of the scipy spectral factory 
"""


def periodogram_(
    x, fs=1.0, window="boxcar", nperseg=None, nfft=None, scaling="density", axis=-1
):
    x = np.asarray(x)

    if x.size == 0:
        return np.empty(x.shape), np.empty(x.shape)

    if nperseg is None:
        nperseg = x.shape[axis]

    return welch_(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nfft,
        scaling=scaling,
        axis=axis,
    )


def welch_(
    x,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    scaling="density",
    axis=-1,
):
    freqs, Pxx = csd_(
        x,
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
        axis=axis,
    )

    return freqs, Pxx.real


def csd_(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    scaling="density",
    axis=-1,
):
    freqs, _, Pxy = spectral_helper(
        x, y, fs, window, nperseg, noverlap, nfft, scaling, axis
    )

    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            Pxy = Pxy.mean(axis=-1)

        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    return freqs, Pxy


def spectral_helper(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    scaling="density",
    axis=-1,
    boundary=None,
    padded=False,
):
    boundary_funcs = {
        "even": even_ext,
        "odd": odd_ext,
        "constant": const_ext,
        "zeros": zero_ext,
        None: None,
    }

    if boundary not in boundary_funcs:
        raise ValueError(
            "Unknown boundary option '{0}', must be one of: {1}".format(
                boundary, list(boundary_funcs.keys())
            )
        )

    same_data = y is x
    axis = int(axis)

    x = np.asarray(x)

    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)

    else:
        outdtype = np.result_type(x, np.complex64)

    if not same_data:
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)

        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape

        except ValueError as e:
            raise ValueError("x and y cannot be broadcast together.") from e

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)

    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.moveaxis(np.empty(outshape), -1, axis)

            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)

            if not same_data and y.ndim > 1:
                y = np.moveaxis(y, axis, -1)

    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)

            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:
        nperseg = int(nperseg)

        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    win, nperseg = triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg

    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")

    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2

    else:
        noverlap = int(noverlap)

    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    nstep = nperseg - noverlap

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)

        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)

    if padded:
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)

        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())

    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2

    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    freqs = sp_fft.rfftfreq(nfft, 1 / fs)

    result = fft_helper(x, win, detrend_func, nperseg, noverlap, nfft)

    if not same_data:
        result_y = fft_helper(y, win, detrend_func, nperseg, noverlap, nfft)
        result = np.conjugate(result) * result_y

    else:
        result = np.conjugate(result) * result

    result *= scale
    if nfft % 2:
        result[..., 1:] *= 2

    else:
        result[..., 1:-1] *= 2

    time = np.arange(
        nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap
    ) / float(fs)

    if boundary is not None:
        time -= (nperseg / 2) / fs

    result = result.astype(outdtype)

    if same_data:
        result = result.real

    if axis < 0:
        axis -= 1

    result = np.moveaxis(result, -1, axis)

    return freqs, time, result


def fft_helper(x, win, detrend_func, nperseg, noverlap, nfft):
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]

    else:
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    result = detrend_func(result)
    result = win * result

    result = result.real
    func = sp_fft.rfft
    result = func(result, n=nfft)

    return result


def triage_segments(window, nperseg, input_length):
    if isinstance(window, str) or isinstance(window, tuple):
        if nperseg is None:
            nperseg = 256

        if nperseg > input_length:
            warnings.warn(
                "nperseg = {0:d} is greater than input length "
                " = {1:d}, using nperseg = {1:d}".format(nperseg, input_length)
            )
            nperseg = input_length

        win = get_window(window, nperseg)

    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")

        if input_length < win.shape[-1]:
            raise ValueError("window is longer than input signal")

        if nperseg is None:
            nperseg = win.shape[0]

        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError(
                    "value specified for nperseg is different" " from length of window"
                )
    return win, nperseg


def detrend_func(d):
    return _signaltools.detrend(d, type="constant", axis=-1)
