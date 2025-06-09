# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def fit_cos(tt, yy):
    print("fit_cos function defined")  # Debug print
    tt = np.asarray(tt)
    yy = np.asarray(yy)
    
    # FFT for initial guesses
    ff = np.fft.fftfreq(len(tt), tt[1] - tt[0])
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
    guess = [np.std(yy) * 2**0.5, 2 * np.pi * guess_freq, 0, np.mean(yy)]

    # Cosine function
    def cosfunc(t, A, w, p, c):
        return A * np.cos(w * t + p) + c

    # Fit with bounds to improve stability
    bounds = ([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
    popt, _ = optimize.curve_fit(cosfunc, tt, yy, p0=guess, bounds=bounds, maxfev=10000)
    A, w, p, c = popt
    
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "fitfunc": lambda t: A * np.cos(w * t + p) + c
    }

def plot_fit(tt, data_, res): 
    plt.plot(tt, data_, "ok", label="Noisy")
    tt_dense = np.linspace(min(tt), max(tt), len(tt) * 10)
    plt.plot(tt_dense, res["fitfunc"](tt_dense), "r-", label="Fit")
    plt.legend()
    plt.show()


def main(data_):

    N, amp, omega, phase, offset, noise = 500, 1.0, 2.0, 0.5, 4.0, 0.5  # Reduced noise for stability
    tt = np.linspace(0, 10, N)
    yy = amp * np.cos(omega * tt + phase) + offset
    yynoise = yy + noise * (np.random.random(N) - 0.5)
    
    # Fit cosine function
    res = fit_cos(tt, yynoise)
    
    # Calculate maximum velocity
    max_velocity = abs(res["amp"] * res["omega"])
    
    # Print results
    print(f"Amplitude: {res['amp']:.3f}, Omega: {res['omega']:.3f}, "
          f"Phase: {res['phase']:.3f}, Offset: {res['offset']:.3f}")
    print(f"Maximum Velocity: {max_velocity:.3f} units/s")
    
    # Plot results
    plot_fit(tt, yynoise, res)





def main_test():
    # Generate sample data
    N, amp, omega, phase, offset, noise = 500, 1.0, 2.0, 0.5, 4.0, 0.5  # Reduced noise for stability
    tt = np.linspace(0, 10, N)
    yy = amp * np.cos(omega * tt + phase) + offset
    yynoise = yy + noise * (np.random.random(N) - 0.5)
    
    # Fit cosine function
    res = fit_cos(tt, yynoise)
    
    # Calculate maximum velocity
    max_velocity = abs(res["amp"] * res["omega"])
    
    # Print results
    print(f"Amplitude: {res['amp']:.3f}, Omega: {res['omega']:.3f}, "
          f"Phase: {res['phase']:.3f}, Offset: {res['offset']:.3f}")
    print(f"Maximum Velocity: {max_velocity:.3f} units/s")
    
    # Plot results
    plot_fit(tt, yynoise, res)



main_test()

