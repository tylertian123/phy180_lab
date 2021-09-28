# -*- coding: utf-8 -*-
"""
Adapted from fitting.py posted on Quercus.

The file format is as follows:
<time1> <angle1>
<time2> <angle2>
...

The data file is named "data.txt" by default, but can be specified as an argument:
python3 fit.py <filename>

This plots the data and best fit on one graph, and the residuals and zero line
(for reference) on the other graph.

Ideally your residuals graph should look like noise, otherwise there is more
information you could extract from your data (or you used the wrong fitting
function).

The program will print the values of A, tau, T and phi along with their standard deviations.
The standard deviations can be used as uncertainties.
"""

import functools
import sys
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Function used to fit
# First variable is the x-data, and the rest are the parameters we want to determine
def fit_func(t: float, a: float, tau: float, period: float, phi: float) -> float:
    return a * np.exp(-t / tau) * np.cos(2 * np.pi * t / period + phi)

def fit(x_data, y_data) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    # Your initial guess of (a, tau, T, phi)
    # TODO: Prompt this instead
    init_guess = (0.6, 25, 0.75, 0)

    # popt: least-squares optimized values for the parameters
    # pcov: covariance matrix
    popt, pcov = optimize.curve_fit(fit_func, x_data, y_data, p0=init_guess)

    a = popt[0]
    tau = popt[1]
    period = popt[2]
    phi = popt[3]

    # The diagonal of the covariance matrix is the variance of each variable
    # Take the square root to get the standard deviation
    stdev_a, stdev_tau, stdev_period, stdev_phi = (np.sqrt(pcov[i, i]) for i in range(4))

    return (a, tau, period, phi), (stdev_a, stdev_tau, stdev_period, stdev_phi)

if __name__ == "__main__":
    # Load values
    filename = sys.argv[1] if len(sys.argv) > 2 else "data.txt"

    x_data = []
    y_data = []
    for line in open(filename, "r", encoding="utf-8"):
        pcs = line.strip().split()
        if len(pcs) < 2:
            print("Error: Invalid data format", file=sys.stderr)
            sys.exit(1)
        x_data.append(pcs[0])
        y_data.append(pcs[1])
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    (a, tau, period, phi), (stdev_a, stdev_tau, stdev_period, stdev_phi) = fit(x_data, y_data)
    bestfit = functools.partial(fit_func, a=a, tau=tau, period=period, phi=phi)

    # Plot best fit curve
    start, stop = min(x_data), max(x_data)
    # Plot 1000 points for the best fit curve; this can be changed
    x_vals = np.arrange(start, stop, (stop - start) / 1000)
    y_vals = bestfit(x_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # hspace is horizontal space between the graphs
    fig.subplots_adjust(hspace=0.6)

    # Plot the data
    ax1.plot(x_data, y_data, fmt=".", label="Collected Data")
    #ax1.errorbar(xdata, ydata, yerr=yerror, xerr=xerror, fmt=".")
    # Plot the best fit curve on top of the data points as a line
    ax1.plot(x_vals, y_vals, label="Best Fit Curve")

    # Label the axes; change here
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    ax1.set_title("Data & Best Fit Curve")

    # Print the values
    print("Quantity\tValue\t\tStdev")
    print(f"A\t{a}\t{stdev_a}")
    print(f"tau\t{tau}\t{stdev_tau}")
    print(f"T\t{period}\t{stdev_period}")
    print(f"phi\t{phi}\t{stdev_phi}")

    # Plot residuals
    residuals = y_data - bestfit(x_data)
    ax2.plot(x_data, residuals, fmt=".", label="Residuals")
    #ax2.errorbar(xdata, residual, yerr=yerror, xerr=xerror, fmt=".")

    # Plot the zero line for reference
    ax2.plot([start, stop], [0, 0])

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Fit Residual")
    ax2.set_title("Fit Residuals")

    plt.show()
