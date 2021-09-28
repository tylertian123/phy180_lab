# -*- coding: utf-8 -*-
"""
python3 fit.py [filename] [angle_format=deg|rad] [time_format=sec|frames] [fps]

Adapted from fitting.py posted on Quercus.

The data file is named "data.txt" by default, but can be specified as an argument.
If angle_format is "deg", the angles will be converted into radians first.
If time_format is "frames", the time values will be divided by the fps (default 30).

The file format is as follows:
<time1> <angle1>
<time2> <angle2>
...

This plots the data and best fit on one graph, and the residuals and zero line
(for reference) on the other graph.

Ideally your residuals graph should look like noise, otherwise there is more
information you could extract from your data (or you used the wrong fitting
function).

The program will print the values of A, tau, T and phi along with their standard deviations.
The standard deviations can be used as uncertainties.

REMEMBER TO SET YOUR INITIAL GUESSES ON LINE 39!!
If you don't set these correctly according to your data, then the program might not be able
to compute the best-fit curve. To get an estimate, try running with DO_FIT = False (line 39),
which turns off the fitting and just plots the points. With your data points you should be
able to estimate your amplitude, period, and any phase shift. I have not figured out a way
to estimate tau, so leaving it at 20-100 should be good.

Note: Make sure you give it a lot of data. Observations show that if you only give it a few
periods, the decay may be negligible and as a result the estimated tau may be very large and
completely unrealistic (e.g. > 10^6).
"""

import functools
import sys
import math
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

DO_FIT = True
# Your initial guess of (a, tau, T, phi)
INIT_GUESS = (1.1, 20, 1.1, 0)

# Function used to fit
# First variable is the x-data, and the rest are the parameters we want to determine
def fit_func(t: float, a: float, tau: float, period: float, phi: float) -> float:
    return a * np.exp(-t / tau) * np.cos(2 * np.pi * t / period + phi)

def fit(x_data, y_data) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    # popt: least-squares optimized values for the parameters
    # pcov: covariance matrix
    popt, pcov = optimize.curve_fit(fit_func, x_data, y_data, p0=INIT_GUESS)

    a = popt[0]
    tau = popt[1]
    period = popt[2]
    phi = popt[3]

    # The diagonal of the covariance matrix is the variance of each variable
    # Take the square root to get the standard deviation
    stdev_a, stdev_tau, stdev_period, stdev_phi = (np.sqrt(pcov[i, i]) for i in range(4))

    return (a, tau, period, phi), (stdev_a, stdev_tau, stdev_period, stdev_phi)

if __name__ == "__main__":
    # Parse args
    filename = sys.argv[1] if len(sys.argv) > 1 else "data.txt"
    angle_format = sys.argv[2] if len(sys.argv) > 2 else "rad"
    if angle_format not in ("rad", "deg"):
        print("Valid angle formats are 'deg' or 'rad'", sys.stderr)
        sys.exit(1)
    time_format = sys.argv[3] if len(sys.argv) > 3 else "sec"
    if time_format not in ("sec", "frames"):
        print("Valid time formats are 'sec' or 'frames'", sys.stderr)
        sys.exit(1)
    try:
        fps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    except ValueError:
        print("Invalid format for fps", sys.stderr)
        sys.exit(1)

    # Load data
    x_data = []
    y_data = []
    init_time = None
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pcs = line.split()
        if len(pcs) < 2:
            print("Error: Invalid data format", file=sys.stderr)
            sys.exit(1)
        time = float(pcs[0])
        if time_format == "frames":
            time /= fps
        angle = float(pcs[1])
        if angle_format == "deg":
            angle = math.radians(angle)
        if init_time is None:
            init_time = time
        time -= init_time
        x_data.append(time)
        y_data.append(angle)
    x_data = np.array(x_data, dtype=np.float128)
    y_data = np.array(y_data, dtype=np.float128)

    if DO_FIT:
        (a, tau, period, phi), (stdev_a, stdev_tau, stdev_period, stdev_phi) = fit(x_data, y_data)
        bestfit = functools.partial(fit_func, a=a, tau=tau, period=period, phi=phi)

    # Plot best fit curve
    start, stop = min(x_data), max(x_data)
    if DO_FIT:
        # Plot 1000 points for the best fit curve; this can be changed
        x_vals = np.arange(start, stop, (stop - start) / 1000)
        y_vals = bestfit(x_vals)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # hspace is horizontal space between the graphs
    fig.subplots_adjust(hspace=0.6)

    if DO_FIT:
        # Plot the best fit curve on top of the data points as a line
        ax1.plot(x_vals, y_vals, "r", label="Best Fit Curve")
    # Plot the data
    ax1.scatter(x_data, y_data, label="Collected Data", s=4)
    #ax1.errorbar(xdata, ydata, yerr=yerror, xerr=xerror, fmt=".")

    # Label the axes; change here
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    ax1.set_title("Data & Best Fit Curve")
    ax1.legend(loc="upper right")

    if DO_FIT:
        # Print the values
        print("Qty\tValue\t\t\tStdev")
        print(f"A\t{a}\t{stdev_a}")
        print(f"tau\t{tau}\t{stdev_tau}")
        print(f"T\t{period}\t{stdev_period}")
        print(f"phi\t{phi}\t{stdev_phi}")

        # Plot residuals
        residuals = y_data - bestfit(x_data)
        ax2.scatter(x_data, residuals, label="Residuals", s=4)
        #ax2.errorbar(xdata, residual, yerr=yerror, xerr=xerror, fmt=".")

        # Plot the zero line for reference
        ax2.plot([start, stop], [0, 0], "r")

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Fit Residual")
        ax2.set_title("Fit Residuals")

    plt.show()
