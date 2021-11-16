import argparse
import functools
from typing import TextIO, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
try:
    import tikzplotlib
except ImportError:
    pass


def load_data(file: TextIO, uncert: bool = False, sep: str = None) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # Load data
    # Note this version does not subtract the initial time
    x_data = []
    y_data = []
    x_uncert = []
    y_uncert = []
    for line in file:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pcs = line.split(sep)
        time = float(pcs[0])
        angle = float(pcs[1])
        x_data.append(time)
        y_data.append(angle)
        if uncert:
            x_uncert.append(float(pcs[2]) if len(pcs) > 2 else 0)
            y_uncert.append(float(pcs[3]) if len(pcs) > 3 else 0)
    if uncert:
        return np.array(x_data), np.array(y_data), np.array(x_uncert), np.array(y_uncert)
    else:
        return np.array(x_data), np.array(y_data)


def fit0(theta: float, t0: float) -> float: # pylint: disable=unused-argument
    return t0


def fit1(theta: float, t0: float, b: float) -> float:
    return t0 + theta * b


def fit2(theta: float, t0: float, b: float, c: float) -> float:
    return t0 + theta * b + theta ** 2 * c


def fit3(theta: float, t0: float, b: float, c: float, d: float) -> float:
    return t0 + theta * b + theta ** 2 * c + theta ** 3 * d


def fit4(theta: float, t0: float, b: float, c: float, d: float, e: float) -> float:
    return t0 + theta * b + theta ** 2 * c + theta ** 3 * d + theta ** 4 * e


def fit5(theta: float, t0: float, b: float, c: float, d: float, e: float, f: float) -> float:
    return t0 + theta * b + theta ** 2 * c + theta ** 3 * d + theta ** 4 * e + theta ** 5 * f


def fit6(theta: float, t0: float, b: float, c: float, d: float, e: float, f: float, g: float) -> float:
    return t0 + theta * b + theta ** 2 * c + theta ** 3 * d + theta ** 4 * e + theta ** 5 * f + theta ** 6 * g


FIT_FUNCS = [fit0, fit1, fit2, fit3, fit4, fit5, fit6]
PARAM_NAMES = ["t0", "b", "c", "d", "e", "f", "g"]


def main(data_in: TextIO, degree: int, guess_period: float, save_graph: str, sep: str, save_residuals: str, limit_angles: float):
    x_data, y_data, x_uncert, y_uncert = load_data(data_in, uncert=True, sep=sep)
    if limit_angles:
        data_range = np.where(np.abs(x_data) <= limit_angles)
        x_data = x_data[data_range]
        y_data = y_data[data_range]
        x_uncert = x_uncert[data_range]
        y_uncert = y_uncert[data_range]
    # Create initial guesses
    # t0 varies but the others should all be roughly 0
    guess = [guess_period] + [0] * degree
    # Do the fit
    popt, pcov = optimize.curve_fit(FIT_FUNCS[degree], x_data, y_data, p0=guess) # pylint: disable=unbalanced-tuple-unpacking
    # Collect params and uncertainties
    params = {name: val for name, val in zip(PARAM_NAMES, popt)}
    stdevs = {name: np.sqrt(pcov[i, i]) for name, i in zip(PARAM_NAMES, range(degree + 1))}
    print("Fit Parameters:")
    for name, val in params.items():
        print(f"{name.upper()}\t{val}")
    print("Standard Deviations (Uncertainties):")
    for name, val in stdevs.items():
        print(f"{name.upper()}\t{val}")
    # Create a new function with the optimal fit parameters
    bestfit = functools.partial(FIT_FUNCS[degree], **params)

    # Plot everything
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # hspace is horizontal space between the graphs
    fig.subplots_adjust(hspace=0.6)
    ax1.errorbar(x_data, y_data, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Collected Data")

    start, stop = min(x_data) * 1.1, max(x_data) * 1.1
    print(f"Domain: [{start}, {stop}]")
    bestfit_x = np.arange(start, stop, (stop - start) / 1000)
    # When degree is 0 since the function is reduced to a constant numpy is confused so a special case is required
    if degree == 0:
        bestfit_y = np.full((len(bestfit_x),), params["t0"])
    else:
        bestfit_y = bestfit(bestfit_x)
    ax1.plot(bestfit_x, bestfit_y, "r", label="Best Fit Curve $T_0(\\theta)$")

    ax1.set_xlabel("Initial Amplitude $\\theta$ (rad)")
    ax1.set_ylabel("Period $t_0$ (s)")
    ax1.set_title("Data & Best Fit Curve")
    ax1.legend(loc="upper right")

    residuals = y_data - bestfit(x_data)
    ax2.errorbar(x_data, residuals, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Residuals")
    ax2.plot([start, stop], [0, 0], "r", label="Zero Line")
    if save_residuals is not None:
        with open(save_residuals, "w", encoding="utf-8") as f:
            for vals in zip(x_data, residuals, x_uncert, y_uncert):
                f.write(" ".join(str(s) for s in vals) + "\n")

    ax2.set_xlabel("Initial Amplitude $\\theta$ (rad)")
    ax2.set_ylabel("Error $t_0 - T_0(\\theta)$ (s)")
    ax2.set_title("Fit Residuals")
    ax2.legend(loc="upper right")

    if save_graph is not None:
        tikzplotlib.save(save_graph)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit period-amplitude data to a power series for Lab 2.")
    parser.add_argument("data_in", type=argparse.FileType("r", encoding="utf-8"), help="The input file. Each line in the file should contain 2 data values, and optionally 2 uncertainty values, separated by spaces or SEP.")
    parser.add_argument("--degree", "-d", type=int, default=0, help="The max degree of the power series. Supported values are 0 (default) to 6. The degree is the number of terms minus 1, since the first term has degree 0.")
    parser.add_argument("--guess-period", "-p", type=float, default=1, help="An initial guess for the period (T0) of the pendulum. The default is 1s, but if your pendulum has a much longer or shorter period you may want to adjust this.")
    parser.add_argument("--save-graph", type=str, default=None, help="Save the graph to a TeX file. Requires tikzplotlib.")
    parser.add_argument("--save-residuals", type=str, default=None, help="Save the fit residuals to a txt file.")
    parser.add_argument("--sep", "-s", type=str, default=None, help="Separator between 2 data values in the input file. Default is any whitespace, but can be set to any string, e.g. set this to a comma if your data is a CSV.")
    parser.add_argument("--limit-angles", type=float, default=None, help="Cap the maximum angle.")
    main(**vars(parser.parse_args()))
