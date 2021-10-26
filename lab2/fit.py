import argparse
import functools
import pprint
from typing import TextIO, Tuple, Union

import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from scipy import optimize


def load_data(file: TextIO, uncert: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
        pcs = line.split()
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


FIT_FUNCS = [fit0, fit1, fit2, fit3, fit4]
PARAM_NAMES = ["t0", "b", "c", "d", "e"]


def main(data_in: TextIO, degree: int, guess_period: float, save_graph: str):
    x_data, y_data, x_uncert, y_uncert = load_data(data_in, uncert=True)
    # Create initial guesses
    # t0 varies but the others should all be roughly 0
    guess = [guess_period] + [0] * degree
    # Do the fit
    popt, pcov = optimize.curve_fit(FIT_FUNCS[degree], x_data, y_data, p0=guess) # pylint: disable=unbalanced-tuple-unpacking
    # Collect params and uncertainties
    params = {name: val for name, val in zip(PARAM_NAMES, popt)}
    stdevs = {name: np.sqrt(pcov[i, i]) for name, i in zip(PARAM_NAMES, range(degree + 1))}
    print("Fit Parameters:")
    pprint.pprint(params, sort_dicts=False)
    print("Standard Deviations:")
    pprint.pprint(stdevs, sort_dicts=False)
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

    ax2.set_xlabel("Initial Amplitude $\\theta$ (rad)")
    ax2.set_ylabel("Error $t_0 - T_0(\\theta)$ (s)")
    ax2.set_title("Fit Residuals")
    ax2.legend(loc="upper right")

    if save_graph is not None:
        tikzplotlib.save(save_graph)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit data for lab 2.0")
    parser.add_argument("data_in", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--guess-period", type=float, default=1)
    parser.add_argument("--save-graph", type=str, default=None)
    main(**vars(parser.parse_args()))
