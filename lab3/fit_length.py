import functools
import click
import numpy as np
from typing import List, TextIO, Tuple, Union
from matplotlib import pyplot as plt
from scipy import optimize, odr


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


def fitfunc(l: float, k: float, n: float, l0: float) -> float:
    return k * (l0 + l) ** n


def odr_fitfunc(p: List[float], l: float) -> float:
    return p[0] * (p[2] + l) ** p[1]


def do_fit(x_data: np.ndarray, y_data: np.ndarray, x_uncert: np.ndarray, y_uncert: np.ndarray, guesses, use_odr: bool) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if use_odr:
        model = odr.Model(odr_fitfunc)
        data = odr.RealData(x_data, y_data, sx=x_uncert, sy=y_uncert)
        output = odr.ODR(data, model, beta0=guesses).run()
        return (output.beta, output.sd_beta)
    else:
        popt, pcov = optimize.curve_fit(fitfunc, x_data, y_data, p0=guesses)
        return (popt, (np.sqrt(pcov[i, i]) for i in range(len(guesses))))


@click.command()
@click.argument("data_in", type=click.File("r", encoding="utf-8"))
@click.option("--guess-k", "-k", type=float, default=2, help="Initial guess for k")
@click.option("--guess-n", "-n", type=float, default=0.5, help="Initial guess for n")
@click.option("--guess-l", "-l", type=float, default=0, help="Initial guess for L0")
@click.option("--sep", "-s", type=str, default=None, help="Separator in the data file")
@click.option("--odr/--no-odr", "use_odr", default=False, help="Use ODR instead of least squares and take into account uncertainties")
@click.option("--save-residuals", type=click.File("w", encoding="utf-8"), default=None, help="Save residuals to a file")
def main(data_in: TextIO, guess_k: float, guess_n: float, guess_l: float, sep: str, use_odr: bool, save_residuals: TextIO):
    """
    Fit period to a function of length for lab 3a.
    """
    x_data, y_data, x_uncert, y_uncert = load_data(data_in, uncert=True, sep=sep)

    (k, n, l0), (sk, sn, sl0) = do_fit(x_data, y_data, x_uncert, y_uncert, (guess_k, guess_n, guess_l), use_odr)
    print("Qty\tValue\t\t\tStdev/Uncertainty")
    print(f"k\t{k}\t{sk}")
    print(f"n\t{n}\t{sn}")
    print(f"L0\t{l0}\t{sl0}")
    
    bestfit = functools.partial(fitfunc, k=k, n=n, l0=l0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    fig.subplots_adjust(hspace=0.6)

    ax1.errorbar(x_data, y_data, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Collected Data")
    ax2.errorbar(x_data, y_data, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Collected Data")
    start, stop = min(x_data), max(x_data)
    print(f"Domain: [{start}, {stop}]")
    start -= (stop - start) * 0.01
    stop += (stop - start) * 0.01
    bestfit_x = np.arange(start, stop, (stop - start) / 1000)
    bestfit_y = bestfit(bestfit_x)
    ax1.plot(bestfit_x, bestfit_y, "r", label="Best Fit Curve $T(L)$")
    ax2.plot(bestfit_x, bestfit_y, "r", label="Best Fit Curve $T(L)$")

    ax1.set_xlabel("String Length $L$ (m)")
    ax1.set_ylabel("Period $T$ (s)")
    ax1.set_title("Data & Best Fit Curve")
    ax1.legend(loc="best")
    ax2.set_xlabel("String Length $L$ (m)")
    ax2.set_ylabel("Period $T$ (s)")
    ax2.set_title("Data & Best Fit Curve (Logarithmic)")
    ax2.legend(loc="best")

    residuals = y_data - bestfit(x_data)
    ax3.errorbar(x_data, residuals, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Residuals")
    ax3.plot([start, stop], [0, 0], "r", label="Zero Line")
    if save_residuals is not None:
        for vals in zip(x_data, residuals, x_uncert, y_uncert):
            save_residuals.write(" ".join(str(s) for s in vals) + "\n")
    ax3.set_xlabel("String Length $L$ (m)")
    ax3.set_ylabel("Error $T - T(L)$ (s)")
    ax3.set_title("Fit Residuals")
    ax3.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
