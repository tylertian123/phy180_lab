import functools
import click
import numpy as np
from fit_length import load_data
from typing import List, TextIO, Tuple
from matplotlib import pyplot as plt
from scipy import odr


# def fitfunc(p: List[float], m: float) -> float:
#     return p[0] + p[1] * m ** p[2]

def fitfunc(p: List[float], m: float) -> float:
    return p[0] + p[1] * np.log(m)


def do_fit(x_data: np.ndarray, y_data: np.ndarray, x_uncert: np.ndarray, y_uncert: np.ndarray, guesses) -> Tuple[List[float], List[float]]:
    model = odr.Model(fitfunc)
    data = odr.RealData(x_data, y_data, sx=x_uncert, sy=y_uncert)
    output = odr.ODR(data, model, beta0=guesses).run()
    print(output.stopreason)
    return (output.beta, output.sd_beta)


@click.command()
@click.argument("data_in", type=click.File("r", encoding="utf-8"))
@click.option("--sep", "-s", type=str, default=None, help="Separator in the data file")
@click.option("--save-residuals", type=click.File("w", encoding="utf-8"), default=None, help="Save residuals to a file")
def main(data_in: TextIO, sep: str, save_residuals: TextIO):
    """
    Fit period to a function of mass for lab 3b.
    """
    x_data, y_data, x_uncert, y_uncert = load_data(data_in, uncert=True, sep=sep)
    #params, uncert = do_fit(x_data, y_data, x_uncert, y_uncert, (2.079, 0, 0.5))
    params, uncert = do_fit(x_data, y_data, x_uncert, y_uncert, (2.079, 1))
    print(params)
    print(uncert)
    bestfit = functools.partial(fitfunc, p=params)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2.set_xscale("log")
    fig.subplots_adjust(hspace=0.6)
    start = min(x_data)
    stop = max(x_data)
    xvals = np.linspace(start, stop, 200)
    yvals = bestfit(m=xvals)
    residuals = y_data - bestfit(m=x_data)

    if save_residuals is not None:
        for x, r, xerr, rerr in zip(x_data, residuals, x_uncert, y_uncert):
            save_residuals.write(f"{x} {r} {xerr} {rerr}\n")

    ax1.errorbar(x_data, y_data, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Collected Data")
    ax1.plot(xvals, yvals, label="Best Fit Line")
    ax1.set_xlabel("Mass $m$ (g)")
    ax1.set_ylabel("Period $T$ (s)")
    ax1.set_title("Data & Best Fit Curve")
    ax1.legend(loc="best")
    ax2.errorbar(x_data, y_data, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Collected Data")
    ax2.plot(xvals, yvals, label="Best Fit Line")
    ax2.set_xlabel("Mass $m$ (g)")
    ax2.set_ylabel("Period $T$ (s)")
    ax2.set_title("Data & Best Fit Curve")
    ax2.legend(loc="best")
    ax3.errorbar(x_data, residuals, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Residuals")
    ax3.axhline(0)

    plt.show()


if __name__ == "__main__":
    main()
