import click
from fit_length import load_data
from typing import TextIO
from matplotlib import pyplot as plt


@click.command()
@click.argument("data_in", type=click.File("r", encoding="utf-8"))
@click.option("--sep", "-s", type=str, default=None, help="Separator in the data file")
@click.option("--save-residuals", type=click.File("w", encoding="utf-8"), default=None, help="Save residuals to a file")
def main(data_in: TextIO, sep: str, save_residuals: TextIO):
    """
    Fit period to a function of mass for lab 3b.
    """
    x_data, y_data, x_uncert, y_uncert = load_data(data_in, uncert=True, sep=sep)

    fig, (ax1) = plt.subplots(1, 1)
    ax1.set_xscale("log")
    fig.subplots_adjust(hspace=0.6)

    ax1.errorbar(x_data, y_data, xerr=x_uncert, yerr=y_uncert, fmt="o", label="Collected Data")
    ax1.set_xlabel("Mass $m$ (g)")
    ax1.set_ylabel("Period $T$ (s)")
    ax1.set_title("Data & Best Fit Curve")
    ax1.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    main()
