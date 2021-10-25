import argparse
from typing import TextIO

from fit import load_data


def main(data_in: TextIO, data_out: TextIO, x_uncert: float, y_uncert: float, x_rel_uncert: float,
         y_rel_uncert: float, x_dep: bool, y_dep: bool) -> None:
    x_data, y_data = load_data(data_in)
    xu = []
    yu = []
    for x, y in zip(x_data, y_data):
        x_unc = max(x_uncert, abs(x_rel_uncert * x))
        y_unc = max(y_uncert, abs(y_rel_uncert * y))
        if x_dep:
            x_unc = max(x_unc, abs((y_unc / y) * x))
        if y_dep:
            y_unc = max(y_unc, abs((x_unc / x) * y))
        xu.append(x_unc)
        yu.append(y_unc)
    for x, y, x_unc, y_unc in zip(x_data, y_data, xu, yu):
        data_out.write(f"{x} {y} {x_unc} {y_unc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate uncertainties for processed data for lab 2.")
    parser.add_argument("data_in", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("data_out", type=argparse.FileType("w", encoding="utf-8"))
    parser.add_argument("--x-uncert", type=float, default=0.0)
    parser.add_argument("--y-uncert", type=float, default=0.0)
    parser.add_argument("--x-rel-uncert", type=float, default=0.0)
    parser.add_argument("--y-rel-uncert", type=float, default=0.0)
    parser.add_argument("--x-dep", action="store_true")
    parser.add_argument("--y-dep", action="store_true")
    main(**vars(parser.parse_args()))