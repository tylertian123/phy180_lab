import click
from typing import TextIO
from fit_length import fitfunc

spring_k = 194
g = 9.806
length = 1.0745

k = 2.018099069322445
n = 0.4965279402153859
l0 = -0.01272678366160228

@click.command()
@click.argument("data_in", type=click.File("r", encoding="utf-8"))
@click.argument("data_out", type=click.File("w", encoding="utf-8"))
def main(data_in: TextIO, data_out: TextIO) -> None:
    for line in data_in:
        x, y, xu, yu = (float(i) for i in line.split())
        correction = fitfunc(length + (x / 1000 * g) / spring_k, k, n, l0) - fitfunc(length, k, n, l0)
        y -= correction
        data_out.write(f"{x} {y} {xu} {yu}\n")


if __name__ == "__main__":
    main()
