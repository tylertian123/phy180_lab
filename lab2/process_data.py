from typing import List, Optional, TextIO, Tuple
from scipy import signal
from matplotlib import pyplot as plt
import tikzplotlib
import numpy as np
import argparse
import itertools


def load_data(file: TextIO) -> Tuple[np.ndarray, np.ndarray]:
    # Load data
    x_data = []
    y_data = []
    init_time = None
    for line in file:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pcs = line.split()
        time = float(pcs[0])
        angle = float(pcs[1])
        if init_time is None:
            init_time = time
        time -= init_time
        x_data.append(time)
        y_data.append(angle)
    return np.array(x_data), np.array(y_data)


def averaged_peaks(x_data: np.ndarray, y_data: np.ndarray, merge_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    peaks, _ = signal.find_peaks(y_data, height=0, threshold=0)
    peak_x = []
    peak_y = []
    x_stdev = []
    i = 0
    while i < len(peaks):
        n = 1
        for j in range(i + 1, len(peaks)):
            if x_data[peaks[j]] - x_data[peaks[j - 1]] < merge_threshold:
                n += 1
            else:
                break
        x = sum(x_data[peaks[k + i]] for k in range(n)) / n
        y = sum(y_data[peaks[k + i]] for k in range(n)) / n
        peak_x.append(x)
        peak_y.append(y)
        x_stdev.append(np.std(np.fromiter((x_data[peaks[k + i]] for k in range(n)), float, n)))
        i += n
    return np.array(peak_x), np.array(peak_y), np.array(x_stdev)


def main(data_in: TextIO, data_out: TextIO, merge_threshold: float, graph: bool, save_graph: Optional[TextIO],
         xlim: List[float], ylim: List[float], no_write: bool, export_extrema: TextIO, n: int) -> None:
    x_data, y_data = load_data(data_in)
    if n is not None:
        x_data = x_data[:n]
        y_data = y_data[:n]
    max_x, max_y, max_x_stdev = averaged_peaks(x_data, y_data, merge_threshold)
    min_x, min_y, min_x_stdev = averaged_peaks(x_data, -y_data, merge_threshold)
    min_y = -min_y
    if export_extrema is not None:
        for x, y in zip(itertools.chain(max_x, min_x), itertools.chain(max_y, min_y)):
            export_extrema.write(f"{x} {y}\n")
    print("x stdev: ", max(np.max(max_x_stdev), np.max(min_x_stdev)))
    # Because it was negated when passed into averaged_peaks
    if not no_write:
        with open(data_out, "w", encoding="utf-8") as out_file:
            for i in range(len(min_x) - 1):
                y = min_y[i]
                dx = min_x[i + 1] - min_x[i]
                unc = max(min_x_stdev[i], min_x_stdev[i + 1])
                out_file.write(f"{y} {dx} {0} {unc}\n")
            for i in range(len(max_x) - 1):
                y = max_y[i]
                dx = max_x[i + 1] - max_x[i]
                unc = max(max_x_stdev[i], max_x_stdev[i + 1])
                out_file.write(f"{y} {dx} {0} {unc}\n")
    if graph:
        plt.scatter(x_data, y_data, label="Data", s=4)
        plt.scatter(max_x, max_y, label="Maxima", s=9)
        plt.scatter(min_x, min_y, label="Minima", s=9, c="#00d000")
        plt.xlabel("Time $t$ (s)")
        plt.ylabel("Angle $\\theta$ (rad)")
        plt.title("Extrema")
        plt.legend(loc="upper right")
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        if save_graph is not None:
            tikzplotlib.save(save_graph)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data from gendata.py into data for lab 2.")
    parser.add_argument("data_in", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("data_out", type=str)
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--save-graph", type=str, default=None)
    parser.add_argument("--xlim", type=float, nargs=2, default=None)
    parser.add_argument("--ylim", type=float, nargs=2, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--export-extrema", type=argparse.FileType("w", encoding="utf-8"), default=None)
    parser.add_argument("-n", type=int, default=None)
    main(**vars(parser.parse_args()))
