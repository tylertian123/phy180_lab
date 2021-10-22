from typing import TextIO, Tuple
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import argparse
import itertools
import statistics


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


def averaged_peaks(x_data: np.ndarray, y_data: np.ndarray, merge_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    peaks, _ = signal.find_peaks(y_data, height=0, threshold=0)
    peak_x = []
    peak_y = []
    i = 0
    while i < len(peaks):
        n = 1
        for j in range(i + 1, len(peaks)):
            if x_data[peaks[j]] - x_data[peaks[j - 1]] < merge_threshold:
                n += 1
            else:
                break
        x = statistics.median(x_data[peaks[k + i]] for k in range(n))
        y = max(y_data[peaks[k + i]] for k in range(n))
        peak_x.append(x)
        peak_y.append(y)
        i += n
    return np.array(peak_x), np.array(peak_y)


def main(data_in: TextIO, data_out: TextIO, merge_threshold: float, graph: bool, half_oscillations: bool) -> None:
    x_data, y_data = load_data(data_in)
    max_x, max_y = averaged_peaks(x_data, y_data, merge_threshold)
    min_x, min_y = averaged_peaks(x_data, -y_data, merge_threshold)
    # Because it was negated when passed into averaged_peaks
    min_y = -min_y
    if half_oscillations:
        # Merge both
        peak_x = []
        peak_y = []
        i = j = 0
        while i < len(min_x) and j < len(max_x):
            if min_x[i] < max_x[j]:
                peak_x.append(min_x[i])
                peak_y.append(min_y[i])
                i += 1
            else:
                peak_x.append(max_x[j])
                peak_y.append(max_y[j])
                j += 1
        for k in range(i, len(min_x)):
            peak_x.append(min_x[k])
            peak_y.append(min_y[k])
        for k in range(j, len(max_x)):
            peak_x.append(max_x[k])
            peak_y.append(max_y[k])
        # Ideally this would go from the most negative amplitude to the most positive but I don't care enough
        for y, x1, x2 in zip(peak_y, peak_x, itertools.islice(peak_x, 1, None)):
            data_out.write(f"{y} {(x2 - x1) * 2}\n")
    else:
        # Start from the most negative amplitude (first min) and go to the most positive
        for y, x1, x2 in itertools.chain(
                zip(min_y, min_x, itertools.islice(min_x, 1, None)),
                zip(reversed(max_y), itertools.islice(reversed(max_x), 1, None), reversed(max_x))):
            data_out.write(f"{y} {x2 - x1}\n")
    if graph:
        plt.scatter(x_data, y_data, label="Data")
        plt.scatter(max_x, max_y, label="Maxima")
        plt.scatter(min_x, min_y, label="Minima")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.title("Extrema")
        plt.legend(loc="upper right")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data from gendata.py into data for lab 2.")
    parser.add_argument("data_in", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("data_out", type=argparse.FileType("w", encoding="utf-8"))
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--half-oscillations", action="store_true")
    main(**vars(parser.parse_args()))
