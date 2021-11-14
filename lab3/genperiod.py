import ast
import click
import cv2
import cvtrack
import math
import numpy as np
import itertools
import pathlib
import sys
from process_data import averaged_peaks
from typing import List, TextIO, Tuple
from matplotlib import pyplot as plt


def parse_time(t: str, framerate: int = 30) -> float:
    if t.endswith("s"):
        return float(t[:-1]) * 1000
    if t.endswith("f"):
        return int(t[:-1]) / framerate * 1000
    if t.endswith("ms"):
        t = t[:-2]
    return float(t)


@click.command()
@click.argument("times_in", type=click.Path(exists=True, readable=True, path_type=pathlib.Path))
@click.argument("data_out", type=click.File("w"))
@click.option("--fx", type=click.FloatRange(min=0, min_open=True), default=None, help="X scaling factor")
@click.option("--fy", type=click.FloatRange(min=0, min_open=True), default=None, help="Y scaling factor")
@click.option("--merge-threshold", "-m", type=click.FloatRange(min=0), default=0.25, help="Minimum time between peaks for them to be recognized as distinct")
@click.option("--x-uncert", "--xu", type=float, default=0, help="Absolute uncertainty for every x value")
@click.option("--x-rel-uncert", "--xru", type=float, default=0, help="Relative uncertainty for every x value")
@click.option("--y-uncert", "--yu", type=float, default=0, help="Absolute uncertainty for every y value")
@click.option("--y-rel-uncert", "--yru", type=float, default=0, help="Relative uncertainty for every y value")
@click.option("--offset", "-o", type=float, default=0, help="Subtract an offset from all x values")
@click.option("--negate/--no-negate", "-n/-N", default=False, help="Negate x values")
@click.option("--plot/--no-plot", default=False, help="Plot extracted angle data")
@click.option("--peak-option", "-p", multiple=True, type=(str, str), help="Additional kwargs to pass to scipy.signal.find_peaks()")
def main(times_in: pathlib.Path, data_out: TextIO, fx: float, fy: float, merge_threshold: float, x_uncert: float,
         x_rel_uncert: float, y_uncert: float, y_rel_uncert: float, offset: float, negate: bool, plot: bool,
         peak_option: List[Tuple[str, str]]) -> None:
    """
    Generate period data.

    VID_IN is the input video and TIMES_IN is the input text file specifying the times of clips to use.
    """
    
    cap = None # type: cv2.VideoCapture
    current_x = 0
    x_step = 0
    peak_options = {arg: ast.literal_eval(val) for arg, val in peak_option}
    with times_in.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            pcs = line.split()
            if pcs[0].startswith("!"):
                pcs[0] = pcs[0][1:]
                if pcs[0] == "v" or pcs[0] == "video":
                    if pathlib.Path(pcs[1]).is_absolute():
                        vidpath = pcs[1]
                    else:
                        vidpath = str(times_in.with_name(pcs[1]))
                    print(f"Using video file {vidpath}")
                    cap = cv2.VideoCapture(vidpath)
                    if cap is None or not cap.isOpened():
                        print(f"Error: Video file {vidpath} not openable!")
                        sys.exit(1)
                elif pcs[0] == "echo":
                    print(line[6:])
                elif pcs[0] == "xval":
                    current_x = float(pcs[1])
                elif pcs[0] == "xstep":
                    x_step = float(pcs[1])
                elif pcs[0] == "xoffset":
                    offset = float(pcs[1])
                elif pcs[0] == "xnegate":
                    negate = pcs[1].lower() != "false"
                continue
            if cap is None:
                print("Error: A video file must be specified first with !v <file> or !video <file>.")
                sys.exit(1)
            if pcs[0] == "~":
                x_val = current_x
                current_x += x_step
            else:
                x_val = float(pcs[0])
            x_val -= offset
            if negate:
                x_val = -x_val
            
            time_range = pcs[1]
            start, stop = time_range.split("-")
            start = parse_time(start)
            stop = parse_time(stop)

            print(f"Processing x={x_val}, range {start}ms to {stop}ms")
            cap.set(cv2.CAP_PROP_POS_MSEC, start)
            time = []
            angle = []
            while True:
                success, img = cap.read()
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not success or ms > stop:
                    break
                ((x, y), (pivot_x, pivot_y)), _ = cvtrack.process_img(img, fx=fx, fy=fy)
                time.append(ms / 1000)
                angle.append(math.atan2(x - pivot_x, y - pivot_y))

            peak_x, peak_y, peak_uncert = averaged_peaks(np.array(time), np.array(angle), merge_threshold, options=peak_options)
            if plot:
                plt.scatter(time, angle)
                plt.scatter(peak_x, peak_y)
                plt.show()
            if len(peak_x) < 2:
                print(f"Error: Less than 2 peaks found for range {start}ms to {stop}ms ({time_range}). Check your ranges?")
                sys.exit(1)
            periods = np.fromiter((b - a for a, b in zip(peak_x, itertools.islice(peak_x, 1, None))), dtype=np.float64)
            period = np.mean(periods)
            period_uncert = np.std(periods) / np.sqrt(len(periods))
            print(f"Averaged {len(peak_x)} peaks for a period of {period}s")

            pu = max(period_uncert, y_uncert, abs(period * max(y_rel_uncert, max(u / t for u, t in zip(peak_uncert, peak_x)))))
            xu = max(x_uncert, abs(x_rel_uncert * x_val))
            data_out.write(f"{x_val} {period} {xu} {pu}\n")


if __name__ == "__main__":
    main()
