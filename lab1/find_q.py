from fit import load_data
import sys
import numpy as np
from scipy import signal

Q_DIVISOR = 3

def main():
    x_data, y_data = load_data(sys.argv)
    amp = abs(y_data[0])
    mag = np.exp(-np.pi / Q_DIVISOR) * amp
    maxima, _ = signal.find_peaks(y_data, height=0, threshold=0)
    minima, _ = signal.find_peaks(-y_data, height=0, threshold=0)

    it = iter(maxima)
    next(it)
    for i, j in zip(maxima, it):
        if x_data[j] - x_data[i] < 0.5:
            print("Error: Invalid peak detected!")
            sys.exit(1)
    
    for i in range(min(len(minima), len(maxima))):
        # Find first minimum or maximum with value less than or equal to mag
        max_diff = mag - abs(y_data[maxima[i]])
        min_diff = mag - abs(y_data[minima[i]])
        if max_diff >= 0 and min_diff >= 0:
            # If we're taking the maximum and the data started negative
            # or if we're taking the minimum and the data started positive
            # then there's a missing half cycle
            if max_diff < min_diff:
                osc = i + 1 if y_data[0] > 0 else i + 0.5
                t = x_data[maxima[i]]
                y = y_data[maxima[i]]
            else:
                osc = i + 1 if y_data[0] < 0 else i + 0.5
                t = x_data[minima[i]]
                y = y_data[minima[i]]
            break
        if max_diff >= 0:
            osc = i + 1 if y_data[0] > 0 else i + 0.5
            t = x_data[maxima[i]]
            y = y_data[maxima[i]]
            break
        if min_diff >= 0:
            osc = i + 1 if y_data[0] < 0 else i + 0.5
            t = x_data[minima[i]]
            y = y_data[minima[i]]
            break
    else:
        print(f"Error: No peak found with amplitude <= {mag}. Try adjusting the Q divisor.")
        sys.exit(1)
    print(f"Found peak: t={t}, angle={y}")
    print("Q:", osc * Q_DIVISOR)

if __name__ == "__main__":
    main()
