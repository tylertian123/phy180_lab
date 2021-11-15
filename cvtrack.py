import cv2
import numpy as np
from typing import Tuple

HSV = Tuple[int, int, int]

BOB_THRESH = ((170, 75, 80), (10, 255, 255))
PIVOT_THRESH = ((37, 50, 80), (50, 255, 255))


def generate_thresh(low: HSV, high: HSV) -> Tuple[Tuple[HSV, HSV], Tuple[HSV, HSV]]:
    if low[0] <= high[0]:
        return ((low, high), ((0, 0, 0), (0, 0, 0)))
    return ((low, (180, high[1], high[2])), ((0, low[1], low[2]), high))


def thresh_img(img, thresh: Tuple[Tuple[HSV, HSV], Tuple[HSV, HSV]]):
    bin1 = cv2.inRange(img, thresh[0][0], thresh[0][1])
    bin2 = cv2.inRange(img, thresh[1][0], thresh[1][1])
    binary = bin1 + bin2
    return binary


def center(img):
    moments = cv2.moments(img)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


def largest_blob(img: np.ndarray, binary: np.ndarray, raise_on_fail: bool = False) -> Tuple[int, int]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if raise_on_fail:
            cv2.imwrite("failure_img.png", img)
            cv2.imwrite("failure_binary.png", binary)
            raise ValueError("ERROR: Object not found! Failure images written.")
        else:
            return (None, None)
    else:
        try:
            largest = max(contours, key=cv2.contourArea)
            return center(largest)
        except ZeroDivisionError as e:
            if raise_on_fail:
                cv2.imwrite("failure_img.png", img)
                cv2.imwrite("failure_binary.png", binary)
                raise ValueError("ERROR: Object not found! Failure images written.")
            else:
                return (None, None)


def process_img(img, fx=None, fy=None, bob_thresh=BOB_THRESH, pivot_thresh=PIVOT_THRESH, raise_on_fail=True):
    if fx is not None or fy is not None:
        img = cv2.resize(img, None, fx=fx, fy=fy)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binary = thresh_img(hsv, generate_thresh(*bob_thresh))

    x, y = largest_blob(img, binary, raise_on_fail)

    green_binary = thresh_img(hsv, generate_thresh(*pivot_thresh))
    pivot_x, pivot_y = largest_blob(img, green_binary, raise_on_fail)

    return ((x, y), (pivot_x, pivot_y)), (binary, green_binary)