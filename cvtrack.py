import cv2
from typing import Tuple


HSV = Tuple[int, int, int]

BOB_THRESH = ((170, 75, 80), (10, 255, 255))
PIVOT_THRESH = ((37, 50, 80), (60, 255, 255))


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


def process_img(img, fx=None, fy=None, bob_thresh=BOB_THRESH, pivot_thresh=PIVOT_THRESH, raise_on_fail=True):
    if fx is not None or fy is not None:
        img = cv2.resize(img, None, fx=fx, fy=fy)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    binary = thresh_img(hsv, generate_thresh(*bob_thresh))

    # Get largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if raise_on_fail:
            cv2.imwrite("failure_img.png", img)
            cv2.imwrite("failure_binary.png", binary)
            raise ValueError("ERROR: Bob not found in image! Failure images written.")
        else:
            x = y = None
    else:
        try:
            largest = max(contours, key=cv2.contourArea)
            x, y = center(largest)
        except ZeroDivisionError as e:
            if raise_on_fail:
                cv2.imwrite("failure_img.png", img)
                cv2.imwrite("failure_binary.png", binary)
                raise ValueError("ERROR: Bob not found in image! Failure images written.")
            else:
                x = y = None

    green_binary = thresh_img(hsv, generate_thresh(*pivot_thresh))
    try:
        pivot_x, pivot_y = center(green_binary)
    except ZeroDivisionError as e:
        if raise_on_fail:
            cv2.imwrite("failure_img.png", img)
            cv2.imwrite("failure_pivot_binary.png", green_binary)
            raise ValueError("ERROR: Pivot not found in image! Failure images written.") from e
        else:
            pivot_x = pivot_y = None

    return ((x, y), (pivot_x, pivot_y)), (binary, green_binary)