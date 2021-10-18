import cv2

def center(img):
    moments = cv2.moments(img)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y

def process_img(img):
    img = cv2.resize(img, None, fx=0.35, fy=0.35)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    binary1 = cv2.inRange(hsv, (0, 153, 100), (15, 255, 255))
    binary2 = cv2.inRange(hsv, (174, 153, 100), (180, 255, 255))
    binary = binary1 + binary2

    # Get largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imwrite("failure_img.png", img)
        cv2.imwrite("failure_binary.png", binary)
        raise ValueError("ERROR: Bob not found in image! Failure images written.")
    largest = max(contours, key=cv2.contourArea)
    x, y = center(largest)

    green_binary = cv2.inRange(hsv, (60, 63, 150), (80, 255, 255))
    try:
        pivot_x, pivot_y = center(green_binary)
    except ZeroDivisionError as e:
        cv2.imwrite("failure_img.png", img)
        cv2.imwrite("failure_pivot_binary.png", green_binary)
        raise ValueError("ERROR: Pivot not found in image! Failure images written.") from e

    return ((x, y), (pivot_x, pivot_y)), (binary, green_binary)