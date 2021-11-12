import cv2

def center(img):
    moments = cv2.moments(img)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y

def process_img(img, fx=None, fy=None):
    if fx is not None or fy is not None:
        img = cv2.resize(img, None, fx=fx, fy=fy)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    binary1 = cv2.inRange(hsv, (0, 75, 80), (10, 255, 255))
    binary2 = cv2.inRange(hsv, (170, 75, 80), (180, 255, 255))
    binary = binary1 + binary2

    try:
        # Get largest contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.imwrite("failure_img.png", img)
            cv2.imwrite("failure_binary.png", binary)
            raise ValueError("ERROR: Bob not found in image! Failure images written.")
        largest = max(contours, key=cv2.contourArea)
        x, y = center(largest)

        green_binary = cv2.inRange(hsv, (37, 50, 80), (60, 255, 255))
        try:
            pivot_x, pivot_y = center(green_binary)
        except ZeroDivisionError as e:
            cv2.imwrite("failure_img.png", img)
            cv2.imwrite("failure_pivot_binary.png", green_binary)
            raise ValueError("ERROR: Pivot not found in image! Failure images written.") from e
    except ZeroDivisionError as e:
        try:
            cv2.imwrite("failure_img.png", img)
        except NameError:
            print("ERROR: Unable to write failure image.")
        raise ValueError("ERROR: Something was not found! Failure image written.") from e

    return ((x, y), (pivot_x, pivot_y)), (binary, green_binary)