import cv2
import sys

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
    except ZeroDivisionError:
        cv2.imwrite("failure_img.png", img)
        cv2.imwrite("failure_pivot_binary.png", green_binary)
        raise ValueError("ERROR: Pivot not found in image! Failure images written.")

    return ((x, y), (pivot_x, pivot_y)), (binary, green_binary)


def main():
    pause = True

    vid_name = sys.argv[1]
    start_time = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    cap = cv2.VideoCapture(vid_name)
    wait_time = int(1000 / cap.get(cv2.CAP_PROP_FPS))

    if start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
    
    success, img = cap.read()
    if not success:
        sys.exit(1)

    ((x, y), (pivot_x, pivot_y)), (binary, green_binary) = process_img(img)
    img = cv2.resize(img, None, fx=0.35, fy=0.35)
    cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=cv2.FILLED)
    cv2.circle(img, (pivot_x, pivot_y), 3, (0, 0, 255), thickness=cv2.FILLED)

    cv2.imshow(vid_name, img)
    cv2.imshow("binary", binary)
    cv2.imshow("pivot binary", green_binary)

    while True:
        key = cv2.waitKey(wait_time)
        if key == ord('q'):
            break
        elif key == ord(' '):
            pause = not pause

        if pause:
            continue

        success, img = cap.read()
        if not success:
            break
        ((x, y), (pivot_x, pivot_y)), (binary, green_binary) = process_img(img)
        img = cv2.resize(img, None, fx=0.35, fy=0.35)
        cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=cv2.FILLED)
        cv2.circle(img, (pivot_x, pivot_y), 3, (0, 0, 255), thickness=cv2.FILLED)

        cv2.imshow(vid_name, img)
        cv2.imshow("binary", binary)
        cv2.imshow("pivot binary", green_binary)
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
