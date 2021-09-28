import cv2
import math
from sys import argv

START_TIME = 140 * 1000 // 30
FRAME_TIME_INTERVAL = 100
CENTER = (287, 155)

vid_name = argv[1]
out_name = argv[2]

def process_img(img):
    global time
    img = cv2.resize(img, None, fx=0.75, fy=0.75)
    height, width, _ = img.shape
    cv2.putText(img, str(time), (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2, color=(255, 0, 0))
    return img

cap = cv2.VideoCapture(vid_name)
out_file = open(out_name, "a", encoding="utf-8")
def cleanup():
    out_file.close()
    cap.release()
    cv2.destroyAllWindows()

# First frame
time = START_TIME
print(time)
cap.set(cv2.CAP_PROP_POS_MSEC, time)
success, img = cap.read()
if not success:
    cleanup()
    raise ValueError("Couldn't read frame from capture")

img = process_img(img)
current_frame = img


def click(event, x, y, flags, param):
    global time, current_frame
    if event == cv2.EVENT_MOUSEMOVE:
        img = current_frame.copy()
        cv2.line(img, CENTER, (x, y), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Image", img)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Process angle
        angle = math.atan2(x - CENTER[0], y - CENTER[1])
        out_file.write(f"{str(time / 1000)} {angle}\n")
        print(f"Written: {time / 1000}\t{angle} ({math.degrees(angle)} degrees)")

        time += FRAME_TIME_INTERVAL
        cap.set(cv2.CAP_PROP_POS_MSEC, time)
        success, current_frame = cap.read()
        if not success:
            cleanup()
        current_frame = process_img(current_frame)
        img = current_frame.copy()
        cv2.line(img, CENTER, (x, y), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Image", img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("right")

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click)
cv2.imshow("Image", img)

while True:
    cv2.waitKey(100)
    if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        cleanup()

