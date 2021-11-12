import sys
from typing import Optional
import cv2
import argparse
import cvtrack

def main(vid_name: str, start_time: int, frame_delay: Optional[int], fx: Optional[float], fy: Optional[float]):
    pause = True

    cap = cv2.VideoCapture(vid_name)
    wait_time = int(1000 / cap.get(cv2.CAP_PROP_FPS)) if frame_delay is None else frame_delay

    if start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

    success, img = cap.read()
    if not success:
        sys.exit(1)

    ((x, y), (pivot_x, pivot_y)), (binary, green_binary) = cvtrack.process_img(img, fx=fx, fy=fy)
    if fx is not None and fy is not None:
        img = cv2.resize(img, None, fx=fx, fy=fy)
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
        ((x, y), (pivot_x, pivot_y)), (binary, green_binary) = cvtrack.process_img(img, fx=fx, fy=fy)
        if fx is not None and fy is not None:
            img = cv2.resize(img, None, fx=fx, fy=fy)
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
    parser = argparse.ArgumentParser(description="Visual tracker demo.")
    parser.add_argument("vid_name", type=str)
    parser.add_argument("start_time", type=int, default=0, nargs="?")
    parser.add_argument("--frame-delay", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    main(**vars(parser.parse_args()))
