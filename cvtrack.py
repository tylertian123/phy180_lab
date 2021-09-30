import cv2
import math
import sys
import cvtrack_core as track


SKIP_FRAMES = 2

def main():
    vid_name = sys.argv[1]
    out_name = sys.argv[2]
    start_time = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(vid_name)
    if start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
    out_file = open(out_name, "w", encoding="utf-8")

    while True:
        success, img = cap.read()
        if not success:
            print("Finished")
            break
        ((x, y), (pivot_x, pivot_y)), _ = track.process_img(img)
        angle = math.atan2(x - pivot_x, y - pivot_y)
        time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # At the end of the video the time is zero for some reason
        # This only happens for a few frame so we'll just skip them
        if time != 0:
            out_file.write(f"{time} {angle}\n")
            print(time, "\t", angle, sep="")
        else:
            print("Skipped a frame")
        for _ in range(SKIP_FRAMES):
            cap.read()

    out_file.close()


if __name__ == "__main__":
    main()
