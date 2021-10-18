from typing import TextIO
import cv2
import math
import argparse
import cvtrack

def main(vid_name: str, out_file: TextIO, skip_frames: int, start_time: int):
    cap = cv2.VideoCapture(vid_name)
    if start_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

    while True:
        success, img = cap.read()
        if not success:
            print("Finished")
            break
        ((x, y), (pivot_x, pivot_y)), _ = cvtrack.process_img(img)
        angle = math.atan2(x - pivot_x, y - pivot_y)
        time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # At the end of the video the time is zero for some reason
        # This only happens for a few frame so we'll just skip them
        if time != 0:
            out_file.write(f"{time} {angle}\n")
            print(time, "\t", angle, sep="")
        else:
            print("Skipped a frame")
        for _ in range(skip_frames):
            cap.read()

    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get time vs angle data from video")
    parser.add_argument("vid_name", type=str)
    parser.add_argument("out_file", type=argparse.FileType("w", encoding="utf-8"))
    parser.add_argument("start_time", type=int, default=0, nargs="?")
    parser.add_argument("--skip-frames", type=int, default=3)
    main(**vars(parser.parse_args()))
