import cv2
import click
import cvtrack
import numpy as np
from typing import List


@click.command()
@click.argument("img", type=click.Path(exists=True, readable=True))
@click.option("--bob-thresh-low", nargs=3, type=int, default=cvtrack.BOB_THRESH[0])
@click.option("--bob-thresh-high", nargs=3, type=int, default=cvtrack.BOB_THRESH[1])
@click.option("--pivot-thresh-low", nargs=3, type=int, default=cvtrack.PIVOT_THRESH[0])
@click.option("--pivot-thresh-high", nargs=3, type=int, default=cvtrack.PIVOT_THRESH[1])
def main(img: str, bob_thresh_low: cvtrack.HSV, bob_thresh_high: cvtrack.HSV, pivot_thresh_low: cvtrack.HSV, pivot_thresh_high: cvtrack.HSV) -> None:
    blow = [i for i in bob_thresh_low]
    bhigh = [i for i in bob_thresh_high]
    plow = [i for i in pivot_thresh_low]
    phigh = [i for i in pivot_thresh_high]
    img = cv2.imread(img)

    def show_imgs() -> None:
        print(blow, bhigh, plow, phigh)
        ((x, y), (pivot_x, pivot_y)), (bb, pb) = cvtrack.process_img(img,
            bob_thresh=(tuple(blow), tuple(bhigh)), pivot_thresh=(tuple(plow), tuple(phigh)),
            raise_on_fail=False)
        img2 = np.copy(img)
        if x is not None and y is not None:
            cv2.circle(img2, (x, y), 3, (0, 255, 0), thickness=cv2.FILLED)
        if pivot_x is not None and pivot_y is not None:
            cv2.circle(img2, (pivot_x, pivot_y), 3, (0, 0, 255), thickness=cv2.FILLED)
        cv2.imshow("Image", img2)
        cv2.imshow("Bob", bb)
        cv2.imshow("Pivot", pb)
    
    show_imgs()
    callbacks = []
    for color in (blow, bhigh, plow, phigh):
        cb = []
        for i in range(3):
            def _on_change(val: int, color=color, i=i):
                color[i] = val
                show_imgs()
            cb.append(_on_change)
        callbacks.append(cb)
    cv2.createTrackbar("Bob Hue (Low)", "Bob", blow[0], 180, callbacks[0][0])
    cv2.createTrackbar("Bob Hue (High)", "Bob", bhigh[0], 180, callbacks[1][0])
    cv2.createTrackbar("Bob Saturation (Low)", "Bob", blow[1], 180, callbacks[0][1])
    cv2.createTrackbar("Bob Saturation (High)", "Bob", bhigh[1], 180, callbacks[1][1])
    cv2.createTrackbar("Bob Value (Low)", "Bob", blow[2], 180, callbacks[0][2])
    cv2.createTrackbar("Bob Value (High)", "Bob", bhigh[2], 180, callbacks[1][2])
    cv2.createTrackbar("Pivot Hue (Low)", "Pivot", plow[0], 180, callbacks[2][0])
    cv2.createTrackbar("Pivot Hue (High)", "Pivot", phigh[0], 180, callbacks[3][0])
    cv2.createTrackbar("Pivot Saturation (Low)", "Pivot", plow[1], 180, callbacks[2][1])
    cv2.createTrackbar("Pivot Saturation (High)", "Pivot", phigh[1], 180, callbacks[3][1])
    cv2.createTrackbar("Pivot Value (Low)", "Pivot", plow[2], 180, callbacks[2][2])
    cv2.createTrackbar("Pivot Value (High)", "Pivot", phigh[2], 180, callbacks[3][2])

    while True:
        k = cv2.waitKey(100)
        if k == ord("q"):
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty("Pivot", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty("Bob", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
