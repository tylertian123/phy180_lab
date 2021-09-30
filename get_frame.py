import cv2
from sys import argv

time = int(argv[2])

cap = cv2.VideoCapture(argv[1])
cap.set(cv2.CAP_PROP_POS_MSEC, time)
_, img = cap.read()
cv2.imwrite("frame.png", img)
