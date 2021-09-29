import cv2

cap = cv2.VideoCapture("test_vid.mp4")
_, img = cap.read()
cv2.imwrite("firstframe.png", img)
