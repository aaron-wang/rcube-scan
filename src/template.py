import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)

# print(cv.video)

while 1:
    # Read frame
    _, frame = cap.read()

    # BGR --> HSV
    # this is a frame
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define color
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # apply threshold
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # mask bitwise& original

    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv.waitKey(5)
    if k == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
