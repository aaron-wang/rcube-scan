import numpy as np
import cv2 as cv


cap = cv.VideoCapture(0,cv.CAP_DSHOW)

object_detector = cv.createBackgroundSubtractorMOG2(history=1000,varThreshold=60)
while 1:
    _, frame = cap.read()



    height, width, _ = frame.shape
    # # Extract Region of interest
    # roi = frame[50: 550,100: 520]
    roi = frame[0:,0:]
    mask = object_detector.apply(roi)
    # # 1. Object Detection
    # mask = object_detector.apply(roi)
    _, mask = cv.threshold(mask, 0, 50, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 20000:
            # cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print(area)
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame,[box],0,(0,255,0),2)

    cv.imshow("mask",mask)
    cv.imshow("frame",frame)
    # cv.imshow("roi",roi)
    k = cv.waitKey(5)
    if k == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
