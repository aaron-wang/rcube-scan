import numpy as np
import cv2 as cv

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('vid/dog.mp4')

print(f"width {cap.get(3)} \\ height {cap.get(4)}")

# Main loop
while True:
    ret, frame = cap.read()

    width = int(cap.get(3))  
    height = int(cap.get(4))  

    img = cv.line(frame, (0,0),(width,height), (255,0,0), 10)
    img = cv.line(img, (0,height),(width,0), (0,255,0), 2)

    img = cv.rectangle(img,(100,100),(300,300),(255,128,128),5)

    img = cv.circle(img,(100,100),50,(255,0,0),5)

    font = cv.FONT_HERSHEY_SIMPLEX

    img = cv.putText(img,"Hello",(200,height-100),font,1,(0,0,0),2,cv.LINE_AA)

    cv.imshow('frame',frame)

    if (cv.waitKey(20) == ord('q')):
        break

cap.release()
cv.destroyAllWindows()
# img = cv.imread('img/hi2.png')
# cv.imshow('Image',img)

# def rescaleFrame(frame, scale=0.25):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimension = (width,height)
#     return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)

# def changeRes(width,height):
#     capture.set(3,width)
#     capture.set(4,height)

# capture = cv.VideoCapture('vid/dog.mp4')

# sys.exit("Hello")

# cv.imshow('Image2',rescaleFrame(img))
# while True:
#     isTrue, frame = capture.read()

#     frame_resized = rescaleFrame(frame,scale=.2)
    
#     cv.imshow('Video',frame)
#     cv.imshow('Video Resized',frame_resized)

#     if (cv.waitKey(20) & 255 == ord('d')):
#         break

# capture.release()
# cv.destroyAllWindows()
# cv.waitKey(0)
# cv.waitKey(0)
