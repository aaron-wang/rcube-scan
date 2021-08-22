import numpy as np
import cv2 as cv



cap = cv.VideoCapture(0,cv.CAP_DSHOW)

def myContour(colorName, mask):
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame,contours,-1,(0,0,255),3)
    if (len(contours) >= 1):
        c_threshold = 500
            # c_threshold = 1000
        # text_color = (255,255,255)
        text_color = (0,0,0)
        for c in contours:
            if (cv.contourArea(c) > c_threshold):
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                # if (colorName=="red"):
                    # colorName="amog us"
                cv.putText(frame,colorName,(x+6,y+h//2),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)

# bob = 96

while True:
    ret,frame = cap.read()
    # frame = cv.imread("img/sus.jpg")
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV) 

    #yellow
    # or 160 replaces 80
    lyellow = np.array([19,180,80])
    uyellow = np.array([40,255,255])
    y_mask = cv.inRange(hsv_frame,lyellow,uyellow)
    yellow = cv.bitwise_and(frame,frame,mask=y_mask)

    #green
    lgreen = np.array([70,120,100])
    hgreen = np.array([96,255,255])
    g_mask = cv.inRange(hsv_frame,lgreen,hgreen)
    green = cv.bitwise_and(frame,frame,mask=g_mask)

    #blue
    lblue = np.array([100,97,78])
    hblue = np.array([123,255,255])
    b_mask = cv.inRange(hsv_frame,lblue,hblue)
    blue = cv.bitwise_and(frame,frame,mask=b_mask)

    #red
    lred = np.array([0,99,73])
    hred = np.array([10,255,255])
    r_mask = cv.inRange(hsv_frame,lred,hred)
    lred2 = np.array([170,99,73])
    hred2 = np.array([180,255,255])
    r_mask2 = cv.inRange(hsv_frame,lred2,hred2)
    r_mask |= r_mask2
    red = cv.bitwise_and(frame,frame,mask=r_mask)

    # white
    lwhite = np.array([0,0,200])
    hwhite = np.array([46,50,255])
    w_mask = cv.inRange(hsv_frame,lwhite,hwhite)
    white = cv.bitwise_and(frame,frame,mask=w_mask)

    myContour("yellow",y_mask)
    cv.imshow('yellow mask',yellow)

    myContour("green",g_mask)
    cv.imshow('green mask',green)

    myContour("blue",b_mask)
    cv.imshow('blue mask',blue)

    myContour("red",r_mask)
    cv.imshow("red mask", red)

    myContour("white",w_mask)
    cv.imshow("white mask",white)





    cv.imshow('frame',frame)

    # HS V - hue, saturation, value (brightness)
    key = cv.waitKey(1)
    if (key == ord('q')): 
        break
    if (key == ord('w')):
        bob+=1
        print(bob)
    if (key == ord('d')): 
        bob-=1
        print(bob)

cap.release()
cv.destroyAllWindows()