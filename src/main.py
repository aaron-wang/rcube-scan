import numpy as np
import cv2 as cv

import math

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

def myContour(colorName, mask):
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) >= 1):
        c_threshold = 500
        text_color = (255,255,255)
        if (colorName=="yellow"): text_color = (0,0,0)
        if (colorName=="white"): text_color = (0,0,0)
        for c in contours:
            if (cv.contourArea(c) > c_threshold):
                x, y, w, h = cv.boundingRect(c)
                if (colorName=="white" or colorName=="red" or colorName=="orange"):
                    if (w > 1.2 * h or h > 1.2*w or h > 120):
                        continue
                # General rectangle contour
                # cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # Rotated rectangle
                rect = cv.minAreaRect(c)
                rot_area = rect[1][0] * rect[1][1]
                box = cv.boxPoints(rect)
                box = np.int0(box)

                # Min circle
                (cx,cy),radius = cv.minEnclosingCircle(c)
                center = (int(cx),int(cy))
                circ_area = math.pi * radius * radius
                radius = int(radius)
                # FOR GAN SYTLE RUBIK'S CUBES
                # DETECT CENTER CIRCLE (instead of perceived square)
                # If the bounding circle is "larger" than the square then this is a square.
                # Else this is a circle (the smaller circle is enclosed in the square).
                if (circ_area > 1.1*rot_area):
                    cv.drawContours(frame,[box],0,(0,255,0),2)
                else:
                    cv.circle(frame,center,radius,(0,0,255),2)
                    # optionally draw rough bounding rectangle.
                    cv.rectangle(frame,(x-w,y-h),(x+2*w,y+2*h),(255,0,0),2)
                # Draw text color name.
                cv.putText(frame,colorName,(x+7,y+h//2),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)
                    # M = cv.moments(c)
                    # cx = int(M['m10']/M['m00'])
                    # cy = int(M['m01']/M['m00'])
                    # cv.putText(frame,colorName[0],(cx,cy),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)

                # Contour approximation based on the Douglas - Peucker algorithm
                # over simplification: turn a curve into a similar one with less points.
                epsilon = 0.1*cv.arcLength(c,True)
                approx = cv.approxPolyDP(c,epsilon,True)
                cv.drawContours(frame,approx,-1,(255,255,255),3)

while True:
    ret,frame = cap.read()
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV) 

    #yellow
    # or 160 replaces 80
    lyellow = np.array([19,100,80])
    uyellow = np.array([40,255,255])
    y_mask = cv.inRange(hsv_frame,lyellow,uyellow)
    yellow = cv.bitwise_and(frame,frame,mask=y_mask)

    #green
    lgreen = np.array([70,120,60])
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
    # or use 180 as hue upper bound
    # or use 180 instead of 150 VALUE
    lwhite = np.array([30,0,150])
    hwhite = np.array([150,40,255])
    w_mask = cv.inRange(hsv_frame,lwhite,hwhite)
    white = cv.bitwise_and(frame,frame,mask=w_mask)

    # Orange
    lorange = np.array([4,100,100])
    horange = np.array([26,255,255])
    o_mask = cv.inRange(hsv_frame,lorange,horange)
    orange = cv.bitwise_and(frame,frame,mask=o_mask)



    # Black
    lblack = np.array([80,0,0])
    hblack = np.array([100,55,20])
    black_mask = cv.inRange(hsv_frame,lblack,hblack)
    black = cv.bitwise_and(frame,frame,mask=black_mask)

    
    myContour("yellow",y_mask)
    # cv.imshow('yellow mask',yellow)

    myContour("green",g_mask)
    # cv.imshow('green mask',green)

    myContour("blue",b_mask)
    # cv.imshow('blue mask',blue)

    myContour("red",r_mask)
    # cv.imshow("red mask", red)

    myContour("white",w_mask)
    # cv.imshow("white mask",white)

    myContour("orange",o_mask)
    # cv.imshow("orange mask",orange)

    myContour("black",black_mask)
    # cv.imshow("black mask",black)



    cv.imshow('frame',frame)

    # HS V - hue, saturation, value (brightness)
    key = cv.waitKey(5)
    if (key == ord('q')): 
        break

cap.release()
cv.destroyAllWindows()