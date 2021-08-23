import numpy as np
import cv2 as cv

import math
from math import sin, cos, radians

from constants import *

cubes = []
# ((x,y), angle)
center2 = (0,0)
def contour_bypass(w,h,CONTOUR_BYPASS_SCALE):
    return (w > CONTOUR_BYPASS_SCALE * h 
        or h > CONTOUR_BYPASS_SCALE *w 
        or max(w,h) > MAX_CONTOUR_SQUARE_EDGE_THRESHOLD)



cap = cv.VideoCapture(0,cv.CAP_DSHOW)

def create_rotated_image():
    rotate_angle = 0
    cube_angles = sorted([x[1] for x in cubes])
    mean_angle = sum(cube_angles)
    median_angle = 0
    #get median and average.
    if (len(cubes) > 0): 
        mean_angle /= len(cubes)
        median_angle = cube_angles[len(cubes)//2]
        if (median_angle != 0 and abs(mean_angle-median_angle)/median_angle < 0.08):
            # The average angle is more accurate than the median
            print("AVERAGE USED")
            rotate_angle = mean_angle
        else:
            rotate_angle = median_angle
    
    if (rotate_angle > 45): rotate_angle -= 90
    if (abs(rotate_angle-45) < 0.5):
        print("Please rotate cube")
    else:
        height, width = frame.shape[:2]
        r_matrix = cv.getRotationMatrix2D(center=center2, angle=rotate_angle, scale=1)
        rotated_image = cv.warpAffine(src=frame, M=r_matrix, dsize=(width, height))
        cv.imshow('Rotated image', rotated_image)


def myContour(colorName, mask):
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) >= 1):
        text_color = (255,255,255)
        if (colorName=="yellow" or colorName == "white"): text_color = (0,0,0)
        for c in contours:
            if (cv.contourArea(c) > MIN_CONTOUR_THRESHOLD):
                x, y, w, h = cv.boundingRect(c)
                # Bypass false detection (short ciruit)
                if (contour_bypass(w,h,LENIENT_SQUARE_CONTOUR_BYPASS_RATIO)): 
                    continue
                # Generate rotated rectangle
                rect = cv.minAreaRect(c)
                w2, h2 = rect[1]; rot_area = w2 * h2
                box = cv.boxPoints(rect); box = np.int0(box)
                # Bypass false detection (strict)
                if (contour_bypass(w2,h2,STRICT_SQUARE_CONTOUR_BYPASS_RATIO)):
                    continue

                #angle
                angle = (rect[2])

                # Draw min circle
                (cx,cy),radius = cv.minEnclosingCircle(c)
                center = (int(cx),int(cy))
                circ_area = math.pi * radius * radius
                radius = int(radius)
                # FOR CIRCLE CENTER SYTLE RUBIK'S CUBES:
                # - Detect center circle (instead of perceived square)
                # If the bounding circle is "larger" than the square then this is a square.
                    # ([ ])
                # Else this is a circle (the smaller circle is enclosed in the square).
                    # [ O ]
                is_circle_center = (circ_area <= CIRCLE_CONTOUR_BYPASS_SCALE*rot_area)
                if (not is_circle_center):
                    cv.drawContours(frame,[box],0,(0,255,0),2)
                    pass
                else:
                    cv.circle(frame,center,radius,(0,0,255),2)
                    # optionally draw rough bounding rectangle for entire cube.
                    if (SHOW_ENTIRE_BOUNDING_RECTANGLE):
                        w*=1.3; h*=1.3
                        w = int(w); h = int(h)
                        cv.rectangle(frame,(x-w,y-h),(x+2*w,y+2*h),(255,0,0),2)
                if (not is_circle_center):
                    cubes.append(((x,y),angle))
                else:
                    global center2
                    center2 = center
                
                # Draw text color name
                if (SHOW_CONTOUR_COLOR_TEXT):
                    cv.putText(frame,colorName,(x+7,y+h//2),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)

                # Contour approximation based on the Douglas - Peucker algorithm
                # over simplification: turn a curve into a similar one with less points.
                epsilon = 0.1*cv.arcLength(c,True)
                approx = cv.approxPolyDP(c,epsilon,True)
                cv.drawContours(frame,approx,-1,(0,255,0),3)


while True:
    ret,frame = cap.read()
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV) 

    # Yellow
    lyellow = np.array([19,100,80])
    uyellow = np.array([40,255,255])
    y_mask = cv.inRange(hsv_frame,lyellow,uyellow)
    yellow = cv.bitwise_and(frame,frame,mask=y_mask)

    # Green
    lgreen = np.array([70,60,80])
    hgreen = np.array([96,255,255])
    g_mask = cv.inRange(hsv_frame,lgreen,hgreen)
    green = cv.bitwise_and(frame,frame,mask=g_mask)

    # Blue
    lblue = np.array([100,97,78])
    hblue = np.array([123,255,255])
    b_mask = cv.inRange(hsv_frame,lblue,hblue)
    blue = cv.bitwise_and(frame,frame,mask=b_mask)

    # Red
    lred = np.array([0,135,73])
    hred = np.array([10,255,255])
    r_mask = cv.inRange(hsv_frame,lred,hred)
    lred2 = np.array([170,50,73])
    hred2 = np.array([180,255,255])
    r_mask2 = cv.inRange(hsv_frame,lred2,hred2)
    r_mask |= r_mask2
    red = cv.bitwise_and(frame,frame,mask=r_mask)

    # White
    # or use 180 as hue upper bound
    # or use 180 instead of 150 VALUE
    lwhite = np.array([30,0,150])
    hwhite = np.array([150,40,255])
    w_mask = cv.inRange(hsv_frame,lwhite,hwhite)
    white = cv.bitwise_and(frame,frame,mask=w_mask)

    # Orange
    lorange = np.array([10,150,100])
    horange = np.array([20,255,255])
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
    cv.imshow('green mask',green)

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

    create_rotated_image()

    cv.imshow('frame',frame)
    
    cubes.clear()
    key = cv.waitKey(5)
    if (key == ord('q')): 
        break

cap.release()
cv.destroyAllWindows()

# https://docs.opencv.org/4.5.2/d5/d69/tutorial_py_non_local_means.html
# https://docs.opencv.org/4.5.2/db/d27/tutorial_py_table_of_contents_feature2d.html
# 