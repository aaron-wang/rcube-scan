import numpy as np
import cv2 as cv

# import math
from math import sin, cos, radians, pi

from constants import *

import json, time, copy

# cubes: ((x,y), angle)
cubes = []
widths = []
center2 = (0,0)

rotate_success = True

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

def contour_bypass(w,h,CONTOUR_BYPASS_SCALE):
    return (w > CONTOUR_BYPASS_SCALE * h 
        or h > CONTOUR_BYPASS_SCALE *w 
        or max(w,h) > MAX_CONTOUR_SQUARE_EDGE_THRESHOLD)

def simple_rotated_contour(color,mask):
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) >= 1):
        text_color = (0,0,255)
        for c in contours:
            if (cv.contourArea(c) > MIN_CONTOUR_THRESHOLD):
                x, y, w, h = cv.boundingRect(c)
                if (contour_bypass(w,h,LENIENT_SQUARE_CONTOUR_BYPASS_RATIO)): 
                    continue
                rect = cv.minAreaRect(c)
                w2, h2 = rect[1]; 
                box = cv.boxPoints(rect); box = np.int0(box)
                # Bypass false detection (strict)
                if (contour_bypass(w2,h2,STRICT_SQUARE_CONTOUR_BYPASS_RATIO)):
                    continue
                #angle
                cv.rectangle(rotated_frame,(x,y),(x+w,y+h),(0,0,255),2)
                # Draw text color name
                if (SHOW_CONTOUR_COLOR_TEXT):
                    cv.putText(rotated_frame,color,(x+7,y+h//2),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)

                # Contour approximation based on the Douglas - Peucker algorithm
                # over simplification: turn a curve into a similar one with less points.
                epsilon = 0.1*cv.arcLength(c,True)
                approx = cv.approxPolyDP(c,epsilon,True)
                cv.drawContours(rotated_frame,approx,-1,(0,255,0),3)
def create_rotated_frame(frame):
    global rotated_frame, rotate_success
    widths.sort()
    rotate_angle = 0
    cube_angles = sorted([c[1] for c in cubes])
    mean_angle = sum(cube_angles)
    median_angle = 0
    #get median and average.
    if (len(cubes) > 0): 
        mean_angle /= len(cubes)
        median_angle = cube_angles[len(cubes)//2]
        if (median_angle != 0 and abs(mean_angle-median_angle)/median_angle < 0.08):
            # The average angle is more accurate than the median
            rotate_angle = mean_angle
        else:
            rotate_angle = median_angle
    
    if (rotate_angle > 45): rotate_angle -= 90
    if (abs(rotate_angle-45) < 1.0):
        print("Please rotate cube")
        rotate_success = False
    else:
        height, width = frame.shape[:2]
        r_matrix = cv.getRotationMatrix2D(center=center2, angle=rotate_angle, scale=1)
        # if (len(widths) >= 1):
        rotated_frame = cv.warpAffine(src=frame, M=r_matrix, dsize=(width, height))

def create_contour_preview(frame,window_name,mask):
    preview_frame = cv.bitwise_and(frame,frame,mask=mask)
    cv.imshow(window_name,preview_frame)

def draw_cube_contour(frame,color, mask):
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) >= 1):
        text_color = (255,255,255)
        if (color=="yellow" or color == "white"): text_color = (0,0,0)
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
                circ_area = pi * radius * radius
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
                    widths.append((w2+h2)/2)
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
                    cv.putText(frame,color,(x+7,y+h//2),cv.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)
                
                # Contour approximation based on the Douglas - Peucker algorithm
                # over simplification: turn a curve into a similar one with less points.
                epsilon = 0.1*cv.arcLength(c,True)
                approx = cv.approxPolyDP(c,epsilon,True)
                cv.drawContours(frame,approx,-1,(0,255,0),3)
                

with open("config.json") as f:
    data = json.load(f)

colors = ['yellow', 'green', 'blue', 'red', 'white', 'orange']


HSV_BOUND = dict()
for c in colors:
    HSV_BOUND[c] = (np.array(data['colors'][c]['lower_bound']),np.array(data['colors'][c]['upper_bound']))

RED_UNION_BOUND = (np.array(data['colors']['red']['lower_bound2']),np.array(data['colors']['red']['upper_bound2']))

masks = list()
rotated_masks = list()

while True:
    ret,frame = cap.read()

    ### Main color recognition

    # Convert from BGR to HSV colorspace.
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV) 
    
    # Generate masks.
    masks = dict((c,cv.inRange(hsv_frame,HSV_BOUND[c][0],HSV_BOUND[c][1])) for c in colors)
    masks['red'] |= cv.inRange(hsv_frame,*RED_UNION_BOUND)
    
    # if (SHOW_CONTOUR_PREVIEW):
    original_frame = copy.copy(frame)

    for c in colors:
        draw_cube_contour(frame,c,masks[c])
        if SHOW_CONTOUR_PREVIEW:
            create_contour_preview(original_frame,c,masks[c])
            pass

    ### Handle rotated recognition.

    create_rotated_frame(original_frame)
    if (rotate_success):
        # convert 
        hsv_rotated_frame = cv.cvtColor(rotated_frame,cv.COLOR_BGR2HSV)
        # generate masks.
        rotated_masks = dict((c,cv.inRange(hsv_rotated_frame,HSV_BOUND[c][0],HSV_BOUND[c][1])) for c in colors)
        rotated_masks['red'] |= cv.inRange(hsv_rotated_frame,*RED_UNION_BOUND)
        
        if (SHOW_CONTOUR_PREVIEW):
            original_rotated_frame = copy.copy(rotated_frame)

        for c in colors:
            draw_cube_contour(rotated_frame,c,rotated_masks[c])
            if (SHOW_CONTOUR_PREVIEW):
                create_contour_preview(original_rotated_frame,c,rotated_masks[c])
                pass
        cv.imshow('Rotated image', rotated_frame)
    rotate_success = True

    cv.imshow('frame',frame)
    
    cubes.clear()
    widths.clear()

    key = cv.waitKey(5)
    if (key == ord('q')): 
        break
# End process
cap.release()
cv.destroyAllWindows()

# https://docs.opencv.org/4.5.2/d5/d69/tutorial_py_non_local_means.html
# https://docs.opencv.org/4.5.2/db/d27/tutorial_py_table_of_contents_feature2d.html