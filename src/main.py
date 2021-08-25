import numpy as np
import cv2 as cv

import json
import copy

from math import pi
from constants import *

# cube_angles = (angle)
cube_angles = []

# cubes: ((x,y),'color')
ortho_cubes = []

total_readings = 0

center_cube = (0, 0)

cube_map = [[['_' for col in range(3)] for row in range(3)] for colors in range(6)]

current_cube_map_index = 2
#   B
# W R Y O
#   G

#   0
# 1 2 3 4
#   5


# R R R
# R R R
# R R R

def print_cube(index):
    if (index == 0 or index == 5):
        #print empty cube
        for row in range(3):
            for col in range(2 * 3):
                if (col < 3):
                    print(' ',end=' ')
                else:
                    print(cube_map[index][row][col%3],end=' ')
            print()
    else:
        for row in range(3):
            for col in range(4 * 3):
                curr_dim = col//3 + 1
                # print(curr_dim)
                print(cube_map[curr_dim][row][col%3],end=' ')
                # 0 1 2 
                # 3 4 5
            print()

def print_text_cube_map():
    print_cube(0)
    print_cube(1)
    print_cube(5)
    pass
# exit()
with open("config.json") as f:
    data = json.load(f)


HAS_CIRCLE_CENTER = json.loads(data['hasCircleCenter'])

COLORS = ['yellow', 'green', 'blue', 'red', 'white', 'orange']

HSV_BOUND = dict()

for c in COLORS:
    HSV_BOUND[c] = (np.array(data['colors'][c]['lower_bound']),
                    np.array(data['colors'][c]['upper_bound']))

HSV_RED_UNION_BOUND = (
    np.array(data['colors']['red']['lower_bound2']),
    np.array(data['colors']['red']['upper_bound2']))

masks = list()
rotated_masks = list()

freq_cube = dict()
for c in COLORS:
    freq_cube[c] = [[0 for i in range(3)] for j in range(3)]

max_cube = [[0 for i in range(3)] for j in range(3)]
max_color = [['X' for i in range(3)] for j in range(3)]


def contour_bypass(w, h, CONTOUR_BYPASS_SCALE) -> bool:
    '''
    Checks if contour shape is square-like
    '''
    return (w > CONTOUR_BYPASS_SCALE * h
            or h > CONTOUR_BYPASS_SCALE * w
            or max(w, h) > MAX_CONTOUR_SQUARE_EDGE_THRESHOLD)


def create_rotated_frame(frame):
    '''
    Make detected cube orthogonal (rotates frame)
    '''
    global rotated_frame
    rotate_angle = 0
    cube_angles.sort
    mean_angle = sum(cube_angles)
    median_angle = 0
    # get median and average.
    if (len(cube_angles) == 8):
        mean_angle /= len(cube_angles)
        median_angle = cube_angles[len(cube_angles) // 2]
        if (median_angle != 0 and abs(mean_angle - median_angle) / median_angle < 0.08):
            # The average angle is more accurate than the median
            rotate_angle = mean_angle
        else:
            rotate_angle = median_angle
    else:
        return False
    # Prevent inaccuracies when cube angle is uncertain
    # (two orientations with near likely probability)
    if (abs(rotate_angle - 45) < ANGLE_EPSILON):
        print("Please rotate cube")
        return False
    else:
        if (rotate_angle > 45):
            rotate_angle -= 90
        height, width = frame.shape[:2]
        r_matrix = cv.getRotationMatrix2D(
            center=center_cube, angle=rotate_angle, scale=1)
        rotated_frame = cv.warpAffine(
            src=frame, M=r_matrix, dsize=(width, height))
        return True


def create_contour_preview(frame, window_name, mask):
    '''
    Create weak contour preview based on mask
    '''
    preview_frame = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow(window_name, preview_frame)


def draw_cube_contour(frame, color, mask, is_rotated=False):
    '''
    Draw contours with rotated rectangles and circles.

    Displays detected colors in text (if `SHOW_CONTOUR_COLOR_TEXT`)

    ### Circle Detection

    Detect center circle (instead of perceived square)

    - If the bounding circle is "larger" than the square
        - then this is a square. ([ ])

    - Else this is a circle
        - (the smaller circle is enclosed in the square). [ 0 ]

    *for circle center sytle rubik's cubes*
    '''
    contours, hierarchy = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) >= 1):
        text_color = (255, 255, 255)
        if (color == "yellow" or color == "white"):
            text_color = (0, 0, 0)
        for c in contours:
            if (cv.contourArea(c) > MIN_CONTOUR_THRESHOLD):
                x, y, w, h = cv.boundingRect(c)
                # Bypass false detection (short ciruit)
                if (contour_bypass(w, h, LENIENT_SQUARE_CONTOUR_BYPASS_RATIO)):
                    continue
                # Generate rotated rectangle
                rect = cv.minAreaRect(c)
                w2, h2 = rect[1]
                rot_area = w2 * h2
                box = cv.boxPoints(rect)
                box = np.int0(box)
                # Bypass false detection (strict)
                if (contour_bypass(w2, h2, STRICT_SQUARE_CONTOUR_BYPASS_RATIO)):
                    continue
                # angle
                angle = (rect[2])
                # Draw min circle
                (cx, cy), radius = cv.minEnclosingCircle(c)
                center = (int(cx), int(cy))
                circ_area = pi * radius * radius
                radius = int(radius)

                is_circle_center = (
                    circ_area <= CIRCLE_CONTOUR_BYPASS_SCALE * rot_area)
                if (not is_circle_center or is_rotated):
                    cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
                else:
                    cv.circle(frame, center, radius, (0, 0, 255), 2)
                    # optionally draw rough bounding rectangle for entire cube.
                    if (SHOW_ENTIRE_BOUNDING_RECTANGLE):
                        w *= 1.3
                        h *= 1.3
                        w = int(w)
                        h = int(h)
                        cv.rectangle(frame, (x - w, y - h),
                                     (x + 2 * w, y + 2 * h), (255, 0, 0), 2)
                if is_rotated:
                    ortho_cubes.append(((x, y), color))
                else:
                    if (not is_circle_center):
                        cube_angles.append(angle)
                    else:
                        global center_cube
                        center_cube = center

                # Draw text color name
                if (SHOW_CONTOUR_COLOR_TEXT):
                    cv.putText(frame, color, (x + 7, y + h // 2),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                # Contour approximation based on the Douglas - Peucker algorithm
                # over simplification: turn a curve into a similar one with less points.
                epsilon = 0.1 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)
                cv.drawContours(frame, approx, -1, (0, 255, 0), 3)

def reset_mapping():
    global max_cube, max_color, freq_cube
    global total_readings

    total_readings = 0

    for c in COLORS:
        freq_cube[c] = [[0 for i in range(3)] for j in range(3)]

    max_cube = [[0 for i in range(3)] for j in range(3)]
    max_color = [['X' for i in range(3)] for j in range(3)]

def color_mapping_frequency_success():
    if (total_readings < MIN_TOTAL_READING): 
        return False
    for i in range(3):
        for j in range(3):
            if (max_cube[i][j]/total_readings < MIN_COLOR_CONFIDENCE_THRESHOLD):
                return False
            else:
                pass
    return True

def process_cube():
    '''
    Determine which colors map where on the cube
    '''
    global current_cube_map_index
    # sort by y value
    # then chunk in groups of 3:
    # sort by x value for each respective group
    global total_readings
    if (total_readings > MAX_TOTAL_READING):
        reset_mapping()
    if (len(ortho_cubes) < 9 or len(ortho_cubes) > 9):
        return False
    ortho_cubes.sort(key=lambda x: (x[0][1]))
    for i in range(0, 6+1, 3):
        ortho_cubes[i:i+3] = sorted(ortho_cubes[i:i+3], key=lambda x: x[0][0])

    total_readings += 1
    
    # print(ortho_cubes)
    # if (len(ortho_cubes) == 9):
    for i in range(3):
        for j in range(3):
            color = ortho_cubes[3*i + j][1]
            freq_cube[color][i][j] += 1

            if (freq_cube[color][i][j] > max_cube[i][j]):
                max_cube[i][j] = freq_cube[color][i][j]
                max_color[i][j] = color

            # print(f"{max_color[i][j]} \t ({max_cube[i][j]})", end=' ')
            # print(f"{max_color[i][j]} \t ({max_cube[i][j]/total_readings:.2f})", end=' ')

        # print()
    # print("TOT: ",total_readings)
    # print("-------")
    
    if(color_mapping_frequency_success()):
        for i in range(3):
            for j in range(3):
                cube_map[current_cube_map_index][i][j] = max_color[i][j][0]
        print_text_cube_map()
        reset_mapping()
        print(current_cube_map_index)
        if (current_cube_map_index == 4): 
            current_cube_map_index = 1
        elif (current_cube_map_index == 1):
            current_cube_map_index = 0
        elif (current_cube_map_index == 0):
            current_cube_map_index = 5
        elif (current_cube_map_index == 5):
            exit()
        else:
            current_cube_map_index+=1
        

        # exit()



cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    original_frame = copy.copy(frame)

    # Main color recognition
    # Convert from BGR to HSV colorspace.
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Generate masks.
    masks = dict(
        (c, cv.inRange(hsv_frame, HSV_BOUND[c][0], HSV_BOUND[c][1])) for c in COLORS)
    masks['red'] |= cv.inRange(hsv_frame, *HSV_RED_UNION_BOUND)

    for c in COLORS:
        draw_cube_contour(frame, c, masks[c])
        if SHOW_CONTOUR_PREVIEW:
            create_contour_preview(original_frame, c, masks[c])

    # Handle rotated recognition.
    if (create_rotated_frame(original_frame)):
        # convert
        hsv_rotated_frame = cv.cvtColor(rotated_frame, cv.COLOR_BGR2HSV)

        # generate masks.
        rotated_masks = dict((c, cv.inRange(
            hsv_rotated_frame, HSV_BOUND[c][0], HSV_BOUND[c][1])) for c in COLORS)
        rotated_masks['red'] |= cv.inRange(
            hsv_rotated_frame, *HSV_RED_UNION_BOUND)

        if (SHOW_CONTOUR_PREVIEW):
            original_rotated_frame = copy.copy(rotated_frame)

        for c in COLORS:
            draw_cube_contour(rotated_frame, c, rotated_masks[c], True)
            if (SHOW_CONTOUR_PREVIEW):
                # create_contour_preview(
                # original_rotated_frame, c, rotated_masks[c])
                create_contour_preview(
                    rotated_frame, c, rotated_masks[c])
        cv.imshow('Rotated image', rotated_frame)
    else:
        print("paused")
        # for c in COLORS:
        # freq_cube[c] = [[0 for i in range(3)] for j in range(3)]

    cv.imshow('frame', frame)
    # MAP THE CUBE.
    process_cube()

    cube_angles.clear()
    ortho_cubes.clear()

    key = cv.waitKey(1)
    if (key == ord('q')):
        break
# End process
cap.release()
cv.destroyAllWindows()

# https://docs.opencv.org/4.5.2/d5/d69/tutorial_py_non_local_means.html
# https://docs.opencv.org/4.5.2/db/d27/tutorial_py_table_of_contents_feature2d.html

# https://scikit-image.org/docs/stable/
# https://www.pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

