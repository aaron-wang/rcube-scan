import numpy as np
import cv2 as cv

import json
import copy

from math import pi
from constants import *


class CubeMap:
    nxt = {
        0: 5,
        1: 0,
        2: 3,
        3: 4,
        4: 1,
        5: -1
    }
    bk = {
        0: 1,
        1: 4,
        2: -1,
        3: 2,
        4: 3,
        5: 0
    }
    class Color:
        
        def __init__(self) -> None:
            pass
        
        freq = dict()

        max_freq = [[0 for i in range(3)] for j in range(3)]

        name = [['X' for i in range(3)] for j in range(3)]

        from_index = ['b', 'w', 'r', 'y', 'o', 'g']

    def __init__(self):
        self.index = 2
        self.total_readings = 0
        # cubes: ((x,y),'color')
        self.ortho = []
        # Map of cube faces.
        self.map = [[['_' for col in range(3)]
                for row in range(3)] for colors in range(6)]

    def print_subcube(self, index):
        if (index == 0 or index == 5):
            for row in range(3):
                for col in range(2 * 3):
                    if (col < 3):
                        print(' ', end=' ')
                    else:
                        print(self.map[index][row][col % 3], end=' ')
                print()
        else:
            for row in range(3):
                for col in range(4 * 3):
                    curr_dim = col//3 + 1
                    print(self.map[curr_dim][row][col % 3], end=' ')
                print()

    def print_text_cube_map(self):
        self.print_subcube(0)
        self.print_subcube(1)
        self.print_subcube(5)

    def process(self):
        '''
        Determine which colors map where on the cube
        '''
        # sort by y value
        # then chunk in groups of 3:
        # sort by x value for each respective group
        if (self.total_readings > MAX_TOTAL_READING_BUFFER):
            self.reset_mapping()

        if (len(self.ortho) < 9 or len(self.ortho) > 9):
            return False

        self.ortho.sort(key=lambda x: (x[0][1]))
        for row in range(0, 6+1, 3):
            self.ortho[row:row +
                       3] = sorted(self.ortho[row:row+3], key=lambda x: x[0][0])

        center_color = self.ortho[3*1 + 1][1]

        if (center_color[0] == self.Color.from_index[self.bk[self.index]]):
            print(f"OLD POSITION: move cube to {self.Color.from_index[self.index]}")
            return False
        if (center_color[0] != self.Color.from_index[self.index]):
            print(
                f"WRONG STARTING COLOR: start on {self.Color.from_index[self.index]}")
            return False

        for row in range(3):
            for col in range(3):
                current_color = self.ortho[3*row + col][1]
                self.Color.freq[current_color][row][col] += 1

                current_color_freq = self.Color.freq[current_color][row][col]

                if (current_color_freq > self.Color.max_freq[row][col]):
                    self.Color.max_freq[row][col] = current_color_freq
                    self.Color.name[row][col] = current_color

        self.total_readings += 1
        print(self.total_readings)
        if(self.mapping_frequency_success()):
            for row in range(3):
                for col in range(3):
                    self.map[self.index][row][col] = self.Color.name[row][col][0]
            self.print_text_cube_map()
            self.reset_mapping()

            self.index = self.nxt[self.index]

            if (self.index == -1):
                exit()

    def mapping_frequency_success(self):
            if (self.total_readings < MIN_TOTAL_READING_BUFFER):
                return False
            for i in range(3):
                for j in range(3):
                    if (self.Color.max_freq[i][j]/self.total_readings < MIN_COLOR_CONFIDENCE_THRESHOLD):
                        return False
                    else:
                        pass
            return True

    def reset_mapping(self):
        self.total_readings = 0

        for c in COLORS:
            self.Color.freq[c] = [[0 for i in range(3)] for j in range(3)]

        self.Color.max_freq = [[0 for i in range(3)] for j in range(3)]
        self.Color.name = [['X' for i in range(3)] for j in range(3)]


# cube_angles = (angle)
cube_angles = []


center_cube = (0, 0)


#   B
# W R Y O
#   G

#   0
# 1 2 3 4
#   5

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

for c in COLORS:
    CubeMap.Color.freq[c] = [[0 for i in range(3)] for j in range(3)]


def contour_bypass(w, h, contour_bypass_ratio) -> bool:
    '''
    Checks if contour shape is square-like
    '''
    return (w > contour_bypass_ratio * h
            or h > contour_bypass_ratio * w
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
                    cube_map.ortho.append(((x, y), color))
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


cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cube_map = CubeMap()

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
        pass

    cv.imshow('frame', frame)
    # MAP THE CUBE.
    cube_map.process()

    cube_angles.clear()
    cube_map.ortho.clear()

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
