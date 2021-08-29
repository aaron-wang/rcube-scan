import numpy as np
import cv2 as cv

import json
import copy
import kociemba

from math import pi
from constants import *

PAUSE_AT_END = False


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

        to_index = {'b': 0, 'w': 1, 'r': 2, 'y': 3, 'o': 4, 'g': 5,'_':99}

        to_bgr_color = {
            'b': (217,56,56),
            'w': (255, 255, 255),
            'r': (0, 0, 255),
            'y': (0, 255, 255),
            'o': (0, 128, 255),
            # 'g': (50, 191, 50),
            'g': (74,200,32),
            '_': (128, 128, 128)
        }
        to_bgr_light_color = {
            'b': (255, 128, 128),
            'w': (200, 200, 200),
            'r': (128, 128, 255),
            'y': (180, 255, 255),
            'o': (130, 211, 255),
            'g': (128, 255, 128),
            '_': (128, 128, 128)
        }

    def __init__(self):
        # which cube color
        self.index = 2
        # count of frames read.
        self.total_readings = 0
        # ortho: ((x,y),'fullcolor_name')
        self.ortho = []
        # Map of cube faces.
        self.map = [[['_' for col in range(3)]
                     for row in range(3)] for colors in range(6)]
        # U R F D L B
        # 0 3 2 5 1 4
        self.flat_map = {
            0: 'U',
            3: 'R',
            2: 'F',
            5: 'D',
            1: 'L',
            4: 'B',
            99: 'X'
        }

    def print_subcube(self, index):
        '''
        Print text. Prints as follows.
            B
        W   R   Y   O
            G

            0
        1   2   3   4
            5
        Empty parts of 3 x 4 grid are skipped.
        '''
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
        '''
        Prints text. Used for debugging.
        Print three rows
        Top and bottom print once.
        Middle prints 4 times.
        '''
        self.print_subcube(0)
        self.print_subcube(1)
        self.print_subcube(5)

    def process(self):
        '''
        Determine which colors map where on the cube.
        Called on every frame capture read.
        '''
        global PAUSE_AT_END
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
            # print(
            # f"OLD POSITION: move cube to {self.Color.from_index[self.index]}")
            return False
        if (center_color[0] != self.Color.from_index[self.index]):
            # print(
            # f"WRONG STARTING COLOR: start on {self.Color.from_index[self.index]}")
            return False

        for row in range(3):
            for col in range(3):
                current_color = self.ortho[3*row + col][1]
                self.Color.freq[current_color][row][col] += 1

                current_color_freq = self.Color.freq[current_color][row][col]

                if (current_color_freq > self.Color.max_freq[row][col]):
                    self.Color.max_freq[row][col] = current_color_freq
                    self.Color.name[row][col] = current_color
                self.map[self.index][row][col] = self.Color.name[row][col][0]

        self.total_readings += 1
        if (SHOW_DEBUG_CONSOLE_TXT):
            print(self.total_readings)
        
        if(self.mapping_frequency_success()):
            for row in range(3):
                for col in range(3):
                    self.map[self.index][row][col] = self.Color.name[row][col][0]
            if (SHOW_DEBUG_CONSOLE_TXT):
                self.print_text_cube_map()

            self.reset_mapping()
            self.index = self.nxt[self.index]

            if (self.index == -1):
                PAUSE_AT_END = True
            self.map[self.index][1][1] = self.Color.from_index[self.index]

    def mapping_frequency_success(self):
        '''
        Check if readings match frequency requirements
        '''
        if (self.total_readings < MIN_TOTAL_READING_BUFFER):
            return False
        for i in range(3):
            for j in range(3):
                if (self.Color.max_freq[i][j]/self.total_readings < MIN_COLOR_CONFIDENCE_THRESHOLD or
                        self.Color.max_freq[i][j] < MIN_COLOR_COUNT_THRESHOLD):
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

    def draw_stickers(self, frame, index, x, y,preview=False):
        bottom_right = (x + 3 * (STICKER_SUPER_LENGTH) + 2,
                        y + 3 * (STICKER_SUPER_LENGTH) + 2)

        cv.rectangle(frame, (x-STICKER_GAP, y-STICKER_GAP),
                     bottom_right, (0, 0, 0), -1)

        for i in range(3):
            for j in range(3):
                cx = x + j * (STICKER_SUPER_LENGTH) + j
                cy = y + i * (STICKER_SUPER_LENGTH) + i
                if (not preview):
                    color = self.Color.to_bgr_color[self.map[index][i][j]]
                else:
                    if (index == self.index):
                        color = self.Color.to_bgr_light_color[self.map[index][i][j]]
                    else:
                        color = self.Color.to_bgr_color[self.map[index][i][j]]
                cv.rectangle(frame, (cx, cy), (cx+STICKER_LENGTH, cy+STICKER_LENGTH),
                             color, -1)

    def draw_supercube(self, frame, preview=False):
        '''
        Draws as follows.
        - Top and bottom drawn once.
        - Middle drawn 4 times.
            B
        W   R   Y   O
            G

            0
        1   2   3   4
            5
        Empty parts of 3 x 4 grid are skipped.
        '''
        (cx, cy) = (50, 50)
        self.draw_stickers(frame, 0, cx+CUBE_LENGTH, cy,preview)
        for i in range(4):
            self.draw_stickers(frame, i+1, cx + i *
                               CUBE_LENGTH, cy+CUBE_LENGTH,preview)
        self.draw_stickers(frame, 5, cx+CUBE_LENGTH, cy+2*CUBE_LENGTH,preview)

    def color_to_notation(self, color):
        '''
        Convert the color string into its index.
        Convert the index into standard notation.
        '''
        return self.flat_map[self.Color.to_index[color]]

    def flatten_cube(self):
        '''
        Flatten the cube map into a single string.

        Go through each face and convert colors to standard notation.
        '''
        ret = ""
        for face in self.flat_map.keys():
            if (face == 99):
                continue
            for i in range(3):
                for j in range(3):
                    ret += self.color_to_notation(self.map[face][i][j])
        # ret = ''.join([self.color_to_notation(self.map[face][i][j]) if face != 99 else '' for j in range(3)
        # for i in range(3) for face in self.flat_map.keys()])
        # print(ret)
        return ret

        
class Camera:
    def __init__(self) -> None:
        self.cube_angles = []
        self.center_cube = (0, 0)
        self.rotated_frame = None
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.cube_map = CubeMap()

    def contour_bypass(self, w, h, contour_bypass_ratio) -> bool:
        '''
        Checks if contour shape is square-like
        '''
        return (w > contour_bypass_ratio * h
                or h > contour_bypass_ratio * w
                or max(w, h) > MAX_CONTOUR_SQUARE_EDGE_THRESHOLD)

    def create_rotated_frame(self, frame):
        '''
        Make detected cube orthogonal (rotates frame)
        '''
        rotate_angle = 0
        self.cube_angles.sort
        mean_angle = sum(self.cube_angles)
        median_angle = 0
        # get median and average.
        if (len(self.cube_angles) == 8):
            mean_angle /= len(self.cube_angles)
            median_angle = self.cube_angles[len(self.cube_angles) // 2]
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
                center=self.center_cube, angle=rotate_angle, scale=1)
            self.rotated_frame = cv.warpAffine(
                src=frame, M=r_matrix, dsize=(width, height))
            return True

    def create_contour_preview(self, frame, window_name, mask):
        '''
        Create weak contour preview based on mask
        '''
        preview_frame = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow(window_name, preview_frame)

    def draw_cube_contour(self, frame, color, mask, is_rotated=False):
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
                    if (self.contour_bypass(w, h, LENIENT_SQUARE_CONTOUR_BYPASS_RATIO)):
                        continue
                    # Generate rotated rectangle
                    rect = cv.minAreaRect(c)
                    w2, h2 = rect[1]
                    rot_area = w2 * h2
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    # Bypass false detection (strict)
                    if (self.contour_bypass(w2, h2, STRICT_SQUARE_CONTOUR_BYPASS_RATIO)):
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
                        self.cube_map.ortho.append(((x, y), color))
                    else:
                        if (not is_circle_center):
                            self.cube_angles.append(angle)
                        else:
                            self.center_cube = center

                    # Draw text color name
                    if (SHOW_CONTOUR_COLOR_TEXT):
                        cv.putText(frame, color, (x + 7, y + h // 2),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                    # Contour approximation based on the Douglas - Peucker algorithm
                    # over simplification: turn a curve into a similar one with less points.
                    epsilon = 0.1 * cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, epsilon, True)
                    cv.drawContours(frame, approx, -1, (0, 255, 0), 3)

    def run_main_process(self):
        while True:
            ret, frame = self.cap.read()

            original_frame = copy.copy(frame)

            # Main color recognition
            # Convert from BGR to HSV colorspace.
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Generate masks.
            masks = dict(
                (c, cv.inRange(hsv_frame, HSV_BOUND[c][0], HSV_BOUND[c][1])) for c in COLORS)
            masks['red'] |= cv.inRange(hsv_frame, *HSV_RED_UNION_BOUND)

            for c in COLORS:
                self.draw_cube_contour(frame, c, masks[c])
                if SHOW_CONTOUR_PREVIEW:
                    self.create_contour_preview(original_frame, c, masks[c])

            # Handle rotated recognition.
            if (self.create_rotated_frame(original_frame)):
                # convert
                hsv_rotated_frame = cv.cvtColor(
                    self.rotated_frame, cv.COLOR_BGR2HSV)

                # generate masks.
                rotated_masks = dict((c, cv.inRange(
                    hsv_rotated_frame, HSV_BOUND[c][0], HSV_BOUND[c][1])) for c in COLORS)
                rotated_masks['red'] |= cv.inRange(
                    hsv_rotated_frame, *HSV_RED_UNION_BOUND)

                if (SHOW_CONTOUR_PREVIEW):
                    original_rotated_frame = copy.copy(self.rotated_frame)

                for c in COLORS:
                    self.draw_cube_contour(
                        self.rotated_frame, c, rotated_masks[c], True)
                    if (SHOW_CONTOUR_PREVIEW):
                        if (SHOW_RAW_CONTOUR_PREVIEW):
                            self.create_contour_preview(
                                original_rotated_frame, c, rotated_masks[c])
                        else:
                            self.create_contour_preview(
                                self.rotated_frame, c, rotated_masks[c])
                if (SHOW_ROTATED_FRAME_PREVIEW):
                    cv.imshow('Rotated image', self.rotated_frame)
            else:
                pass

            self.cube_map.process()
            self.cube_map.draw_supercube(frame,True)

            cv.imshow('frame', frame)
            # self.cube_map.flatten_cube()
            # allow cam to keep reading, but not process any further.
            self.cube_map.flatten_cube()
            global PAUSE_AT_END
            if (PAUSE_AT_END):
                try:
                    print(kociemba.solve(self.cube_map.flatten_cube()))
                except ValueError:
                    print("RETRY")
                print("PAUSED")
                while (1):
                    ret, frame = self.cap.read()

                    self.cube_map.draw_supercube(frame,True)

                    cv.imshow('frame', frame)
                    key = cv.waitKey(1)
                    if (key == ord('q')):
                        self.cap.release()
                        cv.destroyAllWindows()
                        PAUSE_AT_END = False
                        return
                    if (key == ord('x')):
                        exit()
                        # exit()

            self.cube_angles.clear()
            self.cube_map.ortho.clear()

            key = cv.waitKey(1)
            if (key == ord('q')):
                break
            if (key == ord('x')):
                exit()
            if (key == ord('r')):
                self.cube_map.index = self.cube_map.bk[self.cube_map.index]
                if (self.cube_map.index == -1):
                    self.cube_map.index = 2
        # End process
        self.cap.release()
        cv.destroyAllWindows()


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

# while True:
cam = Camera()

cam.run_main_process()
# cam.__init__()
# cam = Camera()

# cam.run_main_process()
