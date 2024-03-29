import numpy as np
import cv2 as cv

import json
import copy
import kociemba

from math import pi, dist
from constants import *

import time

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

        # Assign each color a number from 0 to 5 inclusive.
        to_index = {'b': 0, 'w': 1, 'r': 2, 'y': 3, 'o': 4, 'g': 5,'_':99}

        to_bgr_color = {
            'b': (217,56,56),
            'w': (255, 255, 255),
            'r': (0, 0, 255),
            'y': (0, 255, 255),
            'o': (0, 128, 255),
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
        # Which cube color we are on.
        self.index = 2
        # Count of frames read.
        self.total_readings = 0
        # The list ortho has type: ((x,y),'fullcolor_name')
        self.ortho = []
        # The list raw_points has type : (x,y)
        self.raw_points = []
        # Map of cube faces.
        self.map = [[['_' for col in range(3)]
                     for row in range(3)] for colors in range(6)]
        # Transform the color into standard notation. 'X' is an error value.
        # Values map as follows:
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
        # Sort by y value, then chunk into groups of 3.
        # Next, sort by x value for each respective group
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
            return False
        if (center_color[0] != self.Color.from_index[self.index]):
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
        '''
        Initialize mapping as empty.
        '''
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

    def draw_supercube(self, frame, pos, preview=False):
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
        if (LOW_RES_CAMERA):
            (cx, cy) = (330, 50)
        else:
            (cx,cy) = pos
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
        return ret

    def reverse_notation(self,org_notation):
        raw_notations_reversed = list(reversed(org_notation.split(' ')))    
        inverse_notation_list = list()
        for move in raw_notations_reversed:
            if (len(move) == 1):
                inverse_notation_list.append(move + "'")
            elif (len(move) == 2):
                if (move[1] == "2"):
                    inverse_notation_list.append(move)
                else:
                    inverse_notation_list.append(move[0])
            else:
                raise(Exception)
        return ' '.join(inverse_notation_list)      

    def special_draw_text(self,frame, text,cx,thickness,scale,length_buffer):
        '''
        Put text of solution. Auto break line if exceeding box length.
        Slightly tab second line.

        Splits solution string into list, and processes each move unit.
        '''
        tmp = text.split(' ')
        raw_partition = ""
        current_length = 0
        y = cx[1]
        for x in tmp:
            if (current_length >= length_buffer):
                cv.putText(frame, raw_partition , (cx[0] + (13 if y != cx[1]else 0),y),
                    cv.FONT_HERSHEY_SIMPLEX, thickness, (0,0,0), scale)
                raw_partition = ""
                current_length = 0
                y += int(40 * thickness)
            raw_partition +=  " " + x
            current_length+= len(x) + 1
        if (current_length != 0):
            cv.putText(frame, raw_partition , (cx[0] + (13 if y != cx[1]else 0),y),
                cv.FONT_HERSHEY_SIMPLEX, thickness, (0,0,0), scale)
            raw_partition = ""

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
    
    def find_center_cube(self, frame):
        '''
        Return the coordinates of the center cube. (If not circle center).

        Method:
        - Take the average of all points (X)
        - Find the point (A) which is closest to such average (X)
        - "Close" has a strict error bound. The average (X) should be near exact to the point (A)


        Return false if invalid.
        '''
        avg_point_x = sum(x for x, _ in self.cube_map.raw_points) / 9
        avg_point_y = sum(y for _, y in self.cube_map.raw_points) /9
        avg_point = (avg_point_x, avg_point_y)

        (best_point, best_delta) = (self.cube_map.raw_points[0], dist(avg_point,self.cube_map.raw_points[0]))
        (far_point, far_delta) = (best_point, best_delta)

        for pt in self.cube_map.raw_points:
            curr_dist = dist(avg_point,pt)
            if (curr_dist < best_delta):
                best_delta = curr_dist
                best_point = pt
            if (curr_dist > far_delta):
                far_delta = curr_dist
                far_point = pt

        if (SHOW_DEBUG_CONSOLE_TXT):
            pass
        # far_delta is diagonal length.
        # 1.0/3.0 considers the diagonal of one cubie/square on the cube.
        if (best_delta < far_delta * 1.0/3.0 * CENTER_EPSILON):
            return best_point
        else:
            return False   

    def create_rotated_frame(self, frame):
        '''
        Make detected cube orthogonal (rotates frame). Returns true if successful.
        '''
        rotate_angle = 0
        self.cube_angles.sort
        mean_angle = sum(self.cube_angles)
        median_angle = 0

        
        if (HAS_CIRCLE_CENTER):
            # Get the median and average of all the angles.
            if (len(self.cube_angles) == 8):
                mean_angle /= len(self.cube_angles)
                median_angle = self.cube_angles[len(self.cube_angles) // 2]
                if (median_angle != 0 and abs(mean_angle - median_angle) / median_angle < 0.08):
                    # The average angle is more accurate than the median in this case.
                    rotate_angle = mean_angle
                else:
                    rotate_angle = median_angle
            else:
                return False
        else:
            # Proceed exactly as in above.
            if (len(self.cube_angles) == 9):
                mean_angle /= len(self.cube_angles)
                median_angle = self.cube_angles[len(self.cube_angles) // 2]
                if (median_angle != 0 and abs(mean_angle - median_angle) / median_angle < 0.08):
                    # The average angle is more accurate than the median in this case.
                    rotate_angle = mean_angle
                else:
                    rotate_angle = median_angle
                # Find the center first.
                flag = self.find_center_cube(frame)

                if (type(flag) is bool):
                    return False
                else:
                    self.center_cube = flag
            else:
                return False
        # Prevent inaccuracies when the cube angle is uncertain
        # (Informally, this means that if the angle is very close to 45, then return false)
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

                    angle = (rect[2])
                    
                    # Draw minimum enclosing circle
                    (cx, cy), radius = cv.minEnclosingCircle(c)
                    center = (int(cx), int(cy))
                    circ_area = pi * radius * radius
                    radius = int(radius)

                    is_circle_center = (
                        circ_area <= CIRCLE_CONTOUR_BYPASS_SCALE * rot_area)
                    if (not is_circle_center or is_rotated):
                        pass
                    else:
                        if (HAS_CIRCLE_CENTER):
                            cv.circle(frame, center, radius, (0, 0, 255), 2)
                        # Optionally draw rough bounding rectangle for entire cube.
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
                        if (not HAS_CIRCLE_CENTER):
                            self.cube_map.raw_points.append((x,y))

                        if (not is_circle_center):
                            self.cube_angles.append(angle)
                        else:
                            # All cubies are square, so include this in the calculation.
                            if (not HAS_CIRCLE_CENTER):
                                self.cube_angles.append(angle)
                            else:
                                self.center_cube = center

                    # Display text on the cube stating the current identified color 
                    if (SHOW_CONTOUR_COLOR_TEXT):
                        cv.putText(frame, color, (x + 7, y + h // 2),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                    # Contour approximation based on the Douglas - Peucker algorithm.
                    # Informally, the over-simplification is: Turn a curve into a similar one with less points.
                    epsilon = 0.1 * cv.arcLength(c, True)
                    approx = cv.approxPolyDP(c, epsilon, True)
                    cv.drawContours(frame, approx, -1, (0, 255, 0), 5)

    def run_main_process(self):
        '''
        Highest level function. All functions called must be indirectly rooted here.
        #### Steps:
        1.Capture frame. 
            - Convert BGR to HSV.
        2.Generate masks. 
            - Draw contours for each colour. 
            - Show contour preview (optional).
        3.Handle rotated recognition
            - Create rotated mask.
            - Analyze the values of `cube_angles` (processed during 2. in the "unrotated" frame)
            - Repeat process of generating masks as in 2.
        4.Process the cube map. I.e., where do the colours go?
            - Draw the cube
        5.If we scanned the whole cube, go to next step. Otherwise, go back to 1 (continue scanning).
        6.Display solution sequence. Allow camera to keep reading in infinite loop.
            - Terminate upon user keystroke.
        ### NOTES:
        Upon completion, the user will either:
        - Terminate completely (end entire file process)
        - Terminate this instance (let's you "start fresh")
        
        Any modifications you make to `config.json` or `constants.py` may not appear unless you "Terminate completely" first.

        It is recommended to edit these files while `main.py` is closed. Changes will be apparent upon next run.
        '''
        while True:
            start = time.time()
            ret, frame = self.cap.read()
            if (not LOW_RES_CAMERA):
                frame = cv.resize(frame,(800,600))
            original_frame = copy.copy(frame)

            # Main color recognition.
            # Convert from BGR to HSV color space.
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # Generate masks.
            masks = dict(
                (c, cv.inRange(hsv_frame, HSV_BOUND[c][0], HSV_BOUND[c][1])) for c in COLORS)
            masks['red'] |= cv.inRange(hsv_frame, *HSV_RED_UNION_BOUND)
            # Draw contours
            for c in COLORS:
                self.draw_cube_contour(frame, c, masks[c])
                if SHOW_MASK_PREVIEW and not SHOW_ROTATED_MASK_PREVIEW:
                    if (SHOW_RAW_CONTOUR_PREVIEW):
                        self.create_contour_preview(frame, c, masks[c])
                    else:
                        self.create_contour_preview(original_frame, c, masks[c])

            # Handle rotated recognition.
            # NOTE:
            # This code is nearly identical to the above.
            # However, to avoid ambiguity, verbosity was favoured.
            if (self.create_rotated_frame(original_frame)):
                # Convert from BGR to HSV color space (for the rotated frame).
                hsv_rotated_frame = cv.cvtColor(
                    self.rotated_frame, cv.COLOR_BGR2HSV)

                # Generate masks (for the rotated frame).
                rotated_masks = dict((c, cv.inRange(
                    hsv_rotated_frame, HSV_BOUND[c][0], HSV_BOUND[c][1])) for c in COLORS)
                rotated_masks['red'] |= cv.inRange(
                    hsv_rotated_frame, *HSV_RED_UNION_BOUND)

                if SHOW_ROTATED_MASK_PREVIEW:
                    original_rotated_frame = copy.copy(self.rotated_frame)

                for c in COLORS:
                    self.draw_cube_contour(
                        self.rotated_frame, c, rotated_masks[c], True)
                    if SHOW_ROTATED_MASK_PREVIEW:
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
            self.cube_map.draw_supercube(frame,(10,50),True)

            cv.imshow(MAIN_FRAME_NAME, frame)
            # Allow camera to keep reading, but not process any further.
            global PAUSE_AT_END
            if (PAUSE_AT_END):
                try:
                    cv.destroyWindow("Rotated image")
                except:
                    pass
                solve_moves = ""
                rev_solve_moves = ""
                try:
                    solve_moves = (kociemba.solve(self.cube_map.flatten_cube()))
                    rev_solve_moves = self.cube_map.reverse_notation(solve_moves)
                except ValueError:
                    print("RETRY")
                    solve_moves = "INVALID READ SEQUENCE: Please retry. Press 'x' to exit, 'q' to restart, 'c' to toggle SQUARE/CIRCLE custom center recognition"
                print("PAUSED")
                while (1):
                    ret, frame = self.cap.read()

                    if (not LOW_RES_CAMERA):
                        frame = cv.resize(frame,(800,600))

                    self.cube_map.draw_supercube(frame,(10,50),True)
                    if (LOW_RES_CAMERA):
                        cv.rectangle(frame,(10,380),(630,470),(254,217,255),-1)
                        self.cube_map.special_draw_text(frame,solve_moves,(12,405),0.7,2,48)
                        self.cube_map.special_draw_text(frame,rev_solve_moves,(12,460),0.35,1,100)
                    else:
                        cv.rectangle(frame,(10,500),(790,590),(254,217,255),-1)
                        self.cube_map.special_draw_text(frame,solve_moves,(12,525),0.7,2,60)
                        self.cube_map.special_draw_text(frame,rev_solve_moves,(12,575),0.35,1,120)
                    cv.imshow(MAIN_FRAME_NAME, frame)
                    key = cv.waitKey(1)
                    if (key == ord('q')):
                        self.cap.release()
                        cv.destroyAllWindows()
                        PAUSE_AT_END = False
                        return
                    if (key == ord('x')):
                        exit()
                    if (key == ord('c')):
                        with open("config.json") as f:
                            data = json.load(f)
                            flag = json.loads(data['hasCircleCenter'])
                            if (flag):
                                data['hasCircleCenter'] = "false"
                            else:
                                data['hasCircleCenter'] = "true"
                        with open("config.json","w") as fw:
                            json.dump(data,fw,indent=4)

                        self.cap.release()
                        cv.destroyAllWindows()
                        PAUSE_AT_END = False
                        return

            self.cube_angles.clear()
            self.cube_map.ortho.clear()
            self.cube_map.raw_points.clear()


            end = time.time()

            key = cv.waitKey(1)
            if (key == ord('q')):
                break
            if (key == ord('x')):
                exit()
            if (key == ord('r')):
                self.cube_map.index = self.cube_map.bk[self.cube_map.index]
                if (self.cube_map.index == -1):
                    self.cube_map.index = 2
            if (key == ord('c')):
                with open("config.json") as f:
                    data = json.load(f)
                    flag = json.loads(data['hasCircleCenter'])
                    if (flag):
                        data['hasCircleCenter'] = "false"
                    else:
                        data['hasCircleCenter'] = "true"
                with open("config.json","w") as fw:
                    json.dump(data,fw,indent=4)
                break             

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

while True:
    cam = Camera()

    with open("config.json") as f:
        data = json.load(f)

    HAS_CIRCLE_CENTER = json.loads(data['hasCircleCenter'])

    if (HAS_CIRCLE_CENTER):
        MAIN_FRAME_NAME = "Rubik's Cube Reader (CIRCLE)"
    else:
        MAIN_FRAME_NAME = "Rubik's Cube Reader (SQUARE)"

    cam.__init__
    cam.run_main_process()
