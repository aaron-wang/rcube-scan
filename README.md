# Rubik's Cube Scanner

A desktop app using OpenCV that scans and solves 3x3 Rubik's Cubes using a regular webcam!

- Color detection :small_red_triangle:
- Rotation recognition :arrow_right_hook:
- Outputs solution :bulb:
- Recognizes square center :white_large_square: AND circle-center cubes :white_circle:

**Scan Demo:**

Square center:
<!-- [gif2] -->
![GIF-demo-2-square-conde](https://user-images.githubusercontent.com/66176554/149688477-2ee99327-e4c7-482d-841d-731d89fc36d2.gif)

Circle center:
<!-- [gif1] -->
![GIF-demo-1-circle](https://user-images.githubusercontent.com/66176554/149688131-62ae2dcd-8489-4e81-a506-10a864f11815.gif)


<!--**Color Detection:**-->

<!--[gif2]-->

<!--**Rotation Recognition**-->

<!--[gif3]-->

<!--**Custom Circle/Square (Center) Modes:**-->

<!--[gif4a]-->

<!--[gif4b]-->

<!--[insert some images here]-->

**Interface example:**

<!-- [image 1] -->
![image-demo-1](https://user-images.githubusercontent.com/66176554/149688160-c99d6892-bbfe-4ca7-b270-c8ee029666f4.png)


**Solution output (different cube from above):**

<!-- [image 2] -->

![image-demo-2](https://user-images.githubusercontent.com/66176554/149688172-5768c4db-ee6e-4ee8-912d-8e634aa5c228.png)


<!-- **Table of contents:**

[toc] -->

## Installation

Have Python 3 installed.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required dependencies.

```bash
pip install numpy
pip install opencv-python
pip install opencv-contrib-python
pip install kociemba
```
Clone the repository locally.

## Usage

The following section is verbose in order to eliminate ambiguity.

### Start up:

Navigate to where `.../rcube-scan/src`  is installed. This is the directory including `main.py`. 

Run the program by:

```
python main.py
```

This will open a webcam capture window

### Scanning:

Scramble your cube!

- There is no calibration process required currently.

- If you have a square center Rubik's cube, then great! 
  - Please ensure the title of the window says "Rubik's Cube Scanner (SQUARE)".
  - If not, press `c`

- If you have a circle center one, then please press `c` to toggle specialized center circle scanning.
  - Please ensure the title of the window says "Rubik's Cube Scanner (CIRCLE)".

Now assuming you have a scrambled cube, there is an order you must follow. Relative to the camera, hold your Rubik's Cube as follows:

- Red center facing towards the camera
- Blue center on top (facing upwards)

Now verify that:

- Green center is on bottom
- White center is on left, yellow center is on right.

---

**Note: the program may have scanned the red face before you oriented it correctly**

- If this is the case, please either:
  - Press `r` to rescan the previous face -- ***Optimal***
  - Press `q` to restart the program 
  - Press `x` to exit. You must manually run the program again.
- If you see one square of the cube map filled out (in vibrant colors) then the scanning for this face is complete.
- Once you get used to scanning order, restarting/rescanning is not necessary.

---

Now assuming you have red facing the camera, and blue on top. Here's the steps:

1. Start at red center facing the camera, with blue on top. Wait for scan to complete.
   - The scanning is complete once the "faded" colors turn vibrant.

2. Turn the cube left relative to the camera so you arrive **to yellow**
   - You should be on the yellow **center** face.
     - Blue on top, green on bottom; red on left, orange on right.
   - Wait for scan to complete.
3. Turn the cube left ... **to orange**
   - Blue on top, green on bottom; yellow on left, white on right.
4. Turn the cube left ... **to white**
   - Blue on top, green on bottom; orange on left, red on right.
5. Turn the cube to **blue** center - there is a specific way you **must** orient this
   - The easiest way to remember:
     - Turn the cube left again, so you arrive at the red face (where you started)
     - Turn the cube down so that the blue face is towards the camera
   - If you did it correctly: 
     - Blue center facing camera
     - Orange on top, red on bottom; white on left, yellow on right. 
6. Turn the cube to **green** center
   - Rotate the cube 180 degrees down so you reach the green center.
   - If you did it correctly: 
     - Green center facing camera
     - Red on top, orange on bottom; white on left, yellow on right.
7. Scanning should be complete

The preceding steps are verbose, but after trying this process a few time it will become intuitive and quick.

#### Summary:

**Red** (blue on top) -> **Yellow** -> **Orange** -> **White** -> **Blue** (red on bottom)-> **Green** (red on top)

### Solving

There are two scenarios after scanning is complete

1. Error: There was an issue in scanning (Please read the "Error" section below)
2. Success: The solution sequence of move is outputted on screen per standard Rubik's Cube notation.
   - The program will no longer process frames, but will output the raw frames of the webcam capture 
   - Follow the notation on screen (big font). 
     - The small font is for reversing a solved cube into your unsolved state.
   - If you followed the notations on screen but ended up with a unsolved cube, then either:
     - Ensure you have followed the notation correctly: it's common to mess up just one or two moves (which leads to a non-solved cube) OR
     - There was an issue with scanning, where the state was valid, but did not match your cube exactly. This case is extremely rare
   - If you want to scan another cube, please
     - Press `q` to restart the instance.
     - Press `x` to exit. You must manually run the program again.

If you receive an error please read the following:

#### Error:

A solution is produced from the inputs of where the colors are on the cube. 

If you scan a cube that's impossible to solve, then the solving algorithm raises an error.

This could mean:

- the color of one or more squares were scanned incorrectly
- the cube was not rotated correctly when scanned
  - the program enforces an order of scanning (based on the center color). 
  - it cannot determine whether the current face is oriented/rotated in the correct way.
- the cube is actually in an invalid state (impossible to solve)
  - this occurs commonly if you've ever
    - physically switched stickers
    - physically twisted corners or flipped edges 
    - taken apart a cube and reassembled it into a non-solved state
  - this will cause you to consistently get errors telling you to rescan

### Tips:

These are not exhaustive lists. 

#### DO:

- Use an external webcam if possible
  - Certain integrated webcams for laptops are too low quality or "dull" the color vibrancy
- Be in a well-lit room
- Remove any objects with prominent squares with solid color (in the background)
  - If circle-center cube scanning, avoid circles in the background too
- Avoid checkerboard pattern rooms :slightly_smiling_face:

#### DON'T:

- **Have an invalid (unsolvable) cube state!**
  - This occurs commonly if you've ever
    - physically switched stickers
    - physically twisted corners or flipped edges 
    - taken apart a cube and reassembled it into a non-solved state
  - This will cause you to consistently get errors telling you to rescan
- Have excessive lighting (directly onto the cube)
  - For glossy stickers, this reflects white light.
- Have poor lighting
- Obstruct cube face with objects or fingers
- Use sticker-less or borderless cubes
  - If squares of the same color are bordering each other, this registers as one quadrilateral or concave polygon
- Use cubes with black stickers replacing white stickers
- Use non-3x3 Rubik's cubes
- Scan two cubes at the same time




## Controls

### Note: 

If you have a **circle center** style cube:

- please ensure you are in "circle" detection mode.

- Press `c` to toggle. Mode will be indicated in OpenCV window.

If you have a **square center** style cube:

-  the default is "square" center detection mode.

- If you are having issues scanning, check if you are in "circle" mode. In that case, press `c` to toggle.

### Global:

- `c`: Toggle circle center detection on and off (square center detection)
  - This instance will close, and a new window will open
  - The title of the OpenCV window will have `Rubik's ... (CIRCLE)` or `Rubik's ... (SQUARE)` accordingly
- `x`: Quit. Terminate completely (exit)
- `q`: Restart. Terminate this instance (e.g., you want to scan another cube: automatically reopen window)

- `r`: Rescan the previous face.
  - You can press this as many times as you want
  - The previous face color will become *faded* instead of *solid*
  - Once scanning is done, `r` does nothing.



## Configuration

`config.json` contains the follow configurations:

### Color Calibration - `config.json`

There is a default configuration provided. A copy of this default version is stored in `default_config.json`. 

- The main process uses `config.json` for its color configuration. 
- Feel free to edit it locally for your environment.

Color thresholds are based on HSV values, setting lower and upper bounds respectively for each.

- There are six colors, **do not** rename the color names
- Each color `X` has two arguments, `upper_bound` and `lower_bound`:
  - Informally, anything between these bounds is considered color `X`.
  - Each contain arrays of three 8 bit unsigned values (0 - 255)

I may release a calibration tool/mode if I have time to refine it for public use.

### Technical - Flags

`hasCircleCenter:` `true` if scanning mode is set for circle center Rubik's cubes.

- `false` if scanning for square center (default)

#### `constants.py`

Rest of flags/constants are in `constants.py`. You will **not need to edit any of these**. I have provided a basic list of explanations below if you are interested:

- Several debugging flags (true/false)
- Contour constants
  - `MIN_CONTOUR_THRESHOLD`: minimum "area" needed for contour to be considered 
    - removes small or extraneous spots of color
  - `MAX_CONTOUR_SQUARE_EDGE_THRESHOLD`: Maximum square edge width
    - The edge of each square should not be too big
  - `LENIENT_SQUARE_CONTOUR_BYPASS_RATIO`: Maximum "absolute" ratio of length to width
    - Bypass non-square like contours; Heuristic guess
  - `STRICT_SQUARE_CONTOUR_BYPASS_RATIO`: see `LENIENT_SQUARE_CONTOUR_BYPASS_RATIO`
    - Used in second tiered approach similar to the `LENIENT_SQUARE_CONTOUR_BYPASS_RATIO`
  - `CIRCLE_CONTOUR_BYPASS_SCALE`: Used in determining circle:
    - Circle center if `circ_area <= CIRCLE_CONTOUR_BYPASS_SCALE * rot_area`
    - Determine circle bounding - comparison with bounded square
  - `ANGLE_EPSILON`: Allow rotations from `[0, 45 - ANGLE_EPSILON]`
    - default value is `1`, so a reading of (44,45] degrees is "uncertain"
  - `CENTER_EPSILON`: center cube relative/absolute pixel error. Re: `find_center_cube()`
    - Used in determining square center.
- Cube mapping
  - How "confident" does the scan have to be?
    - color % accuracy
    - minimum # of frames
  - Drawing of the cube preview on screen

## Inspiration
I was thinking of project ideas and came across this video one day:
- https://youtu.be/VaW1dmqRE0o

For the rest, I played around with features displayed in the OpenCV-Python tutorials:

- https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

## License
Licensed per the [MIT](https://choosealicense.com/licenses/mit/) License.
