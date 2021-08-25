SHOW_CONTOUR_COLOR_TEXT = True
SHOW_ENTIRE_BOUNDING_RECTANGLE = False
SHOW_CONTOUR_PREVIEW = True

# CONTOURS
# Global contour area requirement
MIN_CONTOUR_THRESHOLD = 500
# Maximum square edge width.
MAX_CONTOUR_SQUARE_EDGE_THRESHOLD = 130
# Bypass non-square like contours - heuristic guess
LENIENT_SQUARE_CONTOUR_BYPASS_RATIO = 1.3
STRICT_SQUARE_CONTOUR_BYPASS_RATIO = 1.2
# Determine circle bounding - comparison with bounded square
CIRCLE_CONTOUR_BYPASS_SCALE = 1.1
# Rotation angle clemency
ANGLE_EPSILON = 1.0

# CUBE MAPPING
# How many frame readings of a color required before 
# locking color reading for this cube side/face.
# Dependent on system
MIN_TOTAL_READING_BUFFER = 10

# 30 for slower

MAX_TOTAL_READING_BUFFER = 40
# 60 for slower


# If after MIN_COLOR_FREQUENCY number of readings, all
# colors have at least this percentage of confidence.
MIN_COLOR_CONFIDENCE_THRESHOLD = 0.90

