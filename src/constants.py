SHOW_CONTOUR_COLOR_TEXT = True
SHOW_ENTIRE_BOUNDING_RECTANGLE = False
SHOW_CONTOUR_PREVIEW = True

# CONTOURS
# --------
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
# -----------
# How many frame readings required before considering
# to lock color mapping for this side/face.
# Lower MIN_TOTAL_READING_BUFFER => faster cube mapping, but higher chance of error
# Also change MIN_COLOR_COUNT_THRESHOLD accordingly
MIN_TOTAL_READING_BUFFER = 5

# Reset reading buffer count to 0 once limit is reached.
MAX_TOTAL_READING_BUFFER = 10

# If after MIN_COLOR_FREQUENCY number of readings, all
# colors have at least this percentage of confidence.
MIN_COLOR_CONFIDENCE_THRESHOLD = 0.90

# For each cube piece, one color should be read at least this many times.
# MIN_TOTAL_READING_BUFFER always takes higher precedence.
MIN_COLOR_COUNT_THRESHOLD = 5

# length of a sticker (square)
STICKER_LENGTH = 20
# gap in between adjacent stickers
STICKER_GAP = 3
# sum of sticker gap + length
STICKER_SUPER_LENGTH = STICKER_LENGTH + STICKER_GAP

CUBE_LENGTH = 3*STICKER_LENGTH + 4 * STICKER_GAP+2
CUBE_GAP = 10