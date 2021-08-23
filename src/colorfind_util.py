import cv2 as cv
import numpy as np

color = np.uint8([[[
#    84,60,52
    # 76,52,36
    # 52,36,28
    # 36,36,58
    # 192,207,204
    92,108,196
]]])

hsv_color = cv.cvtColor(color, cv.COLOR_BGR2HSV)


# frame = cv.imread('img/cube1.jpg')

# hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

# mask = cv.inRange(hsv, 
#     (0,100,76),
#     (10,255,255)
# )

# cv.imshow("orange",mask)
# cv.imshow("frame",frame)
# cv.waitKey()
# cv.destroyAllWindows()

print(hsv_color)
