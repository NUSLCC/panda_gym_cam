import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

lower_green_bound = np.array([40, 50, 50]) # HSV bounds
upper_green_bound = np.array([80, 255, 255])

img_path = 'test_img.png'
img = cv2.imread(img_path)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(img_hsv, lower_green_bound, upper_green_bound)
cv2.imshow("Green image", green_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

green_pixel_count = np.sum(green_mask == 255)
print(f'Green pixel count: \n {green_pixel_count}')