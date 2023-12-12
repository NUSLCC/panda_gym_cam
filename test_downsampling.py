import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

lower_green_bound = np.array([40, 50, 50]) # HSV bounds
upper_green_bound = np.array([80, 255, 255])

img_path = 'test_img.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Img shape: {img.shape}")

downscale_factor = 8

downsampled_image = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)
#downsampled_image = img[::downscale_factor, ::downscale_factor, :]

img = np.array(img).astype(np.uint8)
masked_image = np.array(downsampled_image).astype(np.uint8)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[0].set_title('Original image')
axes[1].imshow(downsampled_image)
axes[1].set_title('Downsampled image')
plt.show()


# cv2.imshow("Image", green_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
