import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

lower_green_bound = np.array([40, 50, 50]) # HSV bounds
upper_green_bound = np.array([80, 255, 255])

img_path = 'test_image.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

W, H, _ = img.shape
num_masked_pixels = int(0.25 * H * W)

# Generate random indices to mask
masked_indices = np.random.choice(H * W, num_masked_pixels, replace=False)

# Create a binary mask
mask = np.ones(H * W)
mask = mask.reshape(-1)
mask[masked_indices] = 0
mask = mask.reshape((W,H))
mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

# Apply the mask to each channel
masked_image = img * mask_3d

img = np.array(img).astype(np.uint8)
masked_image = np.array(masked_image).astype(np.uint8)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[0].set_title('Original image')
axes[1].imshow(masked_image)
axes[1].set_title('Masked image')
plt.show()


# cv2.imshow("Image", green_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
