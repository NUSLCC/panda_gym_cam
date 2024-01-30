import numpy as np
from collections import deque
from datetime import datetime

class ImageContainer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.image_deque = deque(maxlen=max_size)

    def add_image(self, image_array):
        # Assuming image_path is the path to the image file
        timestamp = datetime.now()
        image_array = np.transpose(image_array, (2, 0, 1))
        self.image_deque.append((timestamp, image_array))

    def get_images(self):
        return np.array([image[1] for image in self.image_deque])
        # return np.concatenate([image[1] for image in self.image_deque], axis=concat_dim)


# Example usage:
image_container = ImageContainer(max_size=2)

# Adding images
image_array1 = np.zeros((10, 10, 3))
image_array2 = np.ones((10, 10, 3))
image_array3 = np.random.rand(10, 10, 3)

image_container.add_image(image_array1)
image_container.add_image(image_array2)
image_container.add_image(image_array3)

# Getting current images
print("Current images:", image_container.get_images().shape)

# Adding a new image (automatically removes the oldest one)

image_array4 = np.zeros((10, 10, 3))
image_container.add_image(image_array4)

# Getting current images after adding a new image
print("Current images:", image_container.get_images().shape)
