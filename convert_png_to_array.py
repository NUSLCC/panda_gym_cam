#import pyrealsense2 as rs
import numpy as np
import cv2

active_image = cv2.imread('active1_Color.png')
static_image = cv2.imread('static1_Color.png')

# active_image = cv2.imread('/home/franka/philip_ws/active1_Color.png')
# static_image = cv2.imread('/home/franka/philip_ws/static1_Color.png')

# active_image = cv2.cvtColor(active_image, cv2.COLOR_BGR2RGB)
# static_image = cv2.cvtColor(static_image, cv2.COLOR_BGR2RGB)

static_image = static_image[50:580,100:900,:]

resized_active_image = cv2.resize(active_image, (160,90))
resized_static_image = cv2.resize(static_image, (160,90))

# cv2.imshow("active",resized_active_image)
# cv2.imshow("static", resized_static_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

final_image = np.concatenate([resized_active_image, resized_static_image], axis=-1)
final_image = np.transpose(final_image, (2, 0, 1))

#print("final img shape", final_image.shape)

observation = {
    "observation": final_image.astype(np.uint8),
    "desired_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
    "achieved_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
    "state": np.random.uniform(-10, 10, (10,)).astype(np.float32)
}

print(observation)