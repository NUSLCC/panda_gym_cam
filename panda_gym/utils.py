import sys
import os
import requests
import numpy as np
import math
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mae import models_mae
from mae.util import pos_embed


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist


def calculate_coverage_ratio(a: tuple, b: tuple) -> float:
    """
    Calculate the ratio of a covered by b.

    Args:
        a (tuple): A tuple containing (x1, y1, x2, y2) coordinates of the first box.
        b (tuple): A tuple containing (x1, y1, x2, y2) coordinates of the second box.

    Returns:
        float: The ratio of a covered by b, ranging from 0 to 1.
    """
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(a[0], b[0])
    y1_inter = max(a[1], b[1])
    x2_inter = min(a[2], b[2])
    y2_inter = min(a[3], b[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of a
    a_area = (a[2] - a[0]) * (a[3] - a[1])

    # Calculate the coverage ratio
    coverage_ratio = intersection_area / a_area

    return coverage_ratio


def calculate_coverage_ratio_nparray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the ratio of first array covered by second array.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
    Returns:
        np.ndarray: The ratio of a covered by b, ranging from 0 to 1 in an array.
    """
    assert a.shape == b.shape
    # Calculate the coordinates of the intersection rectangles for all pairs of boxes
    x1_inter = np.maximum(a[:, 0], b[:, 0])
    y1_inter = np.maximum(a[:, 1], b[:, 1])
    x2_inter = np.minimum(a[:, 2], b[:, 2])
    y2_inter = np.minimum(a[:, 3], b[:, 3])

    # Calculate the areas of intersection for all pairs of boxes
    intersection_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(
        0, y2_inter - y1_inter
    )

    # Calculate the areas of a for all pairs of boxes
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])

    # Calculate the coverage ratios for all pairs of boxes
    coverage_ratio = intersection_area / a_area

    return coverage_ratio


def calculate_object_range(initial_x_coord, initial_y_coord, initial_z_coord):
    """
    Calculates the (x,y,z) array ranges where the object can be generated, such that it is inside robot-cam fov

    Args:
        initial_x_coord (float): x coordinate of panda robot-cam in neutral pos
        initial_y_coord (float): y coordinate of panda robot-cam in neutral pos
        initial_z_coord (float): z coordinate of panda robot-cam in neutral pos

    Returns:
        obj_range_low (np.ndarray): coordinates of the minimum of obj range
        obj_range_high (np.ndarray): coordinates of the maximum of obj range
    """
    horiz_total_dis = 2 * initial_z_coord * math.tan(math.radians(87) / 2)
    vert_total_dis = 2 * initial_z_coord * math.tan(math.radians(58) / 2)
    x_min = initial_x_coord - vert_total_dis / 2
    x_max = initial_x_coord + vert_total_dis / 2
    y_min = initial_y_coord - horiz_total_dis / 2
    y_max = initial_y_coord + horiz_total_dis / 2

    # Calculate obj_range_low and obj_range_high - they form the bounding box where the object can be randomly generated
    obj_range_low = np.array([x_min, y_min, 0])
    obj_range_high = np.array([x_max, y_max, 0])

    return obj_range_low, obj_range_high


def generate_object_range():
    """
    Calculates the (x,y,z) array ranges where the object can be generated, such that it is inside reachable area of Panda arm

    Args:
        initial_x_coord (float): x coordinate of panda robot-cam in neutral pos
        initial_y_coord (float): y coordinate of panda robot-cam in neutral pos
        initial_z_coord (float): z coordinate of panda robot-cam in neutral pos

    Returns:
        obj_range_low (np.ndarray): coordinates of the minimum of obj range
        obj_range_high (np.ndarray): coordinates of the maximum of obj range
    """

    # x_min = -0.25
    # x_max = 0.25
    # y_min = -0.35
    # y_max = 0.35
    x_min = -0.15
    x_max = 0.2
    y_min = -0.3
    y_max = 0.3
    # Calculate obj_range_low and obj_range_high - they form the bounding box where the object can be randomly generated
    obj_range_low = np.array([x_min, y_min, 0])
    obj_range_high = np.array([x_max, y_max, 0])

    return obj_range_low, obj_range_high


def generate_semicircle_object_range():
    """
    Calculates the (x,y,z) array ranges where the object can be generated, such that it is inside semi-circle reachable area of Panda arm (radius = 0.64m)

    Returns:
        obj_range_low (np.ndarray): coordinates of the minimum of obj range
        obj_range_high (np.ndarray): coordinates of the maximum of obj range
    """

    radius = 0.64
    base_x = -0.68  # x coord of base of panda robot
    x_min = -0.36
    x_max = (
        -0.04
    )  # The base is 0.24 m to the white table. So x max = -0.04 is the maximum it can go while being 80% of the actual reach.
    y_min = -0.64
    y_max = 0.64
    sampled_x = np.random.uniform(x_min, x_max)
    x_distance_from_base = sampled_x - base_x
    sampled_y = math.sqrt(
        radius**2 - sampled_x**2
    )  # need to adjust y according to equation of a circle

    # Calculate obj_range_low and obj_range_high - they form the bounding box where the object can be randomly generated\
    obj_range_low = np.array([sampled_x, -sampled_y, 0])
    obj_range_high = np.array([sampled_x, sampled_y, 0])

    return obj_range_low, obj_range_high


def sample_object_obstacle_goal(
    object_size,
    goal_range_low,
    goal_range_high,
    object_obstacle_distance,
    obstacle_size,
    obstacle_1_pos,
    obstacle_2_pos,
):
    """
    Calculates the (x,y,z) array goal, such that it is inside reachable area of Panda arm and a certain distance away from obstacles

    Args:
        object_size (float): radius of the object
        obj_range_low (np.ndarray): coordinates of the minimum of obj range
        obj_range_high (np.ndarray): coordinates of the maximum of obj range
        object_obstacle_distance (float): Minimum distance from object to obstacles in m
        obstacle_size (float): Half extent of the obstacle
        obstacle_1_pos (np.ndarray): (x,y,z) array of obstacle 1 pos
        obstacle_2_pos (np.ndarray): (x,y,z) array of obstacle 2 pos

    Returns:
        obj_range_low (np.ndarray): coordinates of the minimum of obj range
        obj_range_high (np.ndarray): coordinates of the maximum of obj range
    """

    obstacle_1_x, obstacle_1_y = obstacle_1_pos[0], obstacle_1_pos[1]
    obstacle_2_x, obstacle_2_y = obstacle_2_pos[0], obstacle_2_pos[1]

    exclude_ranges = {
        "x": [
            (
                obstacle_1_x - obstacle_size - object_obstacle_distance,
                obstacle_1_x + obstacle_size + object_obstacle_distance,
            ),
            (
                obstacle_2_x - obstacle_size - object_obstacle_distance,
                obstacle_2_x + obstacle_size + object_obstacle_distance,
            ),
        ],
        "y": [
            (
                obstacle_1_y - obstacle_size - object_obstacle_distance,
                obstacle_1_y + obstacle_size + object_obstacle_distance,
            ),
            (
                obstacle_2_x - obstacle_size - object_obstacle_distance,
                obstacle_2_x + obstacle_size + object_obstacle_distance,
            ),
        ],
    }

    while True:
        goal = np.array([0.0, 0.0, object_size / 2])  # z offset for the sphere center
        noise = np.random.uniform(goal_range_low, goal_range_high)
        goal += noise

        if not any(
            i[0] <= goal[0] <= i[1] for i in exclude_ranges["x"]
        ) and not any(  # if sampled x value is not within the exclude ranges
            j[0] <= goal[1] <= j[1] for j in exclude_ranges["y"]
        ):  # if sampled y value is not within the exclude ranges
            break  # if sampled x or y value is inside exclude ranges, sample again

    return goal


def colorjitter(img, brightness, contrast, saturation, hue):
    """
    Applies color jitter to the input image
    Args:
        RGB image (np.ndarray) of either active-view or passive-view camera. Shape of W, H, C.
    Returns:
        RGB image (np.ndarray) that has brightness, contrast, saturation and hue jittered. Shape of W, H, C.
    """
    H, W, _ = img.shape
    img = np.array(img).astype(np.uint8).transpose(1, 0, 2)
    pil_img = Image.fromarray(img)
    color_jitter = transforms.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    pil_img = color_jitter(pil_img)
    jittered_img = np.asarray(pil_img).astype(np.uint8).transpose(1, 0, 2)

    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.reshape(H, W, 3))
    # axes[0].set_title('Original image')
    # axes[1].imshow(jittered_img.reshape(H, W, 3))
    # axes[1].set_title('Jittered image')
    # plt.show()

    return jittered_img

def mask_image(img, mask_ratio=0.25):
    """
    Mask out a percentage of the image.
    Args:
        RGB image (np.ndarray) of either active-view or passive-view camera. Shape of W, H, C.
        Mask_ratio (float): Percentage of the image to be masked (default is 25%)
    Returns:
        RGB image (np.ndarray) that is masked out (black). Shape of W, H, C.
    """
    W, H, _ = img.shape
    num_masked_pixels = int(mask_ratio * H * W)

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

    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img.reshape(90,160,3))
    # axes[0].set_title('Original image')
    # axes[1].imshow(masked_image.reshape(90,160,3))
    # axes[1].set_title('Masked image')
    # plt.show()

    return masked_image

def resize_image(img, width=160, height=90):
    """Resize image into width x height
    Args:
        RGB image (np.ndarray)
    Returns:
        RGB image (np.ndarray of shape width, height, channels)
    """
    W, H, _ = img.shape
    img = img.reshape(H,W,3)
    resized_image = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    # img = np.array(img).astype(np.uint8)
    # resized_image = np.array(resized_image).astype(np.uint8)
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img)
    # axes[0].set_title('Original image')
    # axes[1].imshow(resized_image)
    # axes[1].set_title('Resized image')
    # plt.show()

    resized_image = resized_image.reshape(width,height,3) 

    # print(f"Resized img shape: {resized_image.shape}")

    return resized_image


def masked_auto_encoder(img):
    """
    Applies MAE to the input image
    Args:
        RGB image (np.ndarray) of shape of W, H, C.
    Returns:
        RGB image (np.ndarray) which includes the reconstruction of decoder of shape of W, H, C.
    """
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    img = np.array(img).reshape(224, 400, 3)

    img = Image.fromarray(img).resize((224, 224))
    img = np.array(img) / 255.0

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    plt.rcParams["figure.figsize"] = [5, 5]
    show_image(torch.tensor(img))

    chkpt_dir = "mae/mae_visualize_vit_large_ganloss.pth"
    model_mae_gan = prepare_model(chkpt_dir, "mae_vit_large_patch16")

    print("MAE with extra GAN loss:")
    run_one_image(img, model_mae_gan)


def show_image(image, title=""):
    # image is [H, W, 3]
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    assert image.shape[2] == 3
    x = image * imagenet_std + imagenet_mean
    plt.imshow(torch.clip(x * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # Convert tensor to RGB array with reverse normalization

    im_paste_rgb = (
        torch.clip((im_paste * imagenet_std + imagenet_mean) * 255, 0, 255)
        .int()
        .numpy()
    )  # return this later

    # make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.tight_layout()
    plt.show()


def velocity_calculator(
    target_position: np.ndarray,
    initial_velocity: np.ndarray,
    x_min: float = -0.15,
    x_max: float = 0.2,
    y_min: float = -0.3,
    y_max: float = 0.3,
):
    """
    Calculates velocity of the target to avoid it falling off the table

    Args:
        Current position of target (np.ndarray): (x, y, z) pose
        Initial_velocity (np.ndarray): (x, y, z) velocity of target
        x_min (float): Min x value where target can be.
        x_max (float): Max x value where target can be.
        y_min (float): Min y value where target can be.
        y_max (float): Max y value where target can be.

    Returns:
        Velocity (np.ndarray): Velocity of target
    """
    target_position_x, target_position_y = target_position[0], target_position[1]
    if target_position_x > x_max or target_position_x < x_min:
        initial_velocity[0] *= -1
    if target_position_y > y_max or target_position_y < y_min:
        initial_velocity[1] *= -1
    modified_velocity = initial_velocity
    return modified_velocity


def sine_velocity(
    target_position: np.ndarray,
    initial_velocity: np.ndarray,
    A: float = 3,  # used to be 0.1
    B: float = 80,
):
    """
    Creates sinusoidal velocity of the target: v in x direction = A sin(By). v in y direction is an initialised constant.
    NB: Time period T = 2pi/B

    Args:
        Current position of target (np.ndarray): (x, y, z) pose
        Initial_velocity (np.ndarray): (x, y, z) velocity of target
    Returns:
        Velocity (np.ndarray): Velocity of target
    """
    initial_velocity[0] = A * math.sin(B * target_position[1])  # change velocity of x
    return initial_velocity


def color_threshold_pixel_counter(
    img: np.ndarray,
    lower_bound: np.ndarray = np.array([40, 50, 50]),
    upper_bound: np.ndarray = np.array([80, 255, 255]),
):
    """
    Performs thresholding in the input image and returns the number of (default: green object) pixels in the img within the lower and upper bound.

    Args:
        Img (np.ndarray): observation image of panda active camera
        Upper bound (np.ndarray): upper bound of HSV to threshold
        Lower bound (np.ndarray): lower bound of HSV to threshold
    Returns:
        Pixel count (int): Number of green pixels in the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    pixel_count = np.sum(mask == 255)
    return pixel_count
