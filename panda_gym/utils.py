import numpy as np
import math
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space, is_image_space_channels_first
from stable_baselines3.common.type_aliases import TensorDict
from typing import Dict
from gymnasium import spaces
import gymnasium as gym
from timm import create_model
from PIL import Image


class NatureCNN(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0] # last channel is feature channel from raw data [B, C H, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten shape by doing one forward pass
        with torch.no_grad():
            # sample_array = observation_space.sample()[None]
            # sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
            # n_flatten = self.cnn(sample_tensor).shape[1]
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class CNNLSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,0,]
            n_flatten = self.cnn(torch.as_tensor(each_T).float()).shape[1]

        self.fc_lstm = nn.Linear(n_flatten, features_dim)
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.fc1 = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.cnn(observations[:, t, :, :, :]) 
            x = self.fc_lstm(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        return x


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 512,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=cnn_output_dim)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "observation":
                extractors[key] = CNNLSTM(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
    
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
    intersection_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

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
    horiz_total_dis = 2*initial_z_coord*math.tan(math.radians(87)/2) 
    vert_total_dis = 2*initial_z_coord*math.tan(math.radians(58)/2)
    x_min = initial_x_coord - vert_total_dis/2
    x_max = initial_x_coord + vert_total_dis/2
    y_min = initial_y_coord - horiz_total_dis/2
    y_max = initial_y_coord + horiz_total_dis/2

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
    base_x = -0.68 # x coord of base of panda robot
    x_min = -0.36
    x_max = -0.04  # The base is 0.24 m to the white table. So x max = -0.04 is the maximum it can go while being 80% of the actual reach. 
    y_min = -0.64
    y_max = 0.64
    sampled_x = np.random.uniform(x_min, x_max)
    x_distance_from_base = sampled_x - base_x
    sampled_y = math.sqrt(radius ** 2 - sampled_x ** 2) # need to adjust y according to equation of a circle

    # Calculate obj_range_low and obj_range_high - they form the bounding box where the object can be randomly generated\
    obj_range_low = np.array([sampled_x, -sampled_y, 0])
    obj_range_high = np.array([sampled_x, sampled_y, 0]) 

    return obj_range_low, obj_range_high

def colorjitter(img, brightness, contrast, saturation, hue):
    """
    Applies color jitter to the input image
    Args:
        RGB image (np.ndarray) of either active-view or passive-view camera. Shape of W, H, C.
    Returns:
        RGB image (np.ndarray) that has brightness, contrast, saturation and hue jittered. Shape of W, H, C.
    """
    org_img = np.array(img).astype(np.uint8)
    img = np.array(img).astype(np.uint8).transpose(1, 0, 2)
    pil_img = Image.fromarray(img)
    color_jitter = transforms.ColorJitter(brightness = brightness, contrast=contrast, saturation=saturation, hue=hue)
    pil_img = color_jitter(pil_img)
    jittered_img = np.asarray(pil_img).astype(np.uint8).transpose(1, 0, 2)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(org_img.reshape(224, 400, 3))
    axes[0].set_title('Original image')
    axes[1].imshow(jittered_img.reshape(224, 400, 3))
    axes[1].set_title('Jittered image')
    plt.show()
    return jittered_img

def show_image(image,title=''):
    # image is [H, W, 3]
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    assert image.shape[2] == 3
    x = image * imagenet_std + imagenet_mean
    plt.imshow(torch.clip(x * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

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
    A: float = 3, # used to be 0.1
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
    initial_velocity[0] = A*math.sin(B*target_position[1]) # change velocity of x
    return initial_velocity

def color_threshold_pixel_counter(
        img: np.ndarray,
        lower_bound: np.ndarray = np.array([40,50,50]),
        upper_bound: np.ndarray = np.array([80, 255, 255])
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