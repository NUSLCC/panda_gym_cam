import numpy as np


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


def calculate_coverage_ratio(box1, box2):
    """
    Calculate the ratio of box1 covered by box2.

    Args:
        box1 (tuple): A tuple containing (x1, y1, x2, y2) coordinates of the first box.
        box2 (tuple): A tuple containing (x1, y1, x2, y2) coordinates of the second box.

    Returns:
        float: The ratio of box1 covered by box2, ranging from 0 to 1.
    """
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of box1
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # Calculate the coverage ratio
    coverage_ratio = intersection_area / box1_area

    return coverage_ratio