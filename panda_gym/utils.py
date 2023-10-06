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
