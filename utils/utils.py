import numpy as np
import cv2


def process_image(filepath, threshold1=100, threshold2=250):
    """
    Takes an image filepath, converts it to black and white,
    detects and links edges, and returns x and y coordinates
    of the edge points. Edge sensitivity can be adjusted using
    threshold1 and threshold2.

    Args:
        filepath (str):
            The file path to the input image.
        threshold1 (int):
            Lower threshold for Canny edge detection (sensitivity).
        threshold2 (int):
            Upper threshold for Canny edge detection (sensitivity).

    Returns:
        (np.ndarray, np.ndarray):
            Two arrays of x and y coordinates of edge points.
    """
    # Load the image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Detect edges using Canny with adjustable thresholds
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)

    # Find contours (linked edges)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract x and y coordinates of edge points
    x_coords = []
    y_coords = []

    for contour in contours:
        for point in contour:
            x, y = point[0]
            x_coords.append(x)
            y_coords.append(y)

    return np.array(x_coords), np.array(y_coords)



def calculate_weighted_angles_by_edge_length(x_values, y_values):
    """
    Calculates a weighted average of the included angles at
    each vertex of a polygon, where each angle is weighted by
    the sum of the lengths of its adjacent edges.

    Args:
        x_values (list or array):
            The x-coordinates of the polygon's vertices.
        y_values (list or array):
            The y-coordinates of the polygon's vertices.

    Returns:
        weighted_avg_angle (float):
            The edge-length-weighted average of the
            included angles (in degrees).
        roundness (float):
            The roundness of the polygon, defined as the
            weighted average angle divided by 180.
    """
    # Input Validation
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(x_values) < 3:
        raise ValueError(
            "At least three points are required to form a polygon.")

    # Convert to NumPy arrays for efficient computation
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    points = np.column_stack((x_values, y_values))
    num_points = len(points)

    angles = []
    edge_lengths = []

    for i in range(num_points):
        # Get current point and its neighboring points
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[(i + 1) % num_points]

        # Compute vectors for edges
        v1 = p0 - p1  # Edge from p1 to p0
        v2 = p2 - p1  # Edge from p1 to p2

        # Compute magnitudes of the edges
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        # Skip calculation if either edge has zero length
        if len_v1 == 0 or len_v2 == 0:
            continue

        # Compute cosine of the angle between edges
        cos_theta = np.dot(v1, v2) / (len_v1 * len_v2)
        # Ensure value is within valid range
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Compute angle in degrees
        angle = np.degrees(np.arccos(cos_theta))

        # Compute weight as sum of adjacent edge lengths
        weight = len_v1 + len_v2

        # Include angle and weight in the lists
        angles.append(angle)
        edge_lengths.append(weight)

    # Compute total edge length
    total_edge_length = sum(edge_lengths)

    # Handle case where total_edge_length is zero
    if total_edge_length == 0:
        raise ValueError(
            "Total edge length calculated as zero.\
                Check input coordinates for degeneracy.")

    # Compute weighted average angle
    weighted_avg_angle = sum(
        angle * weight for angle, weight in zip(angles, edge_lengths)
    ) / total_edge_length

    roundness = weighted_avg_angle / 180

    return weighted_avg_angle, roundness
