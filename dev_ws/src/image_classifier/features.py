import cv2
import numpy as np

def compute_edge_features(image):
    """
    Compute edge features from the Canny edge-detected image.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        list: Edge features [edge_pixel_count, edge_density].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Compute features
    edge_pixel_count = np.sum(edges > 0)  # Count of non-zero pixels (edge pixels)
    total_pixel_count = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixel_count / total_pixel_count  # Fraction of edge pixels

    return [edge_pixel_count, edge_density]

def compute_shape_features(image):
    """
    Compute shape features for the largest contour in an image.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        list: Shape features [area, perimeter, aspect_ratio, extent, solidity].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Compute features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        bounding_box_area = w * h
        extent = area / float(bounding_box_area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area)

        return [area, perimeter, aspect_ratio, extent, solidity]
    else:
        # Default values if no contour is found
        return [0, 0, 0, 0, 0]
    
def compute_spatial_gradient_features(image, grid_size=(8, 8)):
    """
    Compute spatial gradient features by dividing the image into grid cells
    and extracting gradient magnitude and direction for each cell.

    Args:
        image (np.ndarray): Input grayscale image.
        grid_size (tuple): Number of cells along (rows, columns).

    Returns:
        np.ndarray: Flattened array of spatial gradient features.
    """
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients in the x and y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude and direction of gradients
    magnitude = cv2.magnitude(grad_x, grad_y)
    direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # Divide the image into a grid
    h, w = image.shape
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    features = []

    for i in range(grid_size[0]):  # Iterate over grid rows
        for j in range(grid_size[1]):  # Iterate over grid columns
            # Extract cell region
            cell_mag = magnitude[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_dir = direction[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

            # Compute aggregated features for the cell
            mean_mag = np.mean(cell_mag)
            std_mag = np.std(cell_mag)
            mean_dir = np.mean(cell_dir)
            std_dir = np.std(cell_dir)

            # Append cell features (mean and std of magnitude and direction)
            features.extend([mean_mag, std_mag, mean_dir, std_dir])

    return np.array(features)