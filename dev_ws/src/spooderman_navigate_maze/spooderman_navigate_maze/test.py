import numpy as np
import math

def compute_angle_difference(q1, q2):

    # # TODO: FIX ANGLE DIFFERENCE CALCULATION!!!
    
    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute the dot product
    dot_product = np.dot(q1, q2)
    
    # Clamp the dot product to the range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Compute the angle in radians
    angle_radians = 2 * np.arccos(np.abs(dot_product))
    
    # Convert the angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_radians, angle_degrees

q1 = np.array([0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2])
q2 = np.array([0, 0, 0, 1])

print(compute_angle_difference(q1, q2))