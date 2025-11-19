"""
Feature extraction from HoverNet segmentation results
"""

import numpy as np
import cv2
from typing import List, Dict
import pandas as pd


def extract_nuclei_features(hovernet_output: Dict[str, np.ndarray]) -> List[Dict]:
    """
    Extract morphological features from segmented nuclei

    Args:
        hovernet_output: Dictionary with 'inst_map' and 'type_map'

    Returns:
        List of feature dictionaries, one per nucleus

    Features extracted:
        - area: Nuclear area in pixels
        - perimeter: Nuclear perimeter
        - circularity: 4π×area / perimeter²
        - type: Cell type (1=inflammatory, 2=epithelial, 3=stromal, 4=necrotic)
        - centroid: (x, y) position
        - eccentricity: Ellipse eccentricity
        - solidity: Nucleus solidity (convex hull ratio)
    """
    inst_map = hovernet_output['inst_map']
    type_map = hovernet_output['type_map']

    features = []

    # Get unique nucleus IDs (excluding background=0)
    nucleus_ids = np.unique(inst_map)[1:]

    for nuc_id in nucleus_ids:
        # Create binary mask for this nucleus
        mask = (inst_map == nuc_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        contour = contours[0]

        # Compute morphological features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # Get nucleus type from type_map
        type_pixels = type_map[mask > 0]
        nucleus_type = int(type_pixels[0]) if len(type_pixels) > 0 and type_pixels[0] != 0 else 1

        # Eccentricity (requires ellipse fitting)
        if len(contour) >= 5:  # Need at least 5 points for ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                else:
                    eccentricity = 0
            except:
                eccentricity = 0
        else:
            eccentricity = 0

        # Solidity (convex hull)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0

        # Additional features
        # Bounding rectangle
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = float(w_box) / h_box if h_box > 0 else 0

        # Extent (area vs bounding box area)
        rect_area = w_box * h_box
        extent = float(area) / rect_area if rect_area > 0 else 0

        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # Compile features (15+ features as required)
        nucleus_features = {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'type': nucleus_type,
            'centroid': (cx, cy),
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'equivalent_diameter': equivalent_diameter,
            'major_axis_length': float(major_axis) if len(contour) >= 5 else 0,
            'minor_axis_length': float(minor_axis) if len(contour) >= 5 else 0,
            'orientation': float(ellipse[2]) if len(contour) >= 5 else 0,
            'convex_area': float(hull_area),
            'compactness': (perimeter ** 2) / area if area > 0 else 0
        }

        features.append(nucleus_features)

    return features


def compute_slide_level_features(nuclei_features: List[Dict]) -> Dict[str, float]:
    """
    Aggregate nuclei features to slide-level metrics

    Args:
        nuclei_features: List of nucleus feature dictionaries

    Returns:
        Dictionary of slide-level metrics:
            - lymphocyte_density: Fraction of inflammatory cells
            - tumor_cell_density: Fraction of epithelial cells
            - stromal_density: Fraction of stromal cells
            - necrotic_density: Fraction of necrotic cells
            - mean_nuclear_area: Average nucleus size
            - std_nuclear_area: Std dev of nucleus size
            - mean_circularity: Average circularity
    """
    if len(nuclei_features) == 0:
        return {
            'lymphocyte_density': 0.0,
            'tumor_cell_density': 0.0,
            'stromal_density': 0.0,
            'necrotic_density': 0.0,
            'mean_nuclear_area': 0.0,
            'std_nuclear_area': 0.0,
            'mean_circularity': 0.0
        }

    df = pd.DataFrame(nuclei_features)

    total_nuclei = len(df)

    # Type counts (1=inflammatory/lymphocyte, 2=epithelial/tumor, 3=stromal, 4=necrotic)
    type_counts = df['type'].value_counts()

    metrics = {
        'lymphocyte_density': type_counts.get(1, 0) / total_nuclei,
        'tumor_cell_density': type_counts.get(2, 0) / total_nuclei,
        'stromal_density': type_counts.get(3, 0) / total_nuclei,
        'necrotic_density': type_counts.get(4, 0) / total_nuclei,
        'mean_nuclear_area': df['area'].mean(),
        'std_nuclear_area': df['area'].std(),
        'mean_circularity': df['circularity'].mean()
    }

    return metrics
