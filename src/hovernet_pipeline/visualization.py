"""
Visualization tools for HoverNet segmentation results
Annotates H&E slides with nuclei boundaries, features, and classifications
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple


def annotate_he_slide(image: np.ndarray,
                       inst_map: np.ndarray,
                       type_map: np.ndarray,
                       features: List[Dict],
                       show_boundaries: bool = True,
                       show_centroids: bool = True,
                       show_types: bool = True,
                       show_features: bool = False,
                       nucleus_id: Optional[int] = None) -> np.ndarray:
    """
    Annotate H&E slide with segmentation results and features

    Args:
        image: Original H&E image (H, W, 3)
        inst_map: Instance segmentation map (H, W)
        type_map: Cell type map (H, W)
        features: List of nucleus feature dictionaries
        show_boundaries: Draw nucleus boundaries
        show_centroids: Draw centroid markers
        show_types: Show cell type labels
        show_features: Show feature values
        nucleus_id: If specified, highlight only this nucleus

    Returns:
        Annotated RGB image
    """
    # Create annotated image (copy to avoid modifying original)
    annotated = image.copy()

    # Color scheme for cell types
    type_colors = {
        1: (255, 0, 0),      # Lymphocyte - Red
        2: (0, 255, 0),      # Tumor - Green
        3: (0, 0, 255),      # Stromal - Blue
        4: (255, 255, 0)     # Necrotic - Yellow
    }

    type_names = {
        1: 'Lymph',
        2: 'Tumor',
        3: 'Stroma',
        4: 'Necro'
    }

    # Get unique nucleus IDs
    nucleus_ids = np.unique(inst_map)[1:]  # Skip background

    # If specific nucleus requested, filter
    if nucleus_id is not None:
        nucleus_ids = [nucleus_id] if nucleus_id in nucleus_ids else []

    for idx, nuc_id in enumerate(nucleus_ids):
        # Get mask for this nucleus
        mask = (inst_map == nuc_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = contours[0]

        # Get nucleus type from type_map
        type_pixels = type_map[mask > 0]
        nucleus_type = int(type_pixels[0]) if len(type_pixels) > 0 and type_pixels[0] != 0 else 1

        # Get color for this type
        color = type_colors.get(nucleus_type, (255, 255, 255))

        # Draw boundary
        if show_boundaries:
            cv2.drawContours(annotated, [contour], -1, color, 2)

        # Get centroid from features
        if idx < len(features):
            feat = features[idx]
            cx, cy = feat['centroid']

            # Draw centroid
            if show_centroids:
                cv2.circle(annotated, (cx, cy), 3, color, -1)
                cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), 1)

            # Add type label
            if show_types:
                label = f"{type_names[nucleus_type]}"
                # Position label slightly above centroid
                label_pos = (cx - 20, cy - 10)

                # Add background for text
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated,
                            (label_pos[0] - 2, label_pos[1] - text_h - 2),
                            (label_pos[0] + text_w + 2, label_pos[1] + 2),
                            (0, 0, 0), -1)

                cv2.putText(annotated, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Add feature annotations
            if show_features:
                feature_text = f"A:{feat['area']:.0f}"
                feature_pos = (cx - 20, cy + 15)

                # Background for feature text
                (text_w, text_h), _ = cv2.getTextSize(feature_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(annotated,
                            (feature_pos[0] - 2, feature_pos[1] - text_h - 2),
                            (feature_pos[0] + text_w + 2, feature_pos[1] + 2),
                            (0, 0, 0), -1)

                cv2.putText(annotated, feature_text, feature_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def create_annotated_panel(image: np.ndarray,
                           segmentation_result: Dict[str, np.ndarray],
                           features: List[Dict],
                           title: str = "H&E Slide with Annotations") -> np.ndarray:
    """
    Create a multi-panel visualization with different annotation levels

    Args:
        image: Original H&E image
        segmentation_result: Dict with 'inst_map' and 'type_map'
        features: List of nucleus features
        title: Figure title

    Returns:
        Combined visualization as image array
    """
    inst_map = segmentation_result['inst_map']
    type_map = segmentation_result['type_map']

    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Panel 1: Original H&E
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    axes[0, 0].set_title('Original H&E Slide', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Panel 2: With boundaries only
    annotated_boundaries = annotate_he_slide(
        image, inst_map, type_map, features,
        show_boundaries=True, show_centroids=False,
        show_types=False, show_features=False
    )
    axes[0, 1].imshow(cv2.cvtColor(annotated_boundaries, cv2.COLOR_RGB2BGR))
    axes[0, 1].set_title('Nuclei Boundaries', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Panel 3: With type labels
    annotated_types = annotate_he_slide(
        image, inst_map, type_map, features,
        show_boundaries=True, show_centroids=True,
        show_types=True, show_features=False
    )
    axes[1, 0].imshow(cv2.cvtColor(annotated_types, cv2.COLOR_RGB2BGR))
    axes[1, 0].set_title('Cell Type Classification', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Add legend
    type_colors_mpl = {
        'Lymphocyte': [1, 0, 0],
        'Tumor': [0, 1, 0],
        'Stromal': [0, 0, 1],
        'Necrotic': [1, 1, 0]
    }
    patches = [mpatches.Patch(color=color, label=name)
               for name, color in type_colors_mpl.items()]
    axes[1, 0].legend(handles=patches, loc='upper right', fontsize=10)

    # Panel 4: Full annotation with features
    annotated_full = annotate_he_slide(
        image, inst_map, type_map, features,
        show_boundaries=True, show_centroids=True,
        show_types=True, show_features=True
    )
    axes[1, 1].imshow(cv2.cvtColor(annotated_full, cv2.COLOR_RGB2BGR))
    axes[1, 1].set_title('Full Annotation (Type + Area)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save to temporary file and read back (more reliable than buffer)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    plt.savefig(tmp_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Read back the image
    img_array = cv2.imread(tmp_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Clean up
    os.unlink(tmp_path)

    return img_array


def create_detailed_nucleus_view(image: np.ndarray,
                                 segmentation_result: Dict[str, np.ndarray],
                                 features: List[Dict],
                                 nucleus_idx: int,
                                 zoom_size: int = 128) -> np.ndarray:
    """
    Create detailed view of a single nucleus with all features annotated

    Args:
        image: Original H&E image
        segmentation_result: Segmentation results
        features: Feature list
        nucleus_idx: Index of nucleus to visualize
        zoom_size: Size of zoomed region

    Returns:
        Detailed nucleus visualization
    """
    if nucleus_idx >= len(features):
        raise ValueError(f"Nucleus index {nucleus_idx} out of range (max {len(features)-1})")

    feat = features[nucleus_idx]
    cx, cy = feat['centroid']

    # Extract region around nucleus
    half_size = zoom_size // 2
    y_min = max(0, cy - half_size)
    y_max = min(image.shape[0], cy + half_size)
    x_min = max(0, cx - half_size)
    x_max = min(image.shape[1], cx + half_size)

    # Crop regions
    image_crop = image[y_min:y_max, x_min:x_max].copy()
    inst_crop = segmentation_result['inst_map'][y_min:y_max, x_min:x_max]
    type_crop = segmentation_result['type_map'][y_min:y_max, x_min:x_max]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Annotated crop
    # Adjust centroid to crop coordinates
    features_adjusted = [feat.copy()]
    features_adjusted[0]['centroid'] = (cx - x_min, cy - y_min)

    annotated_crop = annotate_he_slide(
        image_crop, inst_crop, type_crop, features_adjusted,
        show_boundaries=True, show_centroids=True,
        show_types=False, show_features=False
    )
    axes[0].imshow(cv2.cvtColor(annotated_crop, cv2.COLOR_RGB2BGR))
    axes[0].set_title(f'Nucleus #{nucleus_idx + 1}', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Right: Feature table
    type_names = {1: 'Lymphocyte', 2: 'Tumor', 3: 'Stromal', 4: 'Necrotic'}

    feature_text = f"""
NUCLEUS FEATURES
{'='*40}

Identification:
  • Nucleus ID: {nucleus_idx + 1}
  • Cell Type: {type_names.get(feat['type'], 'Unknown')}
  • Position: ({cx}, {cy})

Morphology:
  • Area: {feat['area']:.1f} pixels
  • Perimeter: {feat['perimeter']:.1f} pixels
  • Circularity: {feat['circularity']:.3f}
  • Equivalent Diameter: {feat['equivalent_diameter']:.1f}

Shape Features:
  • Eccentricity: {feat['eccentricity']:.3f}
  • Solidity: {feat['solidity']:.3f}
  • Aspect Ratio: {feat['aspect_ratio']:.3f}
  • Extent: {feat['extent']:.3f}
  • Compactness: {feat['compactness']:.2f}

Ellipse Fit:
  • Major Axis: {feat['major_axis_length']:.1f}
  • Minor Axis: {feat['minor_axis_length']:.1f}
  • Orientation: {feat['orientation']:.1f}°

Convex Hull:
  • Convex Area: {feat['convex_area']:.1f}
    """

    axes[1].text(0.05, 0.95, feature_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1].axis('off')

    plt.tight_layout()

    # Save to temporary file and read back (more reliable than buffer)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    plt.savefig(tmp_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Read back the image
    img_array = cv2.imread(tmp_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Clean up
    os.unlink(tmp_path)

    return img_array


def save_annotated_slide(image: np.ndarray,
                        segmentation_result: Dict[str, np.ndarray],
                        features: List[Dict],
                        output_path: str,
                        annotation_level: str = 'full'):
    """
    Save annotated H&E slide to file

    Args:
        image: Original H&E image
        segmentation_result: Segmentation results
        features: Feature list
        output_path: Output file path
        annotation_level: 'boundaries', 'types', 'full', or 'panel'
    """
    inst_map = segmentation_result['inst_map']
    type_map = segmentation_result['type_map']

    if annotation_level == 'panel':
        # Create multi-panel view
        annotated = create_annotated_panel(image, segmentation_result, features)
        cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    else:
        # Create single annotated image
        show_boundaries = annotation_level in ['boundaries', 'types', 'full']
        show_centroids = annotation_level in ['types', 'full']
        show_types = annotation_level in ['types', 'full']
        show_features = annotation_level == 'full'

        annotated = annotate_he_slide(
            image, inst_map, type_map, features,
            show_boundaries=show_boundaries,
            show_centroids=show_centroids,
            show_types=show_types,
            show_features=show_features
        )
        cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved annotated slide: {output_path}")
