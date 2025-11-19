#!/usr/bin/env python3
"""
Generate annotated H&E slides showing actual segmentation and features
This is the STANDARD visualization for Phase 1 results
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hovernet_pipeline.segmentation import segment_nuclei
from hovernet_pipeline.features import extract_nuclei_features
from hovernet_pipeline.visualization import (
    annotate_he_slide,
    create_annotated_panel,
    create_detailed_nucleus_view,
    save_annotated_slide
)


def create_realistic_he_tile(size=512, n_nuclei=25):
    """Create a more realistic synthetic H&E tile"""
    # Tissue background (light pink/purple)
    tile = np.ones((size, size, 3), dtype=np.uint8)
    tile[:, :, 0] = 245  # R
    tile[:, :, 1] = 220  # G
    tile[:, :, 2] = 235  # B

    # Add some texture/variation
    noise = np.random.normal(0, 5, (size, size, 3))
    tile = np.clip(tile + noise, 0, 255).astype(np.uint8)

    np.random.seed(42)

    nuclei_info = []

    for i in range(n_nuclei):
        # Random position (avoid edges)
        cx = np.random.randint(60, size - 60)
        cy = np.random.randint(60, size - 60)

        # Random size (8-25 pixels radius for realistic nuclei)
        radius = np.random.randint(8, 25)

        # Nucleus color (dark purple/blue - hematoxylin stain)
        # Different shades for different cell types
        cell_type = np.random.choice([1, 2, 3, 4])

        if cell_type == 1:  # Lymphocyte - small, dark, round
            radius = np.random.randint(6, 12)
            base_color = (120, 100, 180)
        elif cell_type == 2:  # Tumor - larger, irregular
            radius = np.random.randint(12, 22)
            base_color = (140, 120, 200)
        elif cell_type == 3:  # Stromal - elongated
            radius = np.random.randint(10, 18)
            base_color = (160, 140, 210)
        else:  # Necrotic - pale, fragmented
            radius = np.random.randint(15, 25)
            base_color = (180, 160, 200)

        # Add color variation
        color = tuple(int(c + np.random.randint(-15, 15)) for c in base_color)

        # Draw main nucleus
        cv2.circle(tile, (cx, cy), radius, color, -1)

        # Add chromatin texture (darker spots inside nucleus)
        for _ in range(np.random.randint(2, 5)):
            offset_x = np.random.randint(-radius//2, radius//2)
            offset_y = np.random.randint(-radius//2, radius//2)
            spot_radius = np.random.randint(2, 5)
            darker_color = tuple(max(0, c - 30) for c in color)
            cv2.circle(tile, (cx + offset_x, cy + offset_y),
                      spot_radius, darker_color, -1)

        # Make some nuclei irregular (especially tumor/necrotic)
        if cell_type in [2, 4] and np.random.random() > 0.5:
            for _ in range(3):
                offset_x = np.random.randint(-8, 8)
                offset_y = np.random.randint(-8, 8)
                bulge_radius = np.random.randint(4, 10)
                cv2.circle(tile, (cx + offset_x, cy + offset_y),
                          bulge_radius, color, -1)

        nuclei_info.append({
            'center': (cx, cy),
            'radius': radius,
            'type': cell_type
        })

    # Add some eosin stain (cytoplasm - pink regions)
    for _ in range(15):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        radius = np.random.randint(20, 40)
        pink_color = (255, 180, 200)
        overlay = tile.copy()
        cv2.circle(overlay, (cx, cy), radius, pink_color, -1)
        # Blend with alpha
        alpha = 0.3
        tile = cv2.addWeighted(tile, 1 - alpha, overlay, alpha, 0)

    return tile, nuclei_info


def main():
    """Generate standard annotated H&E slide visualizations"""
    print("=" * 80)
    print("GENERATING ANNOTATED H&E SLIDES - PHASE 1 STANDARD VISUALIZATION")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path('annotated_slides')
    output_dir.mkdir(exist_ok=True)

    # Generate realistic H&E tile
    print("1. Creating realistic synthetic H&E tile...")
    he_tile, nuclei_info = create_realistic_he_tile(size=512, n_nuclei=25)
    cv2.imwrite(str(output_dir / '01_original_he_slide.png'),
                cv2.cvtColor(he_tile, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Created 512×512 H&E tile with {len(nuclei_info)} nuclei")
    print(f"   ✅ Saved: {output_dir / '01_original_he_slide.png'}")
    print()

    # Run segmentation
    print("2. Running HoverNet segmentation...")
    seg_result = segment_nuclei(he_tile)
    n_detected = len(np.unique(seg_result['inst_map'])) - 1
    print(f"   ✅ Detected {n_detected} nuclei")
    print()

    # Extract features
    print("3. Extracting morphological features...")
    features = extract_nuclei_features(seg_result)
    print(f"   ✅ Extracted {len(features[0]) if features else 0} features per nucleus")
    print()

    # Generate visualizations
    print("4. Creating annotated slide visualizations...")
    print()

    # 4a. Boundaries only
    print("   4a. Nuclei boundaries overlay...")
    annotated_boundaries = annotate_he_slide(
        he_tile, seg_result['inst_map'], seg_result['type_map'], features,
        show_boundaries=True, show_centroids=False,
        show_types=False, show_features=False
    )
    cv2.imwrite(str(output_dir / '02_boundaries_only.png'),
                cv2.cvtColor(annotated_boundaries, cv2.COLOR_RGB2BGR))
    print(f"       ✅ Saved: {output_dir / '02_boundaries_only.png'}")

    # 4b. Type classification
    print("   4b. Cell type classification overlay...")
    annotated_types = annotate_he_slide(
        he_tile, seg_result['inst_map'], seg_result['type_map'], features,
        show_boundaries=True, show_centroids=True,
        show_types=True, show_features=False
    )
    cv2.imwrite(str(output_dir / '03_cell_types.png'),
                cv2.cvtColor(annotated_types, cv2.COLOR_RGB2BGR))
    print(f"       ✅ Saved: {output_dir / '03_cell_types.png'}")

    # 4c. Full annotation with features
    print("   4c. Full annotation (type + area)...")
    annotated_full = annotate_he_slide(
        he_tile, seg_result['inst_map'], seg_result['type_map'], features,
        show_boundaries=True, show_centroids=True,
        show_types=True, show_features=True
    )
    cv2.imwrite(str(output_dir / '04_full_annotation.png'),
                cv2.cvtColor(annotated_full, cv2.COLOR_RGB2BGR))
    print(f"       ✅ Saved: {output_dir / '04_full_annotation.png'}")

    # 4d. Multi-panel comparison
    print("   4d. Multi-panel comparison view...")
    panel_view = create_annotated_panel(
        he_tile, seg_result, features,
        title="HoverNet Segmentation & Feature Extraction - Phase 1"
    )
    cv2.imwrite(str(output_dir / '05_panel_view.png'),
                cv2.cvtColor(panel_view, cv2.COLOR_RGB2BGR))
    print(f"       ✅ Saved: {output_dir / '05_panel_view.png'}")
    print()

    # 5. Detailed individual nucleus views
    print("5. Creating detailed nucleus views (first 5 nuclei)...")
    for i in range(min(5, len(features))):
        try:
            detail_view = create_detailed_nucleus_view(
                he_tile, seg_result, features, i, zoom_size=128
            )
            cv2.imwrite(str(output_dir / f'06_nucleus_{i+1:02d}_detail.png'),
                       cv2.cvtColor(detail_view, cv2.COLOR_RGB2BGR))
            print(f"   ✅ Nucleus #{i+1}: {output_dir / f'06_nucleus_{i+1:02d}_detail.png'}")
        except Exception as e:
            print(f"   ⚠️  Nucleus #{i+1}: Skipped ({e})")
    print()

    # 6. Create summary figure
    print("6. Creating summary statistics overlay...")

    # Count cell types
    type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for feat in features:
        type_counts[feat['type']] = type_counts.get(feat['type'], 0) + 1

    summary_img = he_tile.copy()

    # Add semi-transparent overlay for text
    overlay = summary_img.copy()
    cv2.rectangle(overlay, (10, 10), (250, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, summary_img, 0.4, 0, summary_img)

    # Add text
    y_pos = 35
    cv2.putText(summary_img, "SEGMENTATION SUMMARY", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 25
    cv2.putText(summary_img, f"Total Nuclei: {len(features)}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 25
    cv2.putText(summary_img, "Cell Types:", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 20
    cv2.putText(summary_img, f"  Lymphocyte: {type_counts.get(1, 0)}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1, cv2.LINE_AA)
    y_pos += 20
    cv2.putText(summary_img, f"  Tumor: {type_counts.get(2, 0)}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
    y_pos += 20
    cv2.putText(summary_img, f"  Stromal: {type_counts.get(3, 0)}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1, cv2.LINE_AA)
    y_pos += 20
    cv2.putText(summary_img, f"  Necrotic: {type_counts.get(4, 0)}", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_dir / '07_summary_overlay.png'),
                cv2.cvtColor(summary_img, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Saved: {output_dir / '07_summary_overlay.png'}")
    print()

    # Summary
    print("=" * 80)
    print("ANNOTATED SLIDES GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 Output directory: {output_dir}/")
    print()
    print("📊 Generated files:")
    print("   1. 01_original_he_slide.png       - Original H&E tissue image")
    print("   2. 02_boundaries_only.png         - Nuclei boundaries overlay")
    print("   3. 03_cell_types.png              - Cell type classification")
    print("   4. 04_full_annotation.png         - Full annotation (types + areas)")
    print("   5. 05_panel_view.png              - Multi-panel comparison (4 views)")
    print("   6. 06_nucleus_XX_detail.png       - Detailed individual nucleus views")
    print("   7. 07_summary_overlay.png         - Summary statistics overlay")
    print()
    print(f"✅ Successfully annotated {len(features)} nuclei on H&E slide")
    print(f"✅ Cell type breakdown:")
    print(f"     • Lymphocytes: {type_counts.get(1, 0)}")
    print(f"     • Tumor cells: {type_counts.get(2, 0)}")
    print(f"     • Stromal cells: {type_counts.get(3, 0)}")
    print(f"     • Necrotic cells: {type_counts.get(4, 0)}")
    print()
    print("View annotated slides with:")
    print(f"  open {output_dir}/05_panel_view.png")
    print(f"  open {output_dir}/04_full_annotation.png")
    print()


if __name__ == '__main__':
    main()
