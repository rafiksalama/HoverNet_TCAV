#!/usr/bin/env python3
"""
Generate visual evidence for Phase 1 HoverNet implementation
Creates images and data outputs showing actual segmentation and features
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import pandas as pd

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hovernet_pipeline.segmentation import segment_nuclei
from hovernet_pipeline.features import extract_nuclei_features, compute_slide_level_features
from data_processing.stain_normalization import MacenkoNormalizer


def create_synthetic_he_tile(size=512):
    """Create a synthetic H&E tile with nuclei-like structures"""
    # Create tissue background (light pink)
    tile = np.ones((size, size, 3), dtype=np.uint8) * 240
    tile[:, :, 0] = 250  # R
    tile[:, :, 1] = 220  # G
    tile[:, :, 2] = 235  # B

    # Add some nuclei (dark purple/blue spots)
    np.random.seed(42)
    n_nuclei = 30

    for i in range(n_nuclei):
        # Random position
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)

        # Random size (10-30 pixels radius)
        radius = np.random.randint(8, 20)

        # Draw nucleus (dark purple)
        color = (180 + np.random.randint(-20, 20),  # R
                 140 + np.random.randint(-20, 20),  # G
                 200 + np.random.randint(-20, 20))  # B

        cv2.circle(tile, (cx, cy), radius, color, -1)

        # Add some irregularity
        for _ in range(3):
            offset_x = np.random.randint(-5, 5)
            offset_y = np.random.randint(-5, 5)
            small_radius = np.random.randint(3, 8)
            cv2.circle(tile, (cx + offset_x, cy + offset_y), small_radius, color, -1)

    return tile


def visualize_segmentation(image, result, output_dir):
    """Create segmentation visualization"""
    inst_map = result['inst_map']
    type_map = result['type_map']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    axes[0].set_title('Original H&E Tile', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Instance map (each nucleus different color)
    inst_colored = np.zeros((inst_map.shape[0], inst_map.shape[1], 3), dtype=np.uint8)
    unique_ids = np.unique(inst_map)[1:]  # Skip background
    for nuc_id in unique_ids:
        color = plt.cm.jet(nuc_id / len(unique_ids))[:3]
        color = (np.array(color) * 255).astype(np.uint8)
        inst_colored[inst_map == nuc_id] = color

    axes[1].imshow(inst_colored)
    axes[1].set_title(f'Instance Map ({len(unique_ids)} nuclei)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Type map (color by cell type)
    type_colors = {
        0: [0, 0, 0],          # Background - black
        1: [255, 0, 0],        # Lymphocyte - red
        2: [0, 255, 0],        # Tumor - green
        3: [0, 0, 255],        # Stromal - blue
        4: [255, 255, 0]       # Necrotic - yellow
    }

    type_colored = np.zeros((type_map.shape[0], type_map.shape[1], 3), dtype=np.uint8)
    for type_id, color in type_colors.items():
        type_colored[type_map == type_id] = color

    axes[2].imshow(type_colored)
    axes[2].set_title('Classification Map', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add legend for classification
    patches = [
        mpatches.Patch(color=[1, 0, 0], label='Lymphocyte'),
        mpatches.Patch(color=[0, 1, 0], label='Tumor'),
        mpatches.Patch(color=[0, 0, 1], label='Stromal'),
        mpatches.Patch(color=[1, 1, 0], label='Necrotic')
    ]
    axes[2].legend(handles=patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'segmentation_results.png'}")
    plt.close()


def visualize_features(features_df, output_dir):
    """Create feature distribution plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Area distribution
    axes[0, 0].hist(features_df['area'], bins=20, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Nuclear Area (pixels)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Nuclear Area Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)

    # Circularity distribution
    axes[0, 1].hist(features_df['circularity'], bins=20, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Circularity', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Circularity Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Type distribution
    type_counts = features_df['type'].value_counts().sort_index()
    type_labels = {1: 'Lymphocyte', 2: 'Tumor', 3: 'Stromal', 4: 'Necrotic'}
    colors = ['red', 'green', 'blue', 'yellow']

    bars = axes[0, 2].bar(range(len(type_counts)), type_counts.values,
                          color=colors[:len(type_counts)], edgecolor='black')
    axes[0, 2].set_xticks(range(len(type_counts)))
    axes[0, 2].set_xticklabels([type_labels.get(t, f'Type {t}') for t in type_counts.index],
                                rotation=45, ha='right')
    axes[0, 2].set_ylabel('Count', fontsize=11)
    axes[0, 2].set_title('Cell Type Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].grid(alpha=0.3, axis='y')

    # Eccentricity vs Area scatter
    axes[1, 0].scatter(features_df['area'], features_df['eccentricity'],
                       alpha=0.6, c='purple', edgecolor='black', s=50)
    axes[1, 0].set_xlabel('Area (pixels)', fontsize=11)
    axes[1, 0].set_ylabel('Eccentricity', fontsize=11)
    axes[1, 0].set_title('Area vs Eccentricity', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Solidity distribution
    axes[1, 1].hist(features_df['solidity'], bins=20, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Solidity', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Solidity Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # Compactness distribution
    axes[1, 2].hist(features_df['compactness'], bins=20, color='teal', edgecolor='black')
    axes[1, 2].set_xlabel('Compactness', fontsize=11)
    axes[1, 2].set_ylabel('Count', fontsize=11)
    axes[1, 2].set_title('Compactness Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'feature_distributions.png'}")
    plt.close()


def visualize_slide_metrics(slide_metrics, output_dir):
    """Create slide-level metrics visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cell type densities (pie chart)
    densities = {
        'Lymphocyte': slide_metrics['lymphocyte_density'],
        'Tumor': slide_metrics['tumor_cell_density'],
        'Stromal': slide_metrics['stromal_density'],
        'Necrotic': slide_metrics['necrotic_density']
    }

    colors = ['red', 'green', 'blue', 'yellow']
    axes[0].pie(densities.values(), labels=densities.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 11})
    axes[0].set_title('Cell Type Composition', fontsize=13, fontweight='bold')

    # Morphology metrics (bar chart)
    morph_metrics = {
        'Mean Area': slide_metrics['mean_nuclear_area'],
        'Std Area': slide_metrics['std_nuclear_area'],
        'Mean Circ.': slide_metrics['mean_circularity'] * 100  # Scale for visibility
    }

    bars = axes[1].bar(range(len(morph_metrics)), morph_metrics.values(),
                       color=['steelblue', 'coral', 'purple'], edgecolor='black')
    axes[1].set_xticks(range(len(morph_metrics)))
    axes[1].set_xticklabels(morph_metrics.keys(), fontsize=11)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Morphology Summary', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'slide_metrics.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'slide_metrics.png'}")
    plt.close()


def main():
    """Generate all visual evidence"""
    print("=" * 70)
    print("GENERATING VISUAL EVIDENCE FOR PHASE 1")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path('evidence_outputs')
    output_dir.mkdir(exist_ok=True)

    # 1. Generate synthetic H&E tile
    print("1. Generating synthetic H&E tile...")
    he_tile = create_synthetic_he_tile(size=512)
    cv2.imwrite(str(output_dir / 'original_he_tile.png'),
                cv2.cvtColor(he_tile, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Created 512×512 H&E tile with ~30 nuclei")
    print(f"   ✅ Saved: {output_dir / 'original_he_tile.png'}")
    print()

    # 2. Run segmentation
    print("2. Running HoverNet segmentation...")
    seg_result = segment_nuclei(he_tile)
    n_nuclei = len(np.unique(seg_result['inst_map'])) - 1  # Exclude background
    print(f"   ✅ Detected {n_nuclei} nuclei")
    print(f"   ✅ Instance map shape: {seg_result['inst_map'].shape}")
    print(f"   ✅ Type map shape: {seg_result['type_map'].shape}")
    print()

    # 3. Extract features
    print("3. Extracting morphological features...")
    features = extract_nuclei_features(seg_result)
    print(f"   ✅ Extracted features for {len(features)} nuclei")
    print(f"   ✅ Features per nucleus: {len(features[0]) if features else 0}")
    print()

    # 4. Compute slide-level metrics
    print("4. Computing slide-level metrics...")
    slide_metrics = compute_slide_level_features(features)
    print(f"   ✅ Computed {len(slide_metrics)} slide-level metrics")
    print()

    # 5. Create visualizations
    print("5. Creating visualizations...")
    visualize_segmentation(he_tile, seg_result, output_dir)

    # Convert features to DataFrame for visualization
    features_df = pd.DataFrame(features)
    visualize_features(features_df, output_dir)
    visualize_slide_metrics(slide_metrics, output_dir)
    print()

    # 6. Save data files
    print("6. Saving data outputs...")

    # Save features as CSV
    features_df.to_csv(output_dir / 'nuclei_features.csv', index=False)
    print(f"   ✅ Saved: {output_dir / 'nuclei_features.csv'}")

    # Save slide metrics as JSON
    with open(output_dir / 'slide_metrics.json', 'w') as f:
        json.dump(slide_metrics, f, indent=2)
    print(f"   ✅ Saved: {output_dir / 'slide_metrics.json'}")

    # Save feature summary
    with open(output_dir / 'feature_summary.txt', 'w') as f:
        f.write("PHASE 1 FEATURE EXTRACTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total nuclei detected: {n_nuclei}\n")
        f.write(f"Features per nucleus: {len(features[0]) if features else 0}\n\n")
        f.write("SLIDE-LEVEL METRICS:\n")
        f.write("-" * 70 + "\n")
        for key, value in slide_metrics.items():
            f.write(f"{key:25s}: {value:10.4f}\n")
        f.write("\n")
        f.write("FEATURE STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(features_df.describe().to_string())
        f.write("\n\n")
        f.write("FIRST 5 NUCLEI FEATURES:\n")
        f.write("-" * 70 + "\n")
        f.write(features_df.head().to_string())
    print(f"   ✅ Saved: {output_dir / 'feature_summary.txt'}")
    print()

    # 7. Print summary
    print("=" * 70)
    print("EVIDENCE GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print("📁 Output directory: evidence_outputs/")
    print()
    print("📊 Generated files:")
    print("   • original_he_tile.png          - Input H&E image")
    print("   • segmentation_results.png      - Instance & type maps")
    print("   • feature_distributions.png     - Feature histograms")
    print("   • slide_metrics.png             - Slide-level metrics")
    print("   • nuclei_features.csv           - All feature data")
    print("   • slide_metrics.json            - Aggregated metrics")
    print("   • feature_summary.txt           - Text summary")
    print()
    print(f"✅ Successfully segmented {n_nuclei} nuclei")
    print(f"✅ Extracted {len(features[0]) if features else 0} features per nucleus")
    print(f"✅ Computed {len(slide_metrics)} slide-level metrics")
    print()
    print("View images with:")
    print("  open evidence_outputs/segmentation_results.png")
    print("  open evidence_outputs/feature_distributions.png")
    print("  open evidence_outputs/slide_metrics.png")
    print()


if __name__ == '__main__':
    main()
