#!/bin/bash
# Download NSCLC datasets for HoverNet + TCAV prototype
# Datasets: TCGA-LUAD, TCGA-LUSC (via GDC Data Transfer Tool)

set -e  # Exit on error

echo "================================================================================"
echo "NSCLC DATASET DOWNLOAD SCRIPT"
echo "================================================================================"
echo ""

# Configuration
SAMPLE_SIZE=50  # Number of cases to download per subtype (for prototyping)
DATA_DIR="./data"
GDC_CLIENT="./gdc-client"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Creating data directory: $DATA_DIR${NC}"
    mkdir -p "$DATA_DIR"
fi

# Step 1: Download GDC Data Transfer Tool
echo ""
echo "Step 1: Installing GDC Data Transfer Tool"
echo "--------------------------------------------------------------------------------"

if [ ! -f "$GDC_CLIENT" ]; then
    echo -e "${YELLOW}Downloading GDC Data Transfer Tool...${NC}"

    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_OSX_x64.zip
        unzip gdc-client_v1.6.1_OSX_x64.zip
        mv gdc-client ./gdc-client
        chmod +x ./gdc-client
        rm gdc-client_v1.6.1_OSX_x64.zip
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
        unzip gdc-client_v1.6.1_Ubuntu_x64.zip
        mv gdc-client ./gdc-client
        chmod +x ./gdc-client
        rm gdc-client_v1.6.1_Ubuntu_x64.zip
    else
        echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
        echo "Please download GDC client manually from:"
        echo "https://gdc.cancer.gov/access-data/gdc-data-transfer-tool"
        exit 1
    fi

    echo -e "${GREEN}✅ GDC Data Transfer Tool installed${NC}"
else
    echo -e "${GREEN}✅ GDC Data Transfer Tool already installed${NC}"
fi

# Verify installation
echo ""
echo "GDC Client Version:"
./gdc-client --version

# Step 2: Generate manifests (user must do this manually via GDC portal)
echo ""
echo "Step 2: Manifest Files"
echo "--------------------------------------------------------------------------------"
echo -e "${YELLOW}⚠️  MANUAL STEP REQUIRED${NC}"
echo ""
echo "You need to download manifest files from the GDC Data Portal:"
echo ""
echo "For TCGA-LUAD (Lung Adenocarcinoma):"
echo "  1. Go to: https://portal.gdc.cancer.gov/projects/TCGA-LUAD"
echo "  2. Click on the number under 'Diagnostic Slide'"
echo "  3. Click 'Manifest' button to download"
echo "  4. Save as: manifests/tcga_luad_manifest.txt"
echo ""
echo "For TCGA-LUSC (Lung Squamous Cell Carcinoma):"
echo "  1. Go to: https://portal.gdc.cancer.gov/projects/TCGA-LUSC"
echo "  2. Click on the number under 'Diagnostic Slide'"
echo "  3. Click 'Manifest' button to download"
echo "  4. Save as: manifests/tcga_lusc_manifest.txt"
echo ""

# Create manifests directory
mkdir -p manifests

# Check if manifests exist
LUAD_MANIFEST="manifests/tcga_luad_manifest.txt"
LUSC_MANIFEST="manifests/tcga_lusc_manifest.txt"

echo "Checking for manifest files..."
echo ""

if [ -f "$LUAD_MANIFEST" ]; then
    echo -e "${GREEN}✅ Found LUAD manifest: $LUAD_MANIFEST${NC}"
    LUAD_COUNT=$(wc -l < "$LUAD_MANIFEST")
    echo "   Files in manifest: $LUAD_COUNT"
else
    echo -e "${RED}❌ LUAD manifest not found: $LUAD_MANIFEST${NC}"
    echo "   Please download from GDC portal (see instructions above)"
fi

echo ""

if [ -f "$LUSC_MANIFEST" ]; then
    echo -e "${GREEN}✅ Found LUSC manifest: $LUSC_MANIFEST${NC}"
    LUSC_COUNT=$(wc -l < "$LUSC_MANIFEST")
    echo "   Files in manifest: $LUSC_COUNT"
else
    echo -e "${RED}❌ LUSC manifest not found: $LUSC_MANIFEST${NC}"
    echo "   Please download from GDC portal (see instructions above)"
fi

echo ""

# Ask user if they want to proceed
if [ ! -f "$LUAD_MANIFEST" ] && [ ! -f "$LUSC_MANIFEST" ]; then
    echo -e "${RED}No manifest files found. Cannot proceed with download.${NC}"
    echo "Please download manifests and run this script again."
    exit 1
fi

echo "Do you want to proceed with downloading slides? (y/n)"
read -r PROCEED

if [[ ! "$PROCEED" =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Step 3: Download TCGA-LUAD slides
if [ -f "$LUAD_MANIFEST" ]; then
    echo ""
    echo "Step 3: Downloading TCGA-LUAD slides"
    echo "--------------------------------------------------------------------------------"
    echo "Downloading to: $DATA_DIR/TCGA-LUAD/"
    echo "This may take several hours depending on your connection..."
    echo ""

    mkdir -p "$DATA_DIR/TCGA-LUAD"

    ./gdc-client download \
        -m "$LUAD_MANIFEST" \
        --dir "$DATA_DIR/TCGA-LUAD/" \
        --n-processes 4 \
        --log-file logs/tcga_luad_download.log

    echo -e "${GREEN}✅ TCGA-LUAD download complete${NC}"
fi

# Step 4: Download TCGA-LUSC slides
if [ -f "$LUSC_MANIFEST" ]; then
    echo ""
    echo "Step 4: Downloading TCGA-LUSC slides"
    echo "--------------------------------------------------------------------------------"
    echo "Downloading to: $DATA_DIR/TCGA-LUSC/"
    echo "This may take several hours depending on your connection..."
    echo ""

    mkdir -p "$DATA_DIR/TCGA-LUSC"

    ./gdc-client download \
        -m "$LUSC_MANIFEST" \
        --dir "$DATA_DIR/TCGA-LUSC/" \
        --n-processes 4 \
        --log-file logs/tcga_lusc_download.log

    echo -e "${GREEN}✅ TCGA-LUSC download complete${NC}"
fi

# Step 5: Verify downloads
echo ""
echo "Step 5: Verifying downloads"
echo "--------------------------------------------------------------------------------"

if [ -d "$DATA_DIR/TCGA-LUAD" ]; then
    LUAD_FILES=$(find "$DATA_DIR/TCGA-LUAD" -name "*.svs" | wc -l)
    echo -e "${GREEN}TCGA-LUAD: $LUAD_FILES .svs files downloaded${NC}"
fi

if [ -d "$DATA_DIR/TCGA-LUSC" ]; then
    LUSC_FILES=$(find "$DATA_DIR/TCGA-LUSC" -name "*.svs" | wc -l)
    echo -e "${GREEN}TCGA-LUSC: $LUSC_FILES .svs files downloaded${NC}"
fi

echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Run tile extraction: python scripts/extract_tiles.py"
echo "  2. Run HoverNet segmentation: python scripts/run_hovernet.py"
echo "  3. Extract features: python scripts/extract_features.py"
echo ""
