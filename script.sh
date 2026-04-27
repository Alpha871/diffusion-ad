#!/bin/bash
# Reorganize VisA dataset from original structure to DiffusionAD expected structure
# Original: Data/Images/Normal, Data/Images/Anomaly, Data/Masks/Anomaly, DISthresh/good
# Target:   train/good, test/good, test/anomaly, ground_truth/anomaly

VISA_ROOT="/home/jovyan/work/DiffusionAD/datasets/visa"


for cls in candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
    echo "=== Processing $cls ==="
    CLS_DIR="$VISA_ROOT/$cls"

    # Clean old train/good if exists
    rm -f "$CLS_DIR/train/good/"* 2>/dev/null

    # Create target directories
    mkdir -p "$CLS_DIR/train/good"
    mkdir -p "$CLS_DIR/test/good"
    mkdir -p "$CLS_DIR/test/anomaly"
    mkdir -p "$CLS_DIR/ground_truth/anomaly"

    # 1. test/good ← copy DISthresh/good
    echo "  Copying DISthresh/good -> test/good"
    cp "$CLS_DIR/DISthresh/good/"* "$CLS_DIR/test/good/"

    # 2. train/good ← all DISthresh/good images (from Normal), so get_foreground() can find them
    echo "  Copying DISthresh/good images -> train/good"
    cp "$CLS_DIR/DISthresh/good/"* "$CLS_DIR/train/good/"

    # 3. test/anomaly ← Data/Images/Anomaly
    echo "  Copying Anomaly images -> test/anomaly"
    cp "$CLS_DIR/Data/Images/Anomaly/"* "$CLS_DIR/test/anomaly/"

    # 4. ground_truth/anomaly ← Data/Masks/Anomaly
    echo "  Copying Masks -> ground_truth/anomaly"
    cp "$CLS_DIR/Data/Masks/Anomaly/"* "$CLS_DIR/ground_truth/anomaly/"

    echo "  Done: train/good=$(ls "$CLS_DIR/train/good/" | wc -l | tr -d ' '), test/good=$(ls "$CLS_DIR/test/good/" | wc -l | tr -d ' '), test/anomaly=$(ls "$CLS_DIR/test/anomaly/" | wc -l | tr -d ' '), ground_truth/anomaly=$(ls "$CLS_DIR/ground_truth/anomaly/" | wc -l | tr -d ' ')"
done

echo ""
echo "Reorganization complete. Original Data/ and DISthresh/ folders are preserved."
