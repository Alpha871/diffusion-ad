"""
Reorganize VisA dataset and skip corrupted DISthresh images.
Run on server: python reorganize_visa_v2.py
"""
import os
import glob
import shutil
import cv2

VISA_ROOT = "/home/jovyan/work/DiffusionAD/datasets/visa"

classes = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]

for cls in classes:
    print(f"=== Processing {cls} ===")
    cls_dir = os.path.join(VISA_ROOT, cls)

    train_good = os.path.join(cls_dir, "train", "good")
    test_good = os.path.join(cls_dir, "test", "good")
    test_anomaly = os.path.join(cls_dir, "test", "anomaly")
    gt_anomaly = os.path.join(cls_dir, "ground_truth", "anomaly")
    disthresh_good = os.path.join(cls_dir, "DISthresh", "good")
    normal_dir = os.path.join(cls_dir, "Data", "Images", "Normal")
    anomaly_dir = os.path.join(cls_dir, "Data", "Images", "Anomaly")
    masks_dir = os.path.join(cls_dir, "Data", "Masks", "Anomaly")

    # Create directories
    for d in [train_good, test_good, test_anomaly, gt_anomaly]:
        os.makedirs(d, exist_ok=True)

    # Clean old train/good and test/good
    for d in [train_good, test_good]:
        for f in glob.glob(os.path.join(d, "*")):
            os.remove(f)

    # 1. train/good and test/good ← only Normal images with a VALID DISthresh counterpart
    skipped = 0
    copied = 0
    for f in sorted(glob.glob(os.path.join(disthresh_good, "*.JPG"))):
        fname = os.path.basename(f)
        img = cv2.imread(f, 0)
        if img is None:
            skipped += 1
            continue
        src = os.path.join(normal_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(train_good, fname))
            shutil.copy2(src, os.path.join(test_good, fname))
            copied += 1
    print(f"  train/good: {copied}, test/good: {copied}, skipped: {skipped} (corrupted)")

    # 2. test/anomaly ← Data/Images/Anomaly
    anomaly_files = glob.glob(os.path.join(anomaly_dir, "*.JPG"))
    for f in anomaly_files:
        shutil.copy2(f, os.path.join(test_anomaly, os.path.basename(f)))
    print(f"  test/anomaly: {len(anomaly_files)}")

    # 3. ground_truth/anomaly ← Data/Masks/Anomaly
    mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
    for f in mask_files:
        shutil.copy2(f, os.path.join(gt_anomaly, os.path.basename(f)))
    print(f"  ground_truth/anomaly: {len(mask_files)}")

print("\nDone. Corrupted DISthresh images were skipped.")
