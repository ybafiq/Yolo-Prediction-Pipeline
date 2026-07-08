#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
active_learning.py
------------------
Active learning pipeline for YOLO plant disease segmentation.

Implements an uncertainty-based active learning loop:
  1. Score unlabeled images by model uncertainty
  2. Select the most informative images for annotation
  3. Integrate newly annotated images into the training set
  4. Retrain the model on the expanded dataset

Usage:
  python active_learning.py score   <model_path> [--config <config_path>]
  python active_learning.py select  <model_path> [--k 5] [--config <config_path>]
  python active_learning.py integrate [--config <config_path>]
  python active_learning.py retrain <model_path> [--config <config_path>]
  python active_learning.py full    <model_path> [--k 5] [--config <config_path>]
"""

import sys
import os
import json
import shutil
import argparse
import random
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# numpy is imported lazily inside functions that need it,
# so that basic CLI commands (status, integrate, help) work
# even without ML dependencies installed.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "active_learning_config.yaml")

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_config(config_path: str = None) -> dict:
    """Load the active learning configuration from a YAML or JSON file.
    Falls back to sensible defaults if no config file is found."""
    path = config_path or DEFAULT_CONFIG_PATH

    # Also check for a .json version of the config
    json_path = os.path.splitext(path)[0] + '.json'

    if os.path.exists(path):
        with open(path, 'r') as f:
            if yaml is not None:
                cfg = yaml.safe_load(f)
            else:
                # Try parsing as simple key: value pairs
                cfg = _parse_simple_yaml(f.read())
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            cfg = json.load(f)
    else:
        # Use defaults based on script location
        base = os.path.dirname(os.path.abspath(__file__))
        cfg = _default_config(base)
        print(f"[INFO] No config file found. Using default configuration.")

    return cfg


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML-like parser for simple key: value configs (no nesting)."""
    cfg = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Try numeric conversion
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, TypeError):
                pass
            cfg[key] = value
    return cfg


def _default_config(base_dir: str) -> dict:
    """Return default configuration dictionary."""
    return {
        'unlabeled_pool': os.path.join(base_dir, 'active_learning', 'unlabeled_pool', 'images'),
        'selected_dir': os.path.join(base_dir, 'active_learning', 'selected', 'images'),
        'annotated_dir': os.path.join(base_dir, 'active_learning', 'annotated'),
        'history_dir': os.path.join(base_dir, 'active_learning', 'history'),
        'train_dir': os.path.join(base_dir, 'train'),
        'val_dir': os.path.join(base_dir, 'val'),
        'top_k': 5,
        'confidence_threshold': 0.25,
        'expected_detections': 8,
        'val_split_ratio': 0.2,
        'weight_confidence': 0.4,
        'weight_margin': 0.3,
        'weight_detection_count': 0.3,
        'base_model': os.path.join(base_dir, 'yolo26m-seg.pt'),
        'data_yaml': os.path.join(base_dir, 'data_multiclass.yaml'),
        'epochs': 50,
        'imgsz': 640,
        'batch': 6,
        'device': 'cpu',
    }


def ensure_dirs(cfg: dict):
    """Create all required directories if they don't already exist."""
    for key in ('unlabeled_pool', 'selected_dir', 'annotated_dir', 'history_dir'):
        d = cfg.get(key, '')
        if d:
            os.makedirs(d, exist_ok=True)
    # Annotated sub-dirs
    ann_dir = cfg.get('annotated_dir', '')
    if ann_dir:
        os.makedirs(os.path.join(ann_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(ann_dir, 'labels'), exist_ok=True)


# ---------------------------------------------------------------------------
# Uncertainty scoring
# ---------------------------------------------------------------------------

def compute_uncertainty(model, image_path: str, cfg: dict) -> dict:
    """
    Run inference on a single image and compute an uncertainty score.

    Returns a dict with:
      - confidence_score: inverted mean confidence (higher = more uncertain)
      - margin_score:     class ambiguity score
      - detection_score:  deviation from expected detection count
      - combined_score:   weighted combination of the above
      - details:          raw numbers for debugging
    """
    conf_threshold = cfg.get('confidence_threshold', 0.25)
    expected_dets = cfg.get('expected_detections', 8)
    w_conf = cfg.get('weight_confidence', 0.4)
    w_margin = cfg.get('weight_margin', 0.3)
    w_det = cfg.get('weight_detection_count', 0.3)

    import numpy as np

    # Run inference
    results = model.predict(
        source=image_path,
        save=False,
        verbose=False,
        conf=conf_threshold,
        iou=0.45,
    )
    res = results[0]
    boxes = res.boxes

    # --- Handle no-detection case (maximum uncertainty) ---
    if boxes is None or len(boxes) == 0:
        return {
            'confidence_score': 1.0,
            'margin_score': 1.0,
            'detection_score': 1.0,
            'combined_score': 1.0,
            'details': {
                'num_detections': 0,
                'mean_confidence': 0.0,
                'confidences': [],
                'class_counts': {},
            }
        }

    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    num_dets = len(confidences)

    # --- 1. Confidence-based uncertainty ---
    # Mean confidence inverted: low avg conf -> high uncertainty
    mean_conf = float(np.mean(confidences))
    confidence_score = 1.0 - mean_conf

    # Bonus: count detections near the threshold boundary (within 0.15 of threshold)
    borderline_count = int(np.sum(confidences < (conf_threshold + 0.15)))
    borderline_bonus = min(1.0, borderline_count / max(1, num_dets))
    confidence_score = min(1.0, confidence_score + borderline_bonus * 0.2)

    # --- 2. Margin-based uncertainty ---
    # Look at class distribution: if nearly balanced healthy/affected -> ambiguous
    class_counts = {}
    for cls_id in classes:
        class_counts[int(cls_id)] = class_counts.get(int(cls_id), 0) + 1

    healthy_count = class_counts.get(0, 0)
    affected_count = class_counts.get(1, 0)
    total_classified = healthy_count + affected_count

    if total_classified > 0:
        # Ratio of minority class; perfectly balanced = 0.5 -> max ambiguity
        minority_ratio = min(healthy_count, affected_count) / total_classified
        # Scale: 0.5 (balanced) -> 1.0 uncertainty; 0.0 (all one class) -> 0.0
        margin_score = minority_ratio * 2.0
    else:
        margin_score = 0.5  # Unknown classes present

    # Also consider per-detection confidence spread
    if len(confidences) > 1:
        conf_std = float(np.std(confidences))
        # High std -> model is confident on some, unsure on others -> interesting
        margin_score = min(1.0, margin_score + conf_std * 0.5)

    # --- 3. Detection count anomaly ---
    # How far from the expected count of detections
    count_deviation = abs(num_dets - expected_dets) / expected_dets
    detection_score = min(1.0, count_deviation)

    # --- Combined score ---
    combined = (w_conf * confidence_score +
                w_margin * margin_score +
                w_det * detection_score)

    return {
        'confidence_score': round(confidence_score, 4),
        'margin_score': round(margin_score, 4),
        'detection_score': round(detection_score, 4),
        'combined_score': round(combined, 4),
        'details': {
            'num_detections': num_dets,
            'mean_confidence': round(mean_conf, 4),
            'confidences': [round(float(c), 4) for c in confidences],
            'class_counts': class_counts,
        }
    }


def rank_unlabeled_pool(model, pool_dir: str, cfg: dict) -> list:
    """
    Score all images in the unlabeled pool and return a ranked list.

    Returns list of (image_path, score_dict) sorted by combined_score descending.
    """
    if not os.path.isdir(pool_dir):
        print(f"[ERROR] Unlabeled pool directory not found: {pool_dir}")
        return []

    image_files = sorted([
        os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    if not image_files:
        print(f"[WARNING] No images found in unlabeled pool: {pool_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"  Scoring {len(image_files)} unlabeled images...")
    print(f"{'='*60}\n")

    rankings = []
    for i, img_path in enumerate(image_files, 1):
        fname = os.path.basename(img_path)
        try:
            scores = compute_uncertainty(model, img_path, cfg)
            rankings.append((img_path, scores))
            print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  "
                  f"uncertainty={scores['combined_score']:.4f}  "
                  f"(conf={scores['confidence_score']:.3f}  "
                  f"margin={scores['margin_score']:.3f}  "
                  f"det={scores['detection_score']:.3f})")
        except Exception as e:
            print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  [ERROR] {e}")

    # Sort by combined_score descending (most uncertain first)
    rankings.sort(key=lambda x: x[1]['combined_score'], reverse=True)

    return rankings


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_top_k(rankings: list, k: int, selected_dir: str) -> list:
    """
    Pick the top-K most uncertain images and copy them to the selected directory.

    Returns list of selected image paths (in the selected_dir).
    """
    os.makedirs(selected_dir, exist_ok=True)

    # Clear previous selections
    for f in os.listdir(selected_dir):
        fp = os.path.join(selected_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    top_k = rankings[:k]
    selected_paths = []

    print(f"\n{'='*60}")
    print(f"  Selecting top {min(k, len(rankings))} most uncertain images")
    print(f"{'='*60}\n")

    for rank, (img_path, scores) in enumerate(top_k, 1):
        fname = os.path.basename(img_path)
        dest = os.path.join(selected_dir, fname)
        shutil.copy2(img_path, dest)
        selected_paths.append(dest)
        print(f"  #{rank}: {fname}  (uncertainty={scores['combined_score']:.4f})")

    print(f"\n  -> {len(selected_paths)} images copied to: {selected_dir}")
    print(f"  -> Please annotate these images with segmentation masks,")
    print(f"    then place the image + label pairs in the annotated/ directory.\n")

    return selected_paths


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def integrate_annotations(cfg: dict) -> int:
    """
    Move annotated image+label pairs from the annotated directory into
    the training (and optionally validation) set.

    Returns the number of image-label pairs successfully integrated.
    """
    annotated_dir = cfg.get('annotated_dir', '')
    train_dir = cfg.get('train_dir', '')
    val_dir = cfg.get('val_dir', '')
    val_ratio = cfg.get('val_split_ratio', 0.2)

    ann_images_dir = os.path.join(annotated_dir, 'images')
    ann_labels_dir = os.path.join(annotated_dir, 'labels')

    if not os.path.isdir(ann_images_dir):
        print(f"[ERROR] Annotated images directory not found: {ann_images_dir}")
        return 0

    # Find all image-label pairs
    pairs = []
    for f in sorted(os.listdir(ann_images_dir)):
        if not f.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        stem = os.path.splitext(f)[0]
        label_file = stem + '.txt'
        label_path = os.path.join(ann_labels_dir, label_file)
        if os.path.exists(label_path):
            pairs.append((f, label_file))
        else:
            print(f"  [SKIP] No label found for: {f}")

    if not pairs:
        print("[WARNING] No valid image-label pairs found in annotated directory.")
        return 0

    print(f"\n{'='*60}")
    print(f"  Integrating {len(pairs)} annotated image-label pairs")
    print(f"{'='*60}\n")

    # Save a copy of the newly annotated images and labels in a separate folder
    history_dir = cfg.get('history_dir', '')
    round_num = 1
    if os.path.isdir(history_dir):
        existing = [f for f in os.listdir(history_dir) if f.startswith('round_')]
        round_num = len(existing) + 1

    active_learning_parent = os.path.dirname(annotated_dir)
    backup_dataset_dir = os.path.join(active_learning_parent, f"new_annotated_image_dataset_{round_num}")
    
    os.makedirs(os.path.join(backup_dataset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(backup_dataset_dir, 'labels'), exist_ok=True)
    
    backup_count = 0
    for img_file, lbl_file in pairs:
        src_img = os.path.join(ann_images_dir, img_file)
        src_lbl = os.path.join(ann_labels_dir, lbl_file)
        dst_img = os.path.join(backup_dataset_dir, 'images', img_file)
        dst_lbl = os.path.join(backup_dataset_dir, 'labels', lbl_file)
        try:
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            backup_count += 1
        except Exception as e:
            print(f"  [WARNING] Failed to backup {img_file}: {e}")
            
    print(f"  -> Backed up {backup_count} pairs to: {backup_dataset_dir}")

    # Shuffle and split into train / val
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) >= 5 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    integrated_count = 0

    for img_file, lbl_file in train_pairs:
        _move_pair(ann_images_dir, ann_labels_dir, img_file, lbl_file,
                   os.path.join(train_dir, 'images'),
                   os.path.join(train_dir, 'labels'),
                   "TRAIN")
        integrated_count += 1

    for img_file, lbl_file in val_pairs:
        _move_pair(ann_images_dir, ann_labels_dir, img_file, lbl_file,
                   os.path.join(val_dir, 'images'),
                   os.path.join(val_dir, 'labels'),
                   "VAL")
        integrated_count += 1

    print(f"\n  -> Integrated {len(train_pairs)} to train, {len(val_pairs)} to val")
    print(f"  -> Total: {integrated_count} pairs\n")

    return integrated_count


def _move_pair(src_img_dir, src_lbl_dir, img_file, lbl_file,
               dst_img_dir, dst_lbl_dir, split_name):
    """Move a single image-label pair from source to destination."""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    src_img = os.path.join(src_img_dir, img_file)
    src_lbl = os.path.join(src_lbl_dir, lbl_file)
    dst_img = os.path.join(dst_img_dir, img_file)
    dst_lbl = os.path.join(dst_lbl_dir, lbl_file)

    # Handle name collisions by appending a suffix
    if os.path.exists(dst_img):
        stem, ext = os.path.splitext(img_file)
        suffix = datetime.now().strftime("_%Y%m%d%H%M%S")
        img_file = stem + suffix + ext
        lbl_file = stem + suffix + '.txt'
        dst_img = os.path.join(dst_img_dir, img_file)
        dst_lbl = os.path.join(dst_lbl_dir, lbl_file)

    shutil.move(src_img, dst_img)
    shutil.move(src_lbl, dst_lbl)
    print(f"  [{split_name}] {img_file}")


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def retrain_model(model_path: str, cfg: dict) -> str:
    """
    Retrain the YOLO model on the expanded dataset.

    Returns the path to the best model weights from this training run.
    """
    from ultralytics import YOLO

    data_yaml = cfg.get('data_yaml', 'data_multiclass.yaml')
    epochs = cfg.get('epochs', 50)
    imgsz = cfg.get('imgsz', 640)
    batch = cfg.get('batch', 6)
    device = cfg.get('device', 'cpu')

    # Count current training images
    train_img_dir = os.path.join(cfg.get('train_dir', ''), 'images')
    if os.path.isdir(train_img_dir):
        n_train = len([f for f in os.listdir(train_img_dir)
                       if f.lower().endswith(SUPPORTED_EXTENSIONS)])
    else:
        n_train = '?'

    # Generate a round name from current timestamp
    round_name = datetime.now().strftime("al_round_%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  Retraining model")
    print(f"  Base model:    {model_path}")
    print(f"  Training imgs: {n_train}")
    print(f"  Epochs:        {epochs}")
    print(f"  Run name:      {round_name}")
    print(f"{'='*60}\n")

    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="active_learning_runs",
        name=round_name,
        workers=2,
    )

    # Find the best weights path
    best_weights = os.path.join("active_learning_runs", round_name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\n  -> Best weights saved to: {best_weights}")
    else:
        # Fallback: try last.pt
        best_weights = os.path.join("active_learning_runs", round_name, "weights", "last.pt")
        print(f"\n  -> Weights saved to: {best_weights}")

    return best_weights


# ---------------------------------------------------------------------------
# History logging
# ---------------------------------------------------------------------------

def save_round_history(cfg: dict, round_data: dict):
    """Save the results of an active learning round to the history directory."""
    history_dir = cfg.get('history_dir', '')
    os.makedirs(history_dir, exist_ok=True)

    # Find next round number
    existing = [f for f in os.listdir(history_dir) if f.startswith('round_')]
    round_num = len(existing) + 1
    filename = f"round_{round_num:03d}.json"
    filepath = os.path.join(history_dir, filename)

    round_data['timestamp'] = datetime.now().isoformat()
    round_data['round_number'] = round_num

    with open(filepath, 'w') as f:
        json.dump(round_data, f, indent=2, default=str)

    print(f"  -> Round history saved to: {filepath}")


# ---------------------------------------------------------------------------
# Auto-annotation
# ---------------------------------------------------------------------------

def auto_annotate(model, image_dir: str, output_dir: str, cfg: dict) -> int:
    """
    Run model inference on images and save predictions as YOLO-format labels.

    This generates pre-annotations so the user only needs to review/correct
    them in Label Studio instead of annotating from scratch.

    Args:
        model:      Loaded YOLO model
        image_dir:  Directory containing images to auto-annotate
        output_dir: Directory to save the generated label .txt files
        cfg:        Config dictionary

    Returns:
        Number of images successfully auto-annotated.
    """
    import numpy as np
    import cv2

    conf_threshold = cfg.get('confidence_threshold', 0.25)

    if not os.path.isdir(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        return 0

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    if not image_files:
        print(f"[WARNING] No images found in: {image_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Auto-annotating {len(image_files)} images...")
    print(f"  Model confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")

    annotated_count = 0

    for i, fname in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(output_dir, stem + '.txt')

        try:
            # Read image dimensions for normalization
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  [ERROR] Could not read image")
                continue
            img_h, img_w = img.shape[:2]

            # Run inference
            results = model.predict(
                source=img_path,
                save=False,
                verbose=False,
                conf=conf_threshold,
                iou=0.45,
            )
            res = results[0]

            lines = []
            n_detections = 0

            if res.masks is not None and len(res.masks) > 0:
                masks_data = res.masks.data.cpu().numpy()   # (N, H_net, W_net)
                classes = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()

                for j in range(len(masks_data)):
                    cls_id = int(classes[j])
                    conf = float(confs[j])

                    # Resize mask to original image size
                    mask = cv2.resize(masks_data[j], (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST)

                    # Extract contour polygon from binary mask
                    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    # Use the largest contour
                    contour = max(contours, key=cv2.contourArea)

                    # Simplify contour to reduce point count
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                    if len(contour) < 3:
                        continue

                    # Convert to normalized YOLO polygon format
                    # Format: class_id x1 y1 x2 y2 x3 y3 ...
                    points = contour.reshape(-1, 2)
                    normalized = []
                    for px, py in points:
                        normalized.append(f"{px / img_w:.6f}")
                        normalized.append(f"{py / img_h:.6f}")

                    line = f"{cls_id} " + " ".join(normalized)
                    lines.append(line)
                    n_detections += 1

            # Write label file (even if empty, to mark as processed)
            with open(label_path, 'w') as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            annotated_count += 1
            print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  "
                  f"{n_detections} detections -> {os.path.basename(label_path)}")

        except Exception as e:
            print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  [ERROR] {e}")

    print(f"\n  -> {annotated_count} images auto-annotated")
    print(f"  -> Labels saved to: {output_dir}\n")

    return annotated_count


def auto_annotate_for_label_studio(model, image_dir: str, output_json: str,
                                    cfg: dict) -> int:
    """
    Generate Label Studio pre-annotation JSON for importing into Label Studio.

    Creates a JSON file that can be imported directly into a Label Studio project
    with brush-based segmentation labels.

    Args:
        model:       Loaded YOLO model
        image_dir:   Directory containing images
        output_json: Path to save the Label Studio import JSON
        cfg:         Config dictionary

    Returns:
        Number of images processed.
    """
    import numpy as np
    import cv2
    import base64
    from io import BytesIO

    conf_threshold = cfg.get('confidence_threshold', 0.25)

    # Class name mapping
    class_names = {0: 'healthy_leaf', 1: 'affected_leaf'}

    if not os.path.isdir(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        return 0

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    if not image_files:
        print(f"[WARNING] No images found in: {image_dir}")
        return 0

    print(f"\n{'='*60}")
    print(f"  Generating Label Studio pre-annotations...")
    print(f"{'='*60}\n")

    tasks = []

    for i, fname in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, fname)

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            results = model.predict(
                source=img_path,
                save=False,
                verbose=False,
                conf=conf_threshold,
                iou=0.45,
            )
            res = results[0]

            annotations = []

            if res.masks is not None and len(res.masks) > 0:
                masks_data = res.masks.data.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()

                for j in range(len(masks_data)):
                    cls_id = int(classes[j])
                    conf = float(confs[j])
                    cls_name = class_names.get(cls_id, f'class_{cls_id}')

                    # Get bounding box as percentages
                    x1, y1, x2, y2 = boxes_xyxy[j]
                    box_x = float(x1 / img_w * 100)
                    box_y = float(y1 / img_h * 100)
                    box_w = float((x2 - x1) / img_w * 100)
                    box_h = float((y2 - y1) / img_h * 100)

                    # Resize mask to original size and extract polygon
                    mask = cv2.resize(masks_data[j], (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST)
                    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                    if len(contour) < 3:
                        continue

                    # Convert to percentage-based polygon points for Label Studio
                    points = contour.reshape(-1, 2)
                    ls_points = [[float(px / img_w * 100), float(py / img_h * 100)]
                                 for px, py in points]

                    annotation_result = {
                        "type": "polygonlabels",
                        "value": {
                            "points": ls_points,
                            "polygonlabels": [cls_name],
                        },
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": img_w,
                        "original_height": img_h,
                        "image_rotation": 0,
                    }
                    annotations.append(annotation_result)

            # Build the Label Studio task
            task = {
                "data": {
                    "image": f"/data/local-files/?d={os.path.abspath(img_path).replace(os.sep, '/')}",
                },
                "predictions": [{
                    "model_version": "yolo_auto_annotate",
                    "score": float(np.mean(confs)) if res.boxes is not None and len(res.boxes) > 0 else 0.0,
                    "result": annotations,
                }]
            }
            tasks.append(task)
            print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  {len(annotations)} annotations")

        except Exception as e:
            print(f"  [{i:3d}/{len(image_files)}] {fname:40s}  [ERROR] {e}")

    # Write Label Studio JSON
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"\n  -> {len(tasks)} tasks saved to: {output_json}")
    print(f"  -> Import this file into Label Studio as 'Predictions'\n")

    return len(tasks)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_auto_annotate(args, cfg):
    """Auto-annotate images using model predictions."""
    from ultralytics import YOLO

    model = YOLO(args.model_path)

    # Determine source directory: --source flag, or default to selected_dir
    if args.source:
        image_dir = args.source
    else:
        image_dir = cfg.get('selected_dir', '')

    # Determine output directory/format
    annotated_dir = cfg.get('annotated_dir', '')

    if args.format == 'yolo':
        # Output YOLO format labels alongside images in the annotated dir
        output_labels_dir = os.path.join(annotated_dir, 'labels')
        output_images_dir = os.path.join(annotated_dir, 'images')

        count = auto_annotate(model, image_dir, output_labels_dir, cfg)

        if count > 0:
            # Also copy images to annotated/images/ if not already there
            os.makedirs(output_images_dir, exist_ok=True)
            copied = 0
            for f in os.listdir(image_dir):
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    src = os.path.join(image_dir, f)
                    dst = os.path.join(output_images_dir, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied += 1
            if copied:
                print(f"  -> {copied} images copied to: {output_images_dir}")

            print(f"\n  NEXT STEPS:")
            print(f"  1. Review and correct the auto-generated labels")
            print(f"     Labels: {output_labels_dir}")
            print(f"     Images: {output_images_dir}")
            print(f"  2. When satisfied, run: python active_learning.py integrate")
            print(f"  3. Then run: python active_learning.py retrain <model_path>\n")

            save_round_history(cfg, {
                'action': 'auto_annotate',
                'model_path': args.model_path,
                'format': 'yolo',
                'num_annotated': count,
                'source_dir': image_dir,
            })

    elif args.format == 'labelstudio':
        # Output Label Studio JSON for import
        output_json = os.path.join(annotated_dir, 'label_studio_import.json')
        count = auto_annotate_for_label_studio(model, image_dir, output_json, cfg)

        if count > 0:
            print(f"  NEXT STEPS (Label Studio):")
            print(f"  1. Open Label Studio and go to your project")
            print(f"  2. Click 'Import' and upload: {output_json}")
            print(f"  3. Review/correct the pre-annotations")
            print(f"  4. Export as 'YOLO' format")
            print(f"  5. Place exported images+labels in:")
            print(f"       Images: {os.path.join(annotated_dir, 'images')}")
            print(f"       Labels: {os.path.join(annotated_dir, 'labels')}")
            print(f"  6. Run: python active_learning.py integrate\n")

            # Also generate YOLO labels as backup
            output_labels_dir = os.path.join(annotated_dir, 'labels')
            output_images_dir = os.path.join(annotated_dir, 'images')
            auto_annotate(model, image_dir, output_labels_dir, cfg)
            os.makedirs(output_images_dir, exist_ok=True)
            for f in os.listdir(image_dir):
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    src = os.path.join(image_dir, f)
                    dst = os.path.join(output_images_dir, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)

            save_round_history(cfg, {
                'action': 'auto_annotate',
                'model_path': args.model_path,
                'format': 'labelstudio',
                'num_annotated': count,
            })


def cmd_score(args, cfg):
    """Score all images in the unlabeled pool and display rankings."""
    from ultralytics import YOLO

    model = YOLO(args.model_path)
    pool_dir = cfg.get('unlabeled_pool', '')

    rankings = rank_unlabeled_pool(model, pool_dir, cfg)

    if rankings:
        print(f"\n{'='*60}")
        print(f"  FINAL RANKINGS (most uncertain first)")
        print(f"{'='*60}\n")
        for rank, (img_path, scores) in enumerate(rankings, 1):
            fname = os.path.basename(img_path)
            print(f"  #{rank:3d}  {fname:40s}  score={scores['combined_score']:.4f}")

        # Save rankings to history
        save_round_history(cfg, {
            'action': 'score',
            'model_path': args.model_path,
            'num_images': len(rankings),
            'rankings': [
                {'image': os.path.basename(p), 'scores': s}
                for p, s in rankings
            ]
        })


def cmd_select(args, cfg):
    """Score the pool and select the top-K most uncertain images."""
    from ultralytics import YOLO

    model = YOLO(args.model_path)
    pool_dir = cfg.get('unlabeled_pool', '')
    selected_dir = cfg.get('selected_dir', '')
    k = args.k or cfg.get('top_k', 5)

    rankings = rank_unlabeled_pool(model, pool_dir, cfg)

    if rankings:
        selected = select_top_k(rankings, k, selected_dir)

        save_round_history(cfg, {
            'action': 'select',
            'model_path': args.model_path,
            'k': k,
            'num_pool': len(rankings),
            'selected': [os.path.basename(p) for p in selected],
            'all_rankings': [
                {'image': os.path.basename(p), 'scores': s}
                for p, s in rankings
            ]
        })

        print(f"\n  NEXT STEPS:")
        print(f"  1. Annotate the selected images with segmentation masks")
        print(f"     (YOLO format: class_id x1 y1 x2 y2 ... polygon points)")
        print(f"  2. Place the annotated image + .txt label files in:")
        print(f"       Images: {os.path.join(cfg.get('annotated_dir', ''), 'images')}")
        print(f"       Labels: {os.path.join(cfg.get('annotated_dir', ''), 'labels')}")
        print(f"  3. Run: python active_learning.py integrate")
        print(f"  4. Run: python active_learning.py retrain <model_path>\n")
    else:
        print("\n  No images to select from. Add images to the unlabeled pool first.\n")


def cmd_integrate(args, cfg):
    """Integrate annotated images into the training set."""
    count = integrate_annotations(cfg)

    if count > 0:
        save_round_history(cfg, {
            'action': 'integrate',
            'num_integrated': count,
        })

        print(f"  NEXT STEP:")
        print(f"  Run: python active_learning.py retrain <model_path>\n")


def cmd_retrain(args, cfg):
    """Retrain the model on the expanded dataset."""
    best_weights = retrain_model(args.model_path, cfg)

    save_round_history(cfg, {
        'action': 'retrain',
        'base_model': args.model_path,
        'best_weights': best_weights,
    })

    print(f"\n  NEXT STEPS:")
    print(f"  1. Use the new model for prediction:")
    print(f"     python predict_true_individual_severity.py <image> {best_weights}")
    print(f"  2. Start a new active learning round:")
    print(f"     python active_learning.py select {best_weights}\n")


def cmd_full(args, cfg):
    """Run a full active learning round: score -> select -> wait for annotation."""
    from ultralytics import YOLO

    model = YOLO(args.model_path)
    pool_dir = cfg.get('unlabeled_pool', '')
    selected_dir = cfg.get('selected_dir', '')
    k = args.k or cfg.get('top_k', 5)

    print(f"\n{'='*60}")
    print(f"  ACTIVE LEARNING — FULL ROUND")
    print(f"{'='*60}")

    # Step 1: Score
    rankings = rank_unlabeled_pool(model, pool_dir, cfg)

    if not rankings:
        print("\n  [ABORT] No unlabeled images found. Add images to:")
        print(f"    {pool_dir}\n")
        return

    # Step 2: Select
    selected = select_top_k(rankings, k, selected_dir)

    # Save history
    save_round_history(cfg, {
        'action': 'full_round_select',
        'model_path': args.model_path,
        'k': k,
        'num_pool': len(rankings),
        'selected': [os.path.basename(p) for p in selected],
    })

    print(f"\n{'='*60}")
    print(f"  PAUSED — Annotation required")
    print(f"{'='*60}")
    print(f"\n  {len(selected)} images have been selected for annotation.")
    print(f"  Selected images are in: {selected_dir}")
    print(f"\n  When you finish annotating:")
    print(f"  1. Place image + label files in:")
    print(f"       Images: {os.path.join(cfg.get('annotated_dir', ''), 'images')}")
    print(f"       Labels: {os.path.join(cfg.get('annotated_dir', ''), 'labels')}")
    print(f"  2. Run: python active_learning.py integrate")
    print(f"  3. Run: python active_learning.py retrain {args.model_path}\n")


def cmd_auto_loop(args, cfg):
    """Run the entire active learning loop automatically without user intervention:
       1. Score and select top-k images from unlabeled pool.
       2. Auto-annotate the selected images (YOLO format).
       3. Delete/remove the selected images from the unlabeled pool (so they aren't processed again).
       4. Integrate the annotated images into train/val splits.
       5. Retrain the model.
    """
    from ultralytics import YOLO

    model = YOLO(args.model_path)
    pool_dir = cfg.get('unlabeled_pool', '')
    selected_dir = cfg.get('selected_dir', '')
    annotated_dir = cfg.get('annotated_dir', '')
    k = args.k or cfg.get('top_k', 5)

    print(f"\n{'='*60}")
    print(f"  STARTING AUTO ACTIVE LEARNING LOOP")
    print(f"{'='*60}\n")

    # Step 1: Score & Select
    rankings = rank_unlabeled_pool(model, pool_dir, cfg)

    if not rankings:
        print("\n  [INFO] No unlabeled images found in pool. Auto-loop complete or nothing to do.\n")
        return

    selected = select_top_k(rankings, k, selected_dir)
    
    if not selected:
        print("\n  [INFO] No images were selected. Exiting.\n")
        return

    # Step 2: Auto-annotate (generate pseudo-labels)
    output_labels_dir = os.path.join(annotated_dir, 'labels')
    output_images_dir = os.path.join(annotated_dir, 'images')
    
    count = auto_annotate(model, selected_dir, output_labels_dir, cfg)
    
    if count == 0:
        print("\n  [ERROR] Auto-annotation failed to produce labels. Exiting loop.\n")
        return

    # Copy selected images to the annotated directory as well (needed for integration)
    os.makedirs(output_images_dir, exist_ok=True)
    for f in os.listdir(selected_dir):
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            src = os.path.join(selected_dir, f)
            dst = os.path.join(output_images_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Step 3: Remove selected images from the unlabeled pool
    # This prevents them from being selected again in subsequent runs.
    print(f"\n  Cleaning up {len(selected)} processed images from unlabeled pool...")
    for img_path, _ in rankings[:k]:
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                print(f"    Removed: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"    [WARNING] Failed to remove {os.path.basename(img_path)}: {e}")

    # Step 4: Integrate into train/val datasets
    print("\n  Integrating annotations...")
    integrated_count = integrate_annotations(cfg)

    if integrated_count == 0:
        print("\n  [ERROR] No annotations were integrated. Exiting loop.\n")
        return

    # Log history
    save_round_history(cfg, {
        'action': 'auto_loop',
        'model_path': args.model_path,
        'k': k,
        'num_pool': len(rankings),
        'selected': [os.path.basename(p) for p in selected],
        'num_integrated': integrated_count
    })

    # Step 5: Retrain the model on the updated dataset
    print("\n  Starting retraining...")
    best_weights = retrain_model(args.model_path, cfg)
    
    print(f"\n{'='*60}")
    print(f"  AUTO ACTIVE LEARNING LOOP COMPLETE")
    print(f"  New weights saved to: {best_weights}")
    print(f"{'='*60}\n")


def cmd_status(args, cfg):
    """Show the current status of the active learning pipeline."""
    pool_dir = cfg.get('unlabeled_pool', '')
    selected_dir = cfg.get('selected_dir', '')
    annotated_dir = cfg.get('annotated_dir', '')
    train_dir = cfg.get('train_dir', '')
    val_dir = cfg.get('val_dir', '')
    history_dir = cfg.get('history_dir', '')

    def count_images(d):
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d)
                    if f.lower().endswith(SUPPORTED_EXTENSIONS)])

    n_pool = count_images(pool_dir)
    n_selected = count_images(selected_dir)
    n_annotated = count_images(os.path.join(annotated_dir, 'images'))
    n_train = count_images(os.path.join(train_dir, 'images'))
    n_val = count_images(os.path.join(val_dir, 'images'))

    # Count rounds
    n_rounds = 0
    if os.path.isdir(history_dir):
        n_rounds = len([f for f in os.listdir(history_dir)
                        if f.startswith('round_')])

    print(f"\n{'='*60}")
    print(f"  ACTIVE LEARNING STATUS")
    print(f"{'='*60}\n")
    print(f"  Unlabeled pool:    {n_pool:4d} images  ({pool_dir})")
    print(f"  Selected:          {n_selected:4d} images  ({selected_dir})")
    print(f"  Awaiting integrate:{n_annotated:4d} images  ({annotated_dir})")
    print(f"  Training set:      {n_train:4d} images  ({os.path.join(train_dir, 'images')})")
    print(f"  Validation set:    {n_val:4d} images  ({os.path.join(val_dir, 'images')})")
    print(f"  Completed rounds:  {n_rounds:4d}")
    print()

    # Show latest round info
    if n_rounds > 0 and os.path.isdir(history_dir):
        history_files = sorted([f for f in os.listdir(history_dir)
                                if f.startswith('round_')])
        latest = history_files[-1]
        with open(os.path.join(history_dir, latest), 'r') as f:
            data = json.load(f)
        print(f"  Latest round: {latest}")
        print(f"    Action:    {data.get('action', '?')}")
        print(f"    Timestamp: {data.get('timestamp', '?')}")
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_parser():
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Active Learning Pipeline for YOLO Plant Disease Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check pipeline status
  python active_learning.py status

  # Score unlabeled images
  python active_learning.py score sfs104.pt

  # Select top 5 most uncertain images for annotation
  python active_learning.py select sfs104.pt --k 5

  # After annotating, integrate into training set
  python active_learning.py integrate

  # Retrain model on expanded dataset
  python active_learning.py retrain sfs104.pt

  # Run full round (score + select)
  python active_learning.py full sfs104.pt --k 5

  # Auto-annotate selected images (YOLO format)
  python active_learning.py auto-annotate sfs104.pt

  # Auto-annotate with Label Studio JSON export
  python active_learning.py auto-annotate sfs104.pt --format labelstudio

  # Run fully automated active learning loop (pre-label, integrate, retrain)
  python active_learning.py auto-loop sfs104.pt --k 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -- status --
    sp_status = subparsers.add_parser('status', help='Show pipeline status')
    sp_status.add_argument('--config', default=None, help='Path to config YAML')

    # -- score --
    sp_score = subparsers.add_parser('score', help='Score unlabeled images')
    sp_score.add_argument('model_path', help='Path to YOLO model weights')
    sp_score.add_argument('--config', default=None, help='Path to config YAML')

    # -- select --
    sp_select = subparsers.add_parser('select', help='Select top-K uncertain images')
    sp_select.add_argument('model_path', help='Path to YOLO model weights')
    sp_select.add_argument('--k', type=int, default=None,
                           help='Number of images to select (default: from config)')
    sp_select.add_argument('--config', default=None, help='Path to config YAML')

    # -- integrate --
    sp_integrate = subparsers.add_parser('integrate',
                                         help='Move annotated images to training set')
    sp_integrate.add_argument('--config', default=None, help='Path to config YAML')

    # -- retrain --
    sp_retrain = subparsers.add_parser('retrain', help='Retrain model on expanded dataset')
    sp_retrain.add_argument('model_path', help='Path to YOLO model weights to retrain from')
    sp_retrain.add_argument('--config', default=None, help='Path to config YAML')

    # -- full --
    sp_full = subparsers.add_parser('full', help='Full round: score -> select')
    sp_full.add_argument('model_path', help='Path to YOLO model weights')
    sp_full.add_argument('--k', type=int, default=None,
                         help='Number of images to select (default: from config)')
    sp_full.add_argument('--config', default=None, help='Path to config YAML')

    # -- auto-annotate --
    sp_auto = subparsers.add_parser('auto-annotate',
                                     help='Auto-annotate images using model predictions')
    sp_auto.add_argument('model_path', help='Path to YOLO model weights')
    sp_auto.add_argument('--format', choices=['yolo', 'labelstudio'], default='yolo',
                         help='Output format: yolo (default) or labelstudio')
    sp_auto.add_argument('--source', default=None,
                         help='Source image directory (default: selected/ folder)')
    sp_auto.add_argument('--config', default=None, help='Path to config YAML')

    # -- auto-loop --
    sp_autoloop = subparsers.add_parser('auto-loop',
                                        help='Automatically run a full active learning round (score -> select -> auto-annotate -> remove from pool -> integrate -> retrain)')
    sp_autoloop.add_argument('model_path', help='Path to YOLO model weights')
    sp_autoloop.add_argument('--k', type=int, default=None,
                             help='Number of images to select (default: from config)')
    sp_autoloop.add_argument('--config', default=None, help='Path to config YAML')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cfg = load_config(getattr(args, 'config', None))
    ensure_dirs(cfg)

    commands = {
        'status': cmd_status,
        'score': cmd_score,
        'select': cmd_select,
        'auto-annotate': cmd_auto_annotate,
        'auto-loop': cmd_auto_loop,
        'integrate': cmd_integrate,
        'retrain': cmd_retrain,
        'full': cmd_full,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args, cfg)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
