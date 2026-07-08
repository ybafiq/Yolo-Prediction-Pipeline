#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_true_individual_severity.py
-----------------------------------
Inference script to track disease severity on individual plant clones in a test tube rack.
Accepts:
  - image_path: Path to the input image (Front or Back view).
  - multiclass_model_path: Path to the retrained YOLO multi-class segmentation model.
  - tube_model_path (optional): Path to the single-class plant detector (e.g. sfs104.pt)
    used to locate the 8 tube boundaries. If not provided, a programmatic grid fallback is used.
"""

import sys
import os
import json
import numpy as np
import cv2

def map_severity(percentage: float) -> str:
    """Map the numerical severity percentage to a descriptive level."""
    if percentage >= 75.0:
        return "Critical"
    elif percentage >= 50.0:
        return "High"
    elif percentage >= 25.0:
        return "Moderate"
    else:
        return "Low"

def fit_tube_grid(boxes, img_w, img_h):
    """
    Fits a 1D linear grid to the X-centers of detected boxes to assign each to one of 8 slots.
    This handles missing plants or empty tubes by reconstructing the empty slots.
    Returns a list of 8 bounding boxes, one for each slot, sorted from left to right.
    """
    if len(boxes) == 0:
        col_w = img_w / 8.0
        return [[i * col_w, 0.0, (i + 1) * col_w, img_h] for i in range(8)]
    
    # Calculate centers and vertical bounds
    centers = []
    y1s, y2s = [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        centers.append((x1 + x2) / 2.0)
        y1s.append(y1)
        y2s.append(y2)
    
    centers = sorted(centers)
    min_y = max(0.0, float(min(y1s)))
    max_y = min(float(img_h), float(max(y2s)))
    
    best_score = float('inf')
    best_grid = None
    
    # Grid search: try various tube spacings S (typical spacing is 8% to 15% of image width)
    spacings_to_try = np.linspace(img_w / 14.0, img_w / 6.0, 200)
    for S in spacings_to_try:
        # Try assigning the first detection to each of the 8 possible slots
        for start_slot in range(8):
            x0 = centers[0] - start_slot * S
            
            slots = []
            fit_err = 0.0
            valid = True
            for c in centers:
                slot = int(round((c - x0) / S))
                if slot < 0 or slot > 7:
                    valid = False
                    break
                slots.append(slot)
                fit_err += ((x0 + slot * S) - c) ** 2
            
            # Ensure each detection fits a unique slot
            if valid and len(set(slots)) == len(centers):
                if fit_err < best_score:
                    best_score = fit_err
                    best_grid = (x0, S, slots)
                    
    if best_grid is not None:
        x0, S, slots = best_grid
    else:
        # Simple fallback spacing if grid fitting fails
        span = centers[-1] - centers[0]
        if span > 0:
            est_slots = max(1, min(7, int(round(span / (img_w * 0.1)))))
            S = span / est_slots
            x0 = centers[0]
        else:
            S = img_w * 0.1
            x0 = centers[0] - 3.5 * S
            
    # Generate the 8 bounding boxes representing the 8 tube columns
    tube_boxes = []
    for slot in range(8):
        center_x = x0 + slot * S
        x1 = max(0.0, center_x - S / 2.0)
        x2 = min(float(img_w), center_x + S / 2.0)
        tube_boxes.append([x1, min_y, x2, max_y])
        
    return tube_boxes

def process_single_image(image_path, model_multi, model_tube=None):
    """
    Processes a single image and returns the list of severity metrics per clone.
    """
    # Determine view orientation from filename
    filename_lower = os.path.basename(image_path).lower()
    is_back_view = "back" in filename_lower or "_b.png" in filename_lower or "_b.jpg" in filename_lower or "_b.jpeg" in filename_lower
    
    # Run multi-class segmentation prediction
    try:
        results_multi = model_multi.predict(
            source=image_path,
            save=False,
            verbose=False,
            conf=0.25,
            iou=0.45
        )
    except Exception as e:
        return {"error": f"Multi-class inference failed: {str(e)}"}
        
    # Get image metadata and dimensions
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return {"error": f"OpenCV failed to read image: {image_path}"}
        
    img_h, img_w = img_cv.shape[:2]
    
    # Extract multi-class masks and boxes
    res_multi = results_multi[0]
    multi_masks = res_multi.masks
    multi_boxes = res_multi.boxes
    
    # Determine tube boundaries
    tube_boxes = []
    
    # Attempt to run tube detection model if provided
    if model_tube is not None:
        try:
            results_tube = model_tube.predict(
                source=image_path,
                save=False,
                verbose=False,
                conf=0.25,
                iou=0.45
            )
            res_tube = results_tube[0]
            if res_tube.boxes is not None and len(res_tube.boxes) > 0:
                detected_boxes = res_tube.boxes.xyxy.cpu().numpy().tolist()
                tube_boxes = fit_tube_grid(detected_boxes, img_w, img_h)
        except Exception as e:
            # Fall back silently if tube model fails
            pass
            
    # Fallback if no tube boxes were obtained
    if len(tube_boxes) == 0:
        # Extract leaf boxes from the multi-class model to define the rack span
        if multi_boxes is not None and len(multi_boxes) > 0:
            leaf_xyxy = multi_boxes.xyxy.cpu().numpy()
            min_x = float(np.min(leaf_xyxy[:, 0]))
            max_x = float(np.max(leaf_xyxy[:, 2]))
            min_y = float(np.min(leaf_xyxy[:, 1]))
            max_y = float(np.max(leaf_xyxy[:, 3]))
            
            # Pad horizontal span by 5% on each side
            span = max_x - min_x
            min_x = max(0.0, min_x - span * 0.05)
            max_x = min(float(img_w), max_x + span * 0.05)
            
            col_w = (max_x - min_x) / 8.0
            for i in range(8):
                tube_boxes.append([min_x + i * col_w, min_y, min_x + (i + 1) * col_w, max_y])
        else:
            # Ultimate fallback: divide entire image width
            col_w = img_w / 8.0
            tube_boxes = [[i * col_w, 0.0, (i + 1) * col_w, float(img_h)] for i in range(8)]
            
    # X-axis sorting logic based on rack orientation
    # For Front images: ascending X (left to right, physical order 1 to 8)
    # For Back images: descending X (right to left, physical order 1 to 8)
    sorted_boxes = list(tube_boxes)
    if is_back_view:
        sorted_boxes.reverse()
        
    # Prepare binary mask arrays resized to full image shape
    resized_masks = []
    mask_classes = []
    
    if multi_masks is not None and len(multi_masks) > 0:
        mask_data = multi_masks.data.cpu().numpy()  # (N, H_net, W_net)
        classes = multi_boxes.cls.cpu().numpy()
        
        for i in range(len(mask_data)):
            # Resize binary mask to original image dimensions
            mask_resized = cv2.resize(mask_data[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(mask_resized)
            mask_classes.append(int(classes[i]))
            
    # Calculate severity inside each tube box
    output = []
    for idx, box in enumerate(sorted_boxes, start=1):
        x1, y1, x2, y2 = box
        
        healthy_pixels = 0
        affected_pixels = 0
        
        # Check all leaf masks to find pixels falling inside this tube's boundaries
        for mask, cls_id in zip(resized_masks, mask_classes):
            if cls_id not in [0, 1]:
                continue
                
            # Crop mask to tube boundary
            crop = mask[int(y1):int(y2), int(x1):int(x2)]
            active_pixels = int(np.sum(crop > 0.5))
            
            if cls_id == 0:
                healthy_pixels += active_pixels
            elif cls_id == 1:
                affected_pixels += active_pixels
                
        # Apply severity formula
        total_pixels = healthy_pixels + affected_pixels
        if total_pixels > 0:
            severity_pct = (affected_pixels / float(total_pixels)) * 100.0
        else:
            severity_pct = 0.0
            
        # Format output dictionary
        output.append({
            "clone_index": idx,
            "healthy_pixels": healthy_pixels,
            "affected_pixels": affected_pixels,
            "severity_percentage": round(severity_pct, 2),
            "severity_level": map_severity(severity_pct)
        })
        
    return output

def main():
    if len(sys.argv) < 3:
        error_msg = {
            "error": "Insufficient arguments. Usage: python predict_true_individual_severity.py <image_path_or_directory> <multiclass_model_path> [tube_model_path]"
        }
        print(json.dumps(error_msg, indent=2))
        sys.exit(1)
        
    path_arg = sys.argv[1]
    multiclass_model_path = sys.argv[2]
    tube_model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(path_arg):
        print(json.dumps({"error": f"Input path not found: {path_arg}"}, indent=2))
        sys.exit(1)
        
    if not os.path.exists(multiclass_model_path):
        print(json.dumps({"error": f"Multi-class model file not found: {multiclass_model_path}"}, indent=2))
        sys.exit(1)
        
    # Lazy load YOLO to ensure fast error reporting if imports fail
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(json.dumps({"error": f"Failed to import ultralytics: {str(e)}"}, indent=2))
        sys.exit(2)
        
    # Load multi-class segmentation model
    try:
        model_multi = YOLO(multiclass_model_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load multi-class model: {str(e)}"}, indent=2))
        sys.exit(2)
        
    # Load tube detector model if specified and exists
    model_tube = None
    if tube_model_path:
        if os.path.exists(tube_model_path):
            try:
                model_tube = YOLO(tube_model_path)
            except Exception as e:
                # Log warning or fallback silently
                pass
                
    # Check if path_arg is a directory
    is_dir = os.path.isdir(path_arg)
    
    if is_dir:
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = []
        for f in os.listdir(path_arg):
            if f.lower().endswith(supported_extensions):
                image_files.append(os.path.join(path_arg, f))
        image_files.sort()
        
        if not image_files:
            print(json.dumps({"error": f"No images found in directory: {path_arg}"}, indent=2))
            sys.exit(1)
            
        # Process each image in the directory
        aggregated_results = {}
        for img_path in image_files:
            filename = os.path.basename(img_path)
            print(f"Processing {filename}...", file=sys.stderr)
            res = process_single_image(img_path, model_multi, model_tube)
            aggregated_results[filename] = res
            
        # Print JSON of all results to stdout
        print(json.dumps(aggregated_results, indent=2))
        
        # Save to JSON in the directory
        json_out_path = os.path.join(path_arg, "severity_results.json")
        try:
            with open(json_out_path, "w") as jf:
                json.dump(aggregated_results, jf, indent=2)
            print(f"Results saved to JSON: {json_out_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to write JSON output: {str(e)}", file=sys.stderr)
            
        # Save to CSV in the directory
        csv_out_path = os.path.join(path_arg, "severity_results.csv")
        try:
            import csv
            with open(csv_out_path, "w", newline="") as cf:
                writer = csv.writer(cf)
                # Header
                writer.writerow(["image_name", "clone_index", "healthy_pixels", "affected_pixels", "severity_percentage", "severity_level"])
                # Rows
                for filename, clones in aggregated_results.items():
                    if isinstance(clones, dict) and "error" in clones:
                        writer.writerow([filename, "ERROR", "", "", "", clones["error"]])
                    else:
                        for clone in clones:
                            writer.writerow([
                                filename,
                                clone["clone_index"],
                                clone["healthy_pixels"],
                                clone["affected_pixels"],
                                clone["severity_percentage"],
                                clone["severity_level"]
                            ])
            print(f"Results saved to CSV: {csv_out_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to write CSV output: {str(e)}", file=sys.stderr)
            
    else:
        # Single image prediction
        res = process_single_image(path_arg, model_multi, model_tube)
        print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
