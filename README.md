# YOLO Segmentation & Severity Prediction Pipeline

This repository contains a pipeline for training, predicting, and tracking using YOLO segmentation models. It also includes an active learning framework to iteratively select informative images for training.

---

## 📂 Directory Structure & Setup

To operate the pipeline, organize your files as follows:

```text
Prediction/
│
├── models/                           <- Put all your YOLO model weight files (.pt) here
│   ├── best1.pt
│   ├── best2.pt
│   └── best3.pt
│
├── segmentation/
│   └── segmentation_results/         <- Place images to be predicted/evaluated here
│
├── active_learning/                  <- Active learning folder (created & managed by pipeline)
│   ├── history/                      <- Stores history round JSONs
│   ├── annotated/                    <- Annotated active learning images/labels
│   ├── selected/                     <- Selected images for annotation
│   └── unlabeled_pool/               <- Pool of unlabeled images to choose from
│
├── images/                           <- (Optional) Root dataset images
├── labels/                           <- (Optional) Root dataset labels
├── train/                            <- (Optional) Training dataset split
├── val/                              <- (Optional) Validation dataset split
│
├── active_learning_config.yaml       <- Active learning configuration
├── data_custom.yaml                  <- YOLO training dataset configuration (e.g., classes, paths)
├── data_multiclass.yaml              <- YOLO multi-class dataset configuration
│
├── active_learning.py                <- Uncertainty-based active learning loop runner
├── train.py                          <- Script to train/fine-tune a YOLO model
├── predict.py                        <- Run predictions with a single YOLO model
├── predict_multi_model.py            <- Run predictions sequentially with multiple models in models/
└── predict_true_individual_severity.py <- Calculate plant disease severity in 8-tube rack slot-by-slot
```

---

## 🛠️ Installation & Requirements

Ensure you have the required Python packages installed. Run the following command in your environment:

```powershell
pip install ultralytics opencv-python numpy pyyaml
```

---

## 🚀 How to Operate

### 1. Sequential Multi-Model Predictions
Use this script if you have multiple YOLO models and want to predict on the exact same dataset sequentially to compare results.

1. Create a `models/` directory in the project root if it does not exist.
2. Put your `.pt` model files (e.g. `best1.pt`, `best2.pt`) inside the `models/` directory.
3. Put your input images inside the `segmentation/segmentation_results/` directory.
4. Run:
   ```powershell
   python predict_multi_model.py
   ```
* **Output:** Results are saved in individual subfolders named after each model under the `runs/` directory (e.g. `runs/segment/best1/`, `runs/segment/best2/`).

---

### 2. Single Model Predictions
To predict using a single model on your image dataset:

1. Put your model file in the root workspace or specify its name in [predict.py](file:///d:/pipeline/Segmentation/Prediction/predict.py) (e.g., `model = YOLO("best1.pt")`).
2. Run:
   ```powershell
   python predict.py
   ```
* **Output:** Saved under `runs/segment/predict/`.

---

### 3. Individual Clone Disease Severity Tracking
This script tracks disease severity (affected pixels vs. healthy pixels) individually for up to 8 plant clones arranged in a test tube rack.

* **Usage:**
  ```powershell
  python predict_true_individual_severity.py <image_path_or_directory> <multiclass_model_path> [tube_model_path]
  ```
  * `<image_path_or_directory>`: Path to a single image or a folder of images.
  * `<multiclass_model_path>`: Path to your YOLO multi-class plant disease segmentation model (identifying healthy/affected leaves).
  * `[tube_model_path]` (Optional): Path to a single-class tube detector model (e.g., `sfs104.pt`) to locate the 8 tubes. If not provided, it falls back to a programmatic grid.

* **Output:** Prints detailed results to the console (JSON format) and saves aggregate outputs as `severity_results.json` and `severity_results.csv` in the target directory.

---

### 4. Training a YOLO Model
To train/fine-tune a model on your dataset:

1. Set up your dataset configurations in `data_custom.yaml`.
2. Run:
   ```powershell
   python train.py
   ```
* **Output:** Checkpoints and logs will be saved to `segmentation/yolo26m-seg/`.

---

### 5. Uncertainty-Based Active Learning Loop
Iteratively select the most informative/uncertain images from an unlabeled pool to annotate and retrain models.

* **Supported Actions:**
  * **Score:** Calculate uncertainty scores for unlabeled images.
    ```powershell
    python active_learning.py score <model_path>
    ```
  * **Select:** Choose the top `K` images with the highest uncertainty for manual labeling.
    ```powershell
    python active_learning.py select <model_path> --k 5
    ```
  * **Integrate:** Move newly annotated images from the annotation directory to the training pool.
    ```powershell
    python active_learning.py integrate
    ```
  * **Retrain:** Retrain the model on the expanded dataset.
    ```powershell
    python active_learning.py retrain <model_path>
    ```
  * **Full Cycle:** Execute scoring, selection, integration, and retraining in a single run.
    ```powershell
    python active_learning.py full <model_path> --k 5
    ```
