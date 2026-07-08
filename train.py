from ultralytics import YOLO

def main():
    # 1. Load your specific base segmentation model weights
    model = YOLO("D:\FGV\Segmentation\Prediction\yolo26m-seg.pt")

    # 2. Train the model
    model.train(
        data="data_custom.yaml",  # Path to your config file
        epochs=10,                  # Adjust based on dataset size and time
        imgsz=640,                   # Standard YOLO resolution
        batch=6,                    # Adjust lower (e.g., 8 or 4) if your GPU runs out of VRAM
        workers=2,                   # Number of CPU data loading workers
        device="cpu",                    # Use 0 for CUDA GPU, or 'cpu' if you don't have a dedicated GPU
        project="segmentation", # Saves results to a folder with this name
        name="yolo26m-seg"      # Subfolder name for this specific run
    )

if __name__ == "__main__":
    main()