import os
from ultralytics import YOLO

model = YOLO("aug300.pt")

# Kumpul semua imej secara rekursif dari folder dan subfolder
source_dir = r"D:\FGV\Segmentation\Prediction\segmentation\segmentation_results"
supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
image_paths = []

if os.path.isdir(source_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))
elif os.path.isfile(source_dir):
    if source_dir.lower().endswith(supported_extensions):
        image_paths.append(source_dir)

if image_paths:
    # Jalankan prediksi secara pukal pada semua imej yang dijumpai
    model.predict(source=image_paths, show=False, save=True, show_conf=False)
else:
    print(f"Tiada imej dijumpai dalam laluan: {source_dir}")