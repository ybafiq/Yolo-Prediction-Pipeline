import os
from ultralytics import YOLO

# Folder yang mengandungi semua model (.pt)
models_dir = r"D:\FGV\Segmentation\Prediction\models"

# Folder imej untuk prediksi
source_dir = r"D:\FGV\Segmentation\Prediction\segmentation\segmentation_results"
supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
image_paths = []

# Kumpul semua imej secara rekursif dari folder dan subfolder
if os.path.isdir(source_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))
elif os.path.isfile(source_dir):
    if source_dir.lower().endswith(supported_extensions):
        image_paths.append(source_dir)

if not image_paths:
    print(f"Tiada imej dijumpai dalam laluan: {source_dir}")
    exit(1)

# Buat folder models jika belum wujud
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Folder '{models_dir}' telah dicipta. Sila letakkan fail model (.pt) anda di dalamnya.")

# Cari semua fail model (.pt) dalam folder models_dir
model_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.pt')]

if not model_files:
    print(f"\nTiada fail model (.pt) dijumpai dalam: {models_dir}")
    print("Sila letakkan fail model (.pt) anda (contohnya: best1.pt, aug300.pt) ke dalam folder tersebut.")
    exit(0)

print(f"Dijumpai {len(model_files)} model untuk prediksi:")
for idx, model_file in enumerate(model_files, 1):
    print(f"  {idx}. {model_file}")

# Jalankan prediksi untuk setiap model satu demi satu
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model_name = os.path.splitext(model_file)[0]
    
    print("\n" + "="*60)
    print(f"Memulakan prediksi menggunakan model: {model_file}")
    print("="*60)
    
    try:
        # Load model YOLO
        model = YOLO(model_path)
        
        # Jalankan prediksi secara pukal pada semua imej yang dijumpai
        # Keputusan akan disimpan dalam folder runs/<task>/<nama_model> (contoh: runs/segment/best1)
        model.predict(
            source=image_paths, 
            show=False, 
            save=True, 
            show_conf=False,
            name=model_name,      # Namakan folder output mengikut nama model
            exist_ok=True         # Jika folder wujud, simpan dalam folder yang sama tanpa cipta folder baru (cth: best12, best13)
        )
        print(f"Selesai prediksi untuk model: {model_file}")
    except Exception as e:
        print(f"Ralat berlaku semasa memproses model {model_file}: {e}")

print("\n" + "="*60)
print("Semua model telah selesai diproses!")
print("="*60)