from ultralytics import YOLO

# Load model
ripeness_model = YOLO("models/ripeness.pt")
disease_model = YOLO("models/disease.pt")

# Gambar yang akan diuji
image = "test_image/image.png"

# Prediksi
ripeness_results = ripeness_model.predict(
    image,
    conf=0.25,
    verbose=False
)

disease_results = disease_model.predict(
    image,
    conf=0.25,
    verbose=False
)

print("\n===== HASIL DETEKSI =====\n")

# Kematangan
if len(ripeness_results[0].boxes) > 0:
    box = ripeness_results[0].boxes[0]

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    print("KEMATANGAN")
    print(f"Label      : {ripeness_model.names[cls]}")
    print(f"Confidence : {conf:.2%}")
else:
    print("KEMATANGAN : Tidak terdeteksi")

print()

# Penyakit
if len(disease_results[0].boxes) > 0:
    box = disease_results[0].boxes[0]

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    print("PENYAKIT")
    print(f"Label      : {disease_model.names[cls]}")
    print(f"Confidence : {conf:.2%}")
else:
    print("PENYAKIT : Tidak terdeteksi")