import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

# ==========================

# LOAD MODEL

# ==========================

ripeness_model = YOLO("models/ripeness.pt")
disease_model = YOLO("models/disease.pt")

# ==========================

# FUNGSI LOG

# ==========================

def add_log(text):
    log_box.insert(tk.END, text + "\n")
    log_box.see(tk.END)
    root.update()

# ==========================

# TAMPILKAN HASIL YOLO

# ==========================

def tampilkan_hasil_deteksi(results):

    img = results[0].plot()

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(img)

    img.thumbnail((800, 600))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

# ==========================

# PILIH GAMBAR

# ==========================

def pilih_gambar():

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if not file_path:
        return

    proses_ai(file_path)

# ==========================

# PROSES AI

# ==========================

def proses_ai(path):

    log_box.delete(1.0, tk.END)

    kematangan_var.set("-")
    penyakit_var.set("-")

    add_log("📷 Gambar dipilih")
    add_log("🤖 Menjalankan model kematangan...")

    # ==================
    # MODEL KEMATANGAN
    # ==================
    ripeness_results = ripeness_model.predict(
        path,
        conf=0.25,
        verbose=False
    )

    if len(ripeness_results[0].boxes) > 0:

        box = ripeness_results[0].boxes[0]

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = ripeness_model.names[cls]

        kematangan_var.set(
            f"{label} ({conf:.2%})"
        )

        add_log(
            f"✅ Kematangan: {label} ({conf:.2%})"
        )

    else:

        kematangan_var.set("Tidak terdeteksi")

        add_log(
            "❌ Kematangan tidak terdeteksi"
        )

# ==================
# MODEL PENYAKIT
# ==================
    add_log("🤖 Menjalankan model penyakit...")

    disease_results = disease_model.predict(
        path,
        conf=0.25,
        verbose=False
    )

    if len(disease_results[0].boxes) > 0:

        box = disease_results[0].boxes[0]

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = disease_model.names[cls]

        penyakit_var.set(
            f"{label} ({conf:.2%})"
        )

        add_log(
            f"✅ Penyakit: {label} ({conf:.2%})"
        )

    else:

        penyakit_var.set("Tidak terdeteksi")

        add_log(
            "❌ Penyakit tidak terdeteksi"
        )

    tampilkan_hasil_deteksi(
        disease_results
    )

    add_log("🎉 Analisis selesai")

# ==========================

# GUI

# ==========================

root = tk.Tk()

root.title("Sistem Klasifikasi Tomat")
root.geometry("1400x800")

# ==========================

# FRAME UTAMA

# ==========================

main_frame = tk.Frame(root)
main_frame.pack(
fill="both",
expand=True,
padx=10,
pady=10
)

# ==========================

# PANEL GAMBAR

# ==========================

left_frame = tk.LabelFrame(
main_frame,
text="Hasil Deteksi AI",
font=("Arial", 14, "bold")
)

left_frame.grid(
row=0,
column=0,
rowspan=2,
sticky="nsew",
padx=(0, 10)
)

image_label = tk.Label(
left_frame,
bg="lightgray"
)

image_label.pack(
fill="both",
expand=True,
padx=10,
pady=10
)

btn = tk.Button(
left_frame,
text="Pilih Gambar",
font=("Arial", 12, "bold"),
command=pilih_gambar
)

btn.pack(
pady=10
)

# ==========================

# PANEL LOG

# ==========================

process_frame = tk.LabelFrame(
main_frame,
text="Proses AI",
font=("Arial", 14, "bold")
)

process_frame.grid(
row=0,
column=1,
sticky="nsew"
)

log_box = scrolledtext.ScrolledText(
process_frame,
width=50,
height=18
)

log_box.pack(
fill="both",
expand=True,
padx=10,
pady=10
)

# ==========================

# PANEL HASIL

# ==========================

result_frame = tk.LabelFrame(
main_frame,
text="Hasil Analisis",
font=("Arial", 14, "bold")
)

result_frame.grid(
row=1,
column=1,
sticky="nsew",
pady=(10, 0)
)

kematangan_var = tk.StringVar(value="-")
penyakit_var = tk.StringVar(value="-")

tk.Label(
result_frame,
text="Kematangan",
font=("Arial", 12, "bold")
).grid(row=0, column=0, sticky="w", padx=10, pady=10)

tk.Label(
result_frame,
textvariable=kematangan_var,
font=("Arial", 12)
).grid(row=0, column=1, sticky="w")

tk.Label(
result_frame,
text="Penyakit",
font=("Arial", 12, "bold")
).grid(row=1, column=0, sticky="w", padx=10, pady=10)

tk.Label(
result_frame,
textvariable=penyakit_var,
font=("Arial", 12)
).grid(row=1, column=1, sticky="w")

# ==========================

# RESPONSIVE

# ==========================

main_frame.columnconfigure(0, weight=3)
main_frame.columnconfigure(1, weight=2)

main_frame.rowconfigure(0, weight=3)
main_frame.rowconfigure(1, weight=2)

root.mainloop()
