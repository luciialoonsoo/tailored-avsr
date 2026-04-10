import pandas as pd
import numpy as np
import os
import cv2
import subprocess
from tqdm import tqdm

# --- CONFIGURACIÓN ---
CSV_INPUT = "/home/export/pfc/lalomat/Documents/tfg/csv_real_mtedx.csv"
OUTPUT_DIR = "/home/export/pfc/lalomat/Documents/tfg/videos_finales_buenos"

# Usamos los modelos clásicos de OpenCV (Haar Cascades)
# Estos archivos ya existen en tu instalación de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_mouth_crop(frame):
    if frame is None: return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Detectar cara (escalamos un poco para ayudar a la detección)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None

    # Cogemos la cara más grande
    (x, y, w, h) = sorted(faces, key=lambda f: f[2], reverse=True)[0]

    # 2. Estimación anatómica de la boca
    # En una cara estándar, la boca está en el 75% del alto y centrada horizontalmente
    cx = x + (w // 2)
    cy = y + int(h * 0.75)

    # 3. Recorte fijo de 96x96
    # Usamos un radio de 48 para que el total sea 96
    r = 48
    y1, y2 = max(0, cy - r), min(gray.shape[0], cy + r)
    x1, x2 = max(0, cx - r), min(gray.shape[1], cx + r)

    crop = gray[y1:y2, x1:x2]

    # Si el recorte no es 96x96 (por estar en el borde), redimensionamos
    if crop.shape[0] != 96 or crop.shape[1] != 96:
        crop = cv2.resize(crop, (96, 96))

    return crop

def process_segment(row):
    output_p = os.path.join(OUTPUT_DIR, f"{row['sampleID']}.npy")
    if os.path.exists(output_p): return

    duration = float(row['end_time']) - float(row['start_time'])
    temp_video = f"temp_{row['sampleID']}.mp4"

    # Cortamos el trozo con FFmpeg
    cmd = ['ffmpeg', '-y', '-ss', str(row['start_time']), '-t', str(duration),
           '-i', row['video_path'], '-c:v', 'libx264', '-preset', 'ultrafast', '-an', temp_video]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(temp_video): return

    cap = cv2.VideoCapture(temp_video)
    mouth_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        m = get_mouth_crop(frame)
        if m is not None:
            mouth_frames.append(m)
    cap.release()

    if os.path.exists(temp_video): os.remove(temp_video)

    # Si hemos sacado suficientes frames, guardamos
    if len(mouth_frames) > 5:
        np.save(output_p, np.array(mouth_frames))

if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT).head(10) # Probamos con 10
    print(f"?? Procesando con OpenCV Cascades (Sin MediaPipe)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        process_segment(row)
