import pandas as pd
import numpy as np
import os
import cv2
import dlib
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
TFG_PATH = "/home/export/pfc/lalomat/Documents/tfg"
CSV_INPUT = os.path.join(TFG_PATH, "csv_limpio.csv")
OUTPUT_DIR = os.path.join(TFG_PATH, "videos_finales")
# Ruta al archivo de puntos faciales que descargamos
PREDICTOR_PATH = os.path.join(TFG_PATH, "shape_predictor_68_face_landmarks.dat")

# Inicializar los "ojos" de la IA
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract_mouth(frame):
    """Busca la cara y devuelve solo el recorte de la boca 96x96."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None

    # Cogemos la primera cara detectada
    shape = predictor(gray, faces[0])

    # Puntos de la boca (índices 48 a 67 en Dlib)
    x_pts = [shape.part(i).x for i in range(48, 68)]
    y_pts = [shape.part(i).y for i in range(48, 68)]

    # Calculamos el centro de la boca
    cx, cy = int(np.mean(x_pts)), int(np.mean(y_pts))

    # Recorte de 96x96 (48 píxeles hacia cada lado desde el centro)
    h, w = gray.shape
    y1, y2 = max(0, cy-48), min(h, cy+48)
    x1, x2 = max(0, cx-48), min(w, cx+48)

    mouth_crop = gray[y1:y2, x1:x2]

    # Si el recorte no es perfecto por estar cerca del borde, lo forzamos a 96x96
    if mouth_crop.shape != (96, 96):
        mouth_crop = cv2.resize(mouth_crop, (96, 96))

    return mouth_crop

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    mouth_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Intentamos extraer la boca de este frame
        mouth = extract_mouth(frame)
        if mouth is not None:
            mouth_frames.append(mouth)

    cap.release()

    if len(mouth_frames) > 0:
        # Guardamos la secuencia de labios como un archivo .npy
        np.save(output_path, np.array(mouth_frames))
        return True
    return False

if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT)
    print(f"Iniciando extracción de labios para {len(df)} vídeos...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        video_p = row['video_path']
        output_p = os.path.join(OUTPUT_DIR, f"{row['sampleID']}.npy")

        if not os.path.exists(output_p):
            try:
                success = process_video(video_p, output_p)
            except Exception as e:
                continue
