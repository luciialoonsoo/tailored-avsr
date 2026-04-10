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
PREDICTOR_PATH = os.path.join(TFG_PATH, "shape_predictor_68_face_landmarks.dat")

# Inicializar los "ojos" de la IA
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract_mouth(gray_frame):
    """Busca los labios en un frame en gris y devuelve el recorte 96x96."""
    faces = detector(gray_frame, 1)

    if len(faces) == 0:
        return None

    shape = predictor(gray_frame, faces[0])

    # Puntos de la boca (48 a 67)
    x_pts = [shape.part(i).x for i in range(48, 68)]
    y_pts = [shape.part(i).y for i in range(48, 68)]
    cx, cy = int(np.mean(x_pts)), int(np.mean(y_pts))

    # Recorte 96x96
    h, w = gray_frame.shape
    crop = gray_frame[max(0, cy-48):cy+48, max(0, cx-48):cx+48]
    if crop.shape != (96, 96):
        crop = cv2.resize(crop, (96, 96))
    return crop

def process_video_with_patience(video_path, output_path):
    """Procesa el vídeo entero buscando una cara con paciencia."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mouth_frames = []

    # Forzamos la búsqueda de cara en varios puntos si falla al principio
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Intentamos primero en el frame 0, si falla, saltamos.
    face_found = False

    # Vamos frame a frame leyendo y procesando
    for f in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Si aún no hemos encontrado cara, lo intentamos en este frame
        if not face_found:
            mouth = extract_mouth(gray)
            if mouth is not None:
                # ¡Lo encontramos! Guardamos este frame y marcamos éxito.
                mouth_frames.append(mouth)
                face_found = True
            else:
                # Si es un clip corto de diapositiva y no hay cara en ningún frame,
                # este bucle terminará con mouth_frames vacío.
                continue
        else:
            # Una vez que la cara ha sido encontrada, la seguimos en los siguientes frames.
            mouth = extract_mouth(gray)
            if mouth is not None:
                mouth_frames.append(mouth)
            # (Opcional: aquí podrías añadir lógica para dejar de buscar si se pierde la cara)

    cap.release()

    if len(mouth_frames) > 0:
        np.save(output_path, np.array(mouth_frames))
        return True
    return False

if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT)
    print(f"Iniciando preprocesamiento de {len(df)} vídeos...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        video_p = row['video_path']
        output_p = os.path.join(OUTPUT_DIR, f"{row['sampleID']}.npy")

        if not os.path.exists(output_p):
            try:
                # Usamos la nueva función con paciencia
                process_video_with_patience(video_p, output_p)
            except Exception as e:
                continue
