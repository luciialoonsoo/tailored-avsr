import cv2
import numpy as np
import pandas as pd
import os
from insightface.app import FaceAnalysis
from tqdm import tqdm
import subprocess

# --- NUEVA RUTA PARA EVITAR CACHÉ ---
CSV_INPUT = "/home/export/pfc/lalomat/Documents/tfg/csv_real_mtedx.csv"
OUTPUT_DIR = "/home/export/pfc/lalomat/Documents/tfg/recortes_TEST_NUEVOS"

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=-1)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_aligned_mouth(frame):
    if frame is None: return None
    faces = app.get(frame)
    if not faces: return None
    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]

    # Alineación (Transformación de Similitud del paper) [cite: 342]
    eye_l, eye_r = face.kps[0], face.kps[1]
    angle = np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))

    lmk = face.landmark_2d_106
    mouth_pts = lmk[52:72]
    cx, cy = int(np.mean([p[0] for p in mouth_pts])), int(np.mean([p[1] for p in mouth_pts]))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # ZOOM EXTREMO (ROI de labios 96x96) [cite: 340, 343]
    p52, p61 = lmk[52], lmk[61]
    dist_boca = np.linalg.norm(p52 - p61)
    side = int(dist_boca * 0.8) # Zoom para que solo se vea la boca

    crop = rotated[max(0,cy-side):min(rotated.shape[0],cy+side),
                   max(0,cx-side):min(rotated.shape[1],cx+side)]

    if crop.size == 0: return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (96, 96))

def process_video(row):
    # Imprimimos el vídeo real que se está abriendo para diagnosticar
    print(f"\n?? ABRIENDO VÍDEO: {row['video_path']}")
    output_p = os.path.join(OUTPUT_DIR, f"{row['sampleID']}.npy")

    temp_v = f"temp_{row['sampleID']}.mp4"
    duration = float(row['end_time']) - float(row['start_time'])

    subprocess.run(['ffmpeg', '-y', '-ss', str(row['start_time']), '-t', str(duration),
                   '-i', row['video_path'], '-c:v', 'libx264', '-an', temp_v],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(temp_v)
    frames_extraidos = [] # Usamos un nombre claro
    while True:
        ret, frame = cap.read()
        if not ret: break
        mouth = get_aligned_mouth(frame)
        if mouth is not None:
            frames_extraidos.append(mouth)
    cap.release()

    if os.path.exists(temp_v):
        os.remove(temp_v)

    # --- BLOQUE DE GUARDADO CON CHIVATO ---
    if len(frames_extraidos) > 2:
        ruta_absoluta = os.path.abspath(output_p)
        print(f"?? GUARDANDO EN: {ruta_absoluta} | Frames: {len(frames_extraidos)}")
        np.save(output_p, np.array(frames_extraidos))
    else:
        print(f"?? El vídeo {row['sampleID']} falló: solo se sacaron {len(frames_extraidos)} frames.")
if __name__ == "__main__":
    # Cogemos 5 filas totalmente aleatorias [cite: 311, 318]
    df = pd.read_csv(CSV_INPUT)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        process_video(row)
