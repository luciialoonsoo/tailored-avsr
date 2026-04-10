import pandas as pd
import cv2
import dlib
import numpy as np
import os
import subprocess

# --- RUTAS ---
TFG_PATH = "/home/export/pfc/lalomat/Documents/tfg"
CSV_PATH = os.path.join(TFG_PATH, "csv_limpio.csv")
PREDICTOR_PATH = os.path.join(TFG_PATH, "shape_predictor_68_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def extraer_con_puente_ffmpeg():
    df = pd.read_csv(CSV_PATH)
    # Probamos con el segundo vídeo (Helena hablando)
    video_input = df.iloc[1]['video_path']
    temp_frame = "frame_extraido.png"

    print(f"Extrayendo frame de: {video_input} usando FFmpeg...")

    # 1. Forzamos a FFmpeg a sacar el frame 100 (segundo ~4)
    # -ss 4 (ir al segundo 4), -frames:v 1 (sacar 1 imagen)
    cmd = [
        'ffmpeg', '-y', '-ss', '4', '-i', video_input,
        '-frames:v', '1', '-q:v', '2', temp_frame
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(temp_frame):
        print("? FFmpeg no pudo crear la imagen. ¿La ruta del vídeo es correcta?")
        return

    # 2. Leer la imagen que FFmpeg SÍ ha podido crear
    frame = cv2.imread(temp_frame)
    if frame is None or np.mean(frame) < 1.0:
        print("? La imagen sigue saliendo negra. El archivo de vídeo original podría estar dañado.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bright = cv2.equalizeHist(gray) # Iluminamos por si acaso

    # 3. Detección con Dlib
    faces = detector(gray_bright, 1)

    if len(faces) == 0:
        cv2.imwrite("REVISAR_ESTO.png", gray_bright)
        print("? No hay cara en el segundo 4. Mira 'REVISAR_ESTO.png' en VNC.")
        return

    # 4. Puntos de la boca y recorte
    shape = predictor(gray_bright, faces[0])
    x_pts = [shape.part(i).x for i in range(48, 68)]
    y_pts = [shape.part(i).y for i in range(48, 68)]
    cx, cy = int(np.mean(x_pts)), int(np.mean(y_pts))

    crop = gray[max(0,cy-48):cy+48, max(0,cx-48):cx+48]
    crop = cv2.resize(crop, (96, 96))

    # 5. Guardar resultados
    cv2.imwrite("LABIOS_FINAL.png", crop)
    print("? ¡POR FIN! Mira 'LABIOS_FINAL.png' en tu VNC.")

    # Limpieza
    if os.path.exists(temp_frame): os.remove(temp_frame)

extraer_con_puente_ffmpeg()
