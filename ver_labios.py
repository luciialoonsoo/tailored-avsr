# import numpy as np
# import cv2
# import os
# import glob
#
# # 1. Buscar el archivo más reciente
# files = glob.glob('/home/export/pfc/lalomat/Documents/tfg/videos_finales/*.npy')
# if not files:
#     print("No hay archivos .npy para visualizar.")
# else:
#     archivo = files[0] # Usamos el primero que encuentre
#     data = np.load(archivo)
#
#     # data tiene forma (Frames, 96, 96)
#     # Extraemos el frame medio del clip para ver movimiento
#     frame_idx = len(data) // 2
#     frame = data[frame_idx]
#     # Añade esta línea en ver_labios.py antes de guardar
#     frame_normalizado = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
#     cv2.imwrite("check_labios_claro.png", frame_normalizado)
#     # Guardar como imagen PNG
#     nombre_salida = "check_labios.png"
#     cv2.imwrite(nombre_salida, frame)
#
#     print(f"--- Visualización de: {os.path.basename(archivo)} ---")
#     print(f"Frame extraído: {frame_idx} de {len(data)}")
#     print(f"Imagen guardada como: {os.getcwd()}/{nombre_salida}")
#     print("Descarga este archivo a tu ordenador para verlo.")

import numpy as np
import cv2
import dlib
import os
import glob

# Rutas (Asegúrate de que coinciden con las tuyas)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = "/home/export/pfc/lalomat/Documents/tfg/videos_finales"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 1. Cargar el primer .npy generado
files = glob.glob(os.path.join(OUTPUT_DIR, "*.npy"))
if not files:
    print("No hay archivos .npy para analizar.")
    exit()

data = np.load(files[0])
frame = data[len(data)//2] # Frame central

# 2. Aumentar contraste para ver algo
frame_brillante = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

# 3. Intentar detectar puntos sobre este recorte (solo para ver si hay formas)
# Nota: Aquí solo queremos ver la imagen con brillo
cv2.imwrite("DIAGNOSTICO_LABIOS.png", frame_brillante)

print(f"--- Diagnóstico ---")
print(f"Archivo analizado: {os.path.basename(files[0])}")
print(f"Media original: {np.mean(frame):.2f}")
print(f"Imagen guardada: DIAGNOSTICO_LABIOS.png")
