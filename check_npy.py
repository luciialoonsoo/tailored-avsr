import numpy as np
import os
import glob
import cv2

# --- CONFIGURACIÓN DE RUTAS ---
# Esta es la carpeta donde se están guardando tus nuevos archivos .npy
NPY_DIR = "/home/export/pfc/lalomat/Documents/tfg/videos_finales_buenos"

# 1. Buscamos todos los archivos .npy en la carpeta
files = glob.glob(os.path.join(NPY_DIR, "*.npy"))

if not files:
    print(f"? Error: No se han encontrado archivos .npy en: {NPY_DIR}")
    print("Asegúrate de que el script de extracción esté funcionando correctamente.")
else:
    # Seleccionamos el archivo más reciente (el último que se ha creado)
    latest_file = max(files, key=os.path.getctime)
    print(f"\n--- ?? ANALIZANDO ARCHIVO: {os.path.basename(latest_file)} ---")

    try:
        # 2. Cargamos la matriz de frames
        data = np.load(latest_file)
        n_frames = len(data)

        # Mostramos información técnica en la terminal
        print(f"?? Ruta completa: {latest_file}")
        print(f"?? Dimensiones: {data.shape} (Frames, Alto, Ancho)")
        print(f"?? Brillo medio (Media): {np.mean(data):.2f}")
        print(f"?? Contraste (Std Dev): {np.std(data):.2f}")

        # 3. GENERAR IMÁGENES DE VERIFICACIÓN
        # Vamos a guardar el primer frame, el del medio y el último para comparar
        print("\n?? Generando fotogramas de prueba para el VNC...")

        # Frame inicial
        cv2.imwrite("verif_INICIO.png", data[0])
        # Frame intermedio
        cv2.imwrite("verif_MEDIO.png", data[n_frames // 2])
        # Frame final
        cv2.imwrite("verif_FINAL.png", data[-1])

        print(f"   ? Creado: verif_INICIO.png")
        print(f"   ? Creado: verif_MEDIO.png")
        print(f"   ? Creado: verif_FINAL.png")

        print("\n? ¡TERMINADO!")
        print("-" * 40)
        print("?? PASO SIGUIENTE: Abre el VNC y compara 'verif_INICIO.png' con 'verif_FINAL.png'.")
        print("?? Si la boca ha cambiado de posición o forma, ¡EL SCRIPT FUNCIONA PERFECTO!")
        print("-" * 40)

    except Exception as e:
        print(f"? Error crítico al leer el archivo: {e}")
