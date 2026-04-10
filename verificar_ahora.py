import numpy as np
import cv2
import os
import glob

# LA RUTA QUE NOS HA CONFIRMADO LA TERMINAL
NPY_DIR = "/home/export/pfc/lalomat/Documents/tfg/recortes_TEST_NUEVOS"

archivos = glob.glob(os.path.join(NPY_DIR, "*.npy"))
archivos.sort(key=os.path.getmtime, reverse=True)

if not archivos:
    print(f"? ERROR: Sigo sin ver nada en {NPY_DIR}")
    print("Prueba a listar la carpeta manualmente con: ls " + NPY_DIR)
else:
    print(f"? ¡ENCONTRADOS {len(archivos)} ARCHIVOS!")

if not archivos:
    print("? No hay archivos .npy para verificar. Ejecuta extraer_labios_PRO.py primero.")
else:
    # Cogemos el último archivo generado
    ultimo = max(archivos, key=os.path.getmtime)
    data = np.load(ultimo)

    # Tomamos 5 frames seguidos (del frame 20 al 25 por ejemplo)
    # para ver si la boca se mueve al hablar
    start_f = len(data) // 2
    tira = [data[i] for i in range(start_f, start_f + 5)]

    # Unimos horizontalmente
    resultado = np.hstack(tira)
    cv2.imwrite("PRUEBA_MOVIMIENTO.png", resultado)
    print(f"?? Verificando archivo: {os.path.basename(ultimo)}")
    print("?? 'PRUEBA_MOVIMIENTO.png' creada. ¡Búscala en el VNC!")
