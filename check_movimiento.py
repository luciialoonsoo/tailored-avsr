import numpy as np
import cv2
import os
import glob

NPY_DIR = "/home/export/pfc/lalomat/Documents/tfg/recortes_TEST_NUEVOS"
archivos = glob.glob(os.path.join(NPY_DIR, "*.npy"))

if archivos:
    # Cogemos el último procesado
    data = np.load(archivos[-1])
    # Seleccionamos 5 frames consecutivos (ej. del 20 al 25)
    indices = range(min(20, len(data)), min(25, len(data)))
    tira = [data[i] for i in indices]

    # Unimos los frames horizontalmente
    resultado = np.hstack(tira)
    cv2.imwrite("PRUEBA_MOVIMIENTO.png", resultado)
    print("?? 'PRUEBA_MOVIMIENTO.png' creada. Si ves los labios moviéndose en secuencia, ¡es perfecto!")
