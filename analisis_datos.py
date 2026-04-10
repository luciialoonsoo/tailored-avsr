import numpy as np
import os

NPY_DIR = "/home/export/pfc/lalomat/muavic/data/muavic/es/video_recortado/train/"

def analizar():
    archivos = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy')]
    if not archivos:
        print("? No hay archivos para analizar.")
        return

    # Analizamos los 3 primeros archivos generados por el nuevo script
    for nombre in archivos[:3]:
        data = np.load(os.path.join(NPY_DIR, nombre))

        # Estadísticas básicas
        mean_val = np.mean(data)
        std_val = np.std(data)
        max_val = np.max(data)
        min_val = np.min(data)

        # Diferencia entre frames (Movimiento real vs Ruido)
        # Si es ruido, la diferencia entre frames será casi igual a la imagen original
        diff = np.mean(np.abs(data[1:] - data[:-1]))

        print(f"\n--- Archivo: {nombre} ---")
        print(f"Brillo medio: {mean_val:.2f} (0=Negro, 255=Blanco)")
        print(f"Contraste (Std): {std_val:.2f}")
        print(f"Rango: [{min_val} - {max_val}]")
        print(f"Variación temporal (Movimiento): {diff:.2f}")

        if max_val == min_val:
            print("?? RESULTADO: El vídeo es un bloque de color sólido (vacío).")
        elif std_val < 2:
            print("?? RESULTADO: Imagen casi plana, no hay formas.")
        elif diff > 50:
            print("?? RESULTADO: Demasiado cambio entre frames. Es RUIDO puro.")
        else:
            print("? RESULTADO: Hay estructura y movimiento coherente. ¡Son datos reales!")

if __name__ == "__main__":
    analizar()
