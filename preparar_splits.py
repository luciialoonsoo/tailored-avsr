import pandas as pd
import os
import wave
import glob

base_datos = "/home/export/pfc/lalomat/muavic/data/muavic/es"
output_dir = "/home/export/pfc/lalomat/Documents/tfg/tailored-avsr/splits/training/speaker-dependent"
os.makedirs(output_dir, exist_ok=True)

def generar_csv_real(nombre_es, nombre_salida, subcarpeta):
    path_es = os.path.join(base_datos, nombre_es)

    # 1. Leer todas las frases del archivo .es
    with open(path_es, 'r', encoding='utf-8') as f:
        frases = [line.strip().upper() for line in f.readlines()]

    # 2. Escanear la carpeta de audio REAL (buscando todos los .wav)
    print(f"--- Escaneando archivos reales en {subcarpeta} ---")
    # Buscamos en audio/train/*/*.wav
    patron = os.path.join(base_datos, "audio", subcarpeta, "*", "*.wav")
    archivos_reales = sorted(glob.glob(patron))

    if not archivos_reales:
        # Probamos sin la carpeta intermedia por si acaso
        patron = os.path.join(base_datos, "audio", subcarpeta, "*.wav")
        archivos_reales = sorted(glob.glob(patron))

    print(f"Encontrados {len(archivos_reales)} archivos .wav en el disco.")

    video_paths, audio_paths, nframes_list, transcriptions = [], [], [], []

    # 3. Emparejar (MuAViC suele mantener el orden del TSV y el archivo .es)
    # Usamos el mínimo para no salirnos de rango
    limite = min(len(archivos_reales), len(frases))

    for i in range(limite):
        a_path = archivos_reales[i]
        # Cambiamos /audio/ por /video/ y .wav por .mp4
        v_path = a_path.replace("/audio/", "/video/").replace(".wav", ".mp4")

        try:
            with wave.open(a_path, 'rb') as f:
                nframes = int((f.getnframes() / f.getframerate()) * 100)
                audio_paths.append(a_path)
                video_paths.append(v_path)
                nframes_list.append(nframes)
                transcriptions.append(frases[i])
        except:
            continue

        if (i+1) % 20000 == 0:
            print(f"Procesados {i+1}...")

    df_final = pd.DataFrame({
        "video_path": video_paths,
        "audio_path": audio_paths,
        "nframes": nframes_list,
        "transcription": transcriptions
    })

    df_final.to_csv(os.path.join(output_dir, f"{nombre_salida}.csv"), index=False)
    print(f"? EXITO REAL: {nombre_salida}.csv con {len(df_final)} lineas.")

# Ejecutar escaneando el disco directamente
generar_csv_real("train.es", "muavic-spanish", "train")
generar_csv_real("valid.es", "muavic-spanish-valid", "valid")
generar_csv_real("test.es", "muavic-spanish-test", "test")
