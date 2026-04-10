import os
import glob
import pandas as pd
import re

# RUTAS
VTT_DIR = "/home/export/pfc/lalomat/muavic/data/mtedx/es-es/data/train/vtt/"
VIDEO_DIR = "/home/export/pfc/lalomat/muavic/data/mtedx/video/es/train/"
OUTPUT_CSV = "/home/export/pfc/lalomat/Documents/tfg/csv_real_mtedx.csv"

def vtt_to_seconds(t_str):
    """Convierte formatos como 00:00:42.500 o 00:42.500 a segundos."""
    try:
        t_str = t_str.strip()
        parts = t_str.split(':')
        if len(parts) == 3: # Formato HH:MM:SS.mmm
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2: # Formato MM:SS.mmm
            m, s = parts
            return int(m) * 60 + float(s)
    except Exception:
        return None
    return None

data = []
vtt_files = glob.glob(os.path.join(VTT_DIR, "*.vtt"))

print(f"Procesando {len(vtt_files)} archivos VTT...")

for vtt_path in vtt_files:
    # El archivo se llama 00Irv6pwJXQ.es.vtt, queremos 00Irv6pwJXQ
    video_id = os.path.basename(vtt_path).split('.')[0]
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    if not os.path.exists(video_path):
        continue

    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if "-->" in lines[i]:
            try:
                # Limpiar la línea de tiempos de posibles etiquetas extras
                clean_line = re.sub(r'<[^>]+>', '', lines[i])
                times = clean_line.strip().split(" --> ")

                # Cogemos solo la parte del tiempo (ignorando configuraciones de posición VTT)
                start_t = times[0].split()[0]
                end_t = times[1].split()[0]

                start = vtt_to_seconds(start_t)
                end = vtt_to_seconds(end_t)

                if start is not None and end is not None:
                    # Buscamos la frase en las siguientes líneas que no estén vacías
                    transcription = ""
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if lines[j].strip() and "-->" not in lines[j]:
                            transcription = lines[j].strip()
                            break

                    data.append({
                        'video_path': video_path,
                        'start_time': start,
                        'end_time': end,
                        'transcription': transcription,
                        'sampleID': f"{video_id}_{i}"
                    })
            except:
                continue

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"? ¡CONSEGUIDO! CSV creado con {len(df)} segmentos.")
print(f"Ruta: {OUTPUT_CSV}")
