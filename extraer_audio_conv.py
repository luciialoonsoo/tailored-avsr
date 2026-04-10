import pandas as pd
import os
import subprocess
from tqdm import tqdm

# RUTAS ORDENADAS
CSV_INPUT = "/home/export/pfc/lalomat/Documents/tfg/muavic_solo_permisos.csv"
OUTPUT_DIR = "/home/export/pfc/lalomat/Documents/tfg/audios_finales"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extraer_audio(row):
    ruta_video = row['video_path']
    output_wav = os.path.join(OUTPUT_DIR, f"{row['sampleID']}.wav")

    command = [
        'ffmpeg', '-i', ruta_video,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        output_wav, '-y', '-loglevel', 'quiet'
    ]
    try:
        subprocess.run(command, check=True)
        return True
    except:
        return False

if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT)
    print(f"??? Extrayendo Audio en: {OUTPUT_DIR}")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        out_wav = os.path.join(OUTPUT_DIR, f"{row['sampleID']}.wav")
        if not os.path.exists(out_wav):
            extraer_audio(row)
