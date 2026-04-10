import pandas as pd
import os

folder = "/home/export/pfc/lalomat/Documents/tfg/tailored-avsr/splits"
subfolders = ["training/speaker-dependent", "validation/speaker-dependent", "test/speaker-dependent"]

for sub in subfolders:
    path = os.path.join(folder, sub, "muavic-spanish.csv")
    if os.path.exists(path):
        print(f"Arreglando {path}...")
        df = pd.read_csv(path)

        # Creamos sampleID a partir del nombre del archivo de audio
        # Si la ruta es /.../video/train/ID/NOMBRE.wav, nos quedamos con NOMBRE
        df['sampleID'] = df['audio_path'].apply(lambda x: os.path.basename(x).replace('.wav', ''))

        # Guardamos el CSV actualizado
        df.to_csv(path, index=False)
        print(f"? Añadida columna sampleID a {len(df)} filas.")

print("\n?? ¡Todo listo! Intenta lanzar ./run_avsr.sh otra vez.")
