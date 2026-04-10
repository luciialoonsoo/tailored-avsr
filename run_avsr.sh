#!/bin/bash

# 1. ENTORNO
source venv/bin/activate

# 2. CONFIGURACIÓN BÁSICA
LANGUAGE="spanish"  # "spanish" o "english"
<<<<<<< HEAD
MODO="AVSR"     # "ASR", "VSR" o "AVSR"
=======
MODO="ASR"     # "ASR", "VSR" o "AVSR"
>>>>>>> 2fb41af (Subida limpia de scripts del TFG)
TYPE="tailored" # "tailored" o "conventional"

# 3. SELECCIÓN DE PARTICIÓN (Nueva opción basada en tu imagen)
# -----------------------------------------------------------
SPLIT_TYPE="speaker-dependent"
# SPLIT_TYPE="speaker-independent"

# 4. SELECCIÓN DE DATASET (Elige el nombre del .csv sin la extensión)
# Ejemplos: "liprtve", "vlrf", "muavic-spanish", "lrs3ted", "lrs2bbc"
<<<<<<< HEAD
DATASET="liprtve"
=======
DATASET="muavic-spanish"
>>>>>>> 2fb41af (Subida limpia de scripts del TFG)

# 5. CONSTRUCCIÓN DE RUTAS DE CONFIGURACIÓN (YAML)
if [ "$MODO" == "ASR" ]; then
    CONFIG="configs/ASR/branchformer_transformer+ctc_${LANGUAGE}_${TYPE}.yaml"
elif [ "$MODO" == "VSR" ]; then
    CONFIG="configs/VSR/conv3dresnet18_branchformer_transformer+ctc_${LANGUAGE}_${TYPE}.yaml"
else
    CONFIG="configs/AVSR/${TYPE}_transformer+ctc_${LANGUAGE}.yaml"
fi

LM_CONFIG="configs/LM/lm-${LANGUAGE}.yaml"

# 6. CONSTRUCCIÓN DE RUTAS DE DATOS (Mapeado con tu imagen de 'splits')
# --------------------------------------------------------------------
TRAIN_DATA="./splits/training/${SPLIT_TYPE}/${DATASET}.csv"
VAL_DATA="./splits/validation/${SPLIT_TYPE}/${DATASET}.csv"
TEST_DATA="./splits/test/${SPLIT_TYPE}/${DATASET}.csv"

# 7. SALIDA: AHÍ GUARDAMOS LOS RESULTADOS
OUT_DIR="./resultados/${LANGUAGE}_${MODO}_${TYPE}_${DATASET}"
mkdir -p "$OUT_DIR"

# 8. EJECUCIÓN: VEMOS QUÉ VAMOS A HACER
echo "Lanzando EXPERIMENTO:"
echo "Modalidad: $MODO | Modelo: $TYPE | Dataset: $DATASET | Partición: $SPLIT_TYPE"

python avsr_main.py \
    --mode both \
    --config-file "$CONFIG" \
    --lm-config-file "$LM_CONFIG" \
    --training-dataset "$TRAIN_DATA" \
    --validation-dataset "$VAL_DATA" \
    --test-dataset "$TEST_DATA" \
    --output-dir "$OUT_DIR" \
    --output-name "resultado_${DATASET}" \
    --mask none \
    --snr-target 9999
