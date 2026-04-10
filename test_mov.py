import numpy as np
import cv2

# Carga el último archivo que generaste
data = np.load("/home/export/pfc/lalomat/Documents/tfg/recortes_TEST_NUEVOS/V8aoSjGmtOU_1025.npy")

# Restamos el frame 10 al frame 20
diff = cv2.absdiff(data[10], data[20])

# Si hay blanco/gris, hay movimiento
cv2.imwrite("MOVIMIENTO_REAL.png", diff * 5) # Multiplicamos por 5 para ver mejor el cambio
print("?? Imagen 'MOVIMIENTO_REAL.png' creada. Si ves siluetas blancas, ¡la boca se mueve!")
