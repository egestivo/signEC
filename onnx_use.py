import cv2
from ultralytics import YOLO

# Cargar el modelo YOLO sin usar half()
model = YOLO("best.pt")

# Abrir la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar a una resolución más pequeña (ejemplo 320x320)
    frame_resized = cv2.resize(frame, (320, 320))

    # Realizar la inferencia
    results = model(frame_resized)

    # Mostrar resultados
    results[0].plot()

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
