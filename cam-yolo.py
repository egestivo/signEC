from ultralytics import YOLO
import cv2

# Carga el modelo
model = YOLO('best.pt')  # 'best.pt' es tu archivo de pesos

# Realiza la detección de objetos usando la cámara
cap = cv2.VideoCapture(0) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ancho del frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto del frame

while True:
    ret, frame = cap.read()  # Lee el frame de la cámara
    if not ret:
        break

    # Predice usando el modelo
    results = model(frame)  # Obtiene las predicciones del modelo

    # Renderiza las detecciones en el frame
    frame_with_detections = results[0].plot()  # Usa el primer resultado y dibuja las detecciones

    # Muestra el frame con las detecciones
    cv2.imshow("Detections", frame_with_detections)

    # Sale si presionas 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
