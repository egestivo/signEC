import cv2

# Abre la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Aquí puedes agregar el procesamiento de detección de objetos

    # Mostrar el frame
    cv2.imshow("Frame", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Libera la cámara
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV
