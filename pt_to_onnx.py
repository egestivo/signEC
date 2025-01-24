from ultralytics import YOLO

# Carga el modelo entrenado
model = YOLO('best.pt')

# Convierte el modelo a ONNX
model.export(format='onnx')  # Esto exporta el modelo a un archivo .onnx
