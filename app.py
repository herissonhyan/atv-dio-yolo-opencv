import cv2
import numpy as np
import pyttsx3

# Inicializando o sintetizador de voz
engine = pyttsx3.init()

# Caminhos para os arquivos do YOLO
yolo_weights = "yolov3-tiny.weights"  # Use o arquivo correto para o modelo Tiny
yolo_config = "yolov3.cfg"       # Use o arquivo de configuração correto
classes_file = "coco.names"

# Carregar nomes das classes
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar a rede YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Configurar as camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Iniciar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Pré-processar a imagem para o YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informações de objetos detectados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        print(len(out))
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Exibir as detecções
    detected_classes = set()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_classes.add(label)
            confidence = confidences[i]

            # Desenhar a caixa delimitadora e o rótulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    for detected_class in detected_classes:
        engine.say(f"Detected: {detected_class}")
        engine.runAndWait()

    # Exibir o feed da webcam
    cv2.imshow("YOLO Object Detection", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
