# COCO, COmmon Objects in Context, är ett populärt dataset för bl.a. object detection, segmentation och key-point detection.
# YOLO, You Only Look Once, är en populär object detection model. YOLO är snabb och träffsäker.

# För detta program behöver vi:
# COCO datasetet, ladda ner filen coco.names från Github: https://github.com/pjreddie/darknet/blob/master/data/coco.names
# YOLOs förtränade modell i form av dess modell/neuron-vikter och konfigurationsfil. 
# Ladda ner dessa på Github och Kaggle: 
# yolov3.cfg : https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg (Skapa en egen yolov3.cfg-fil och kopiera in innehållet från Github-filen)
# yolov3.weights : https://www.kaggle.com/datasets/shivam316/yolov3-weights (Kräver ett Kaggle-konto)
# En test-bild "sample_image.jpg" att testa object detection på. Hitta nånstans på nätet eller ta eget foto.
# Pythons OpenCV-bibliotek, opencv-python. Installeras med `pip install opencv-python`
import cv2
import numpy as np

# Ladda in pre-trained YOLO modell-vikter och config-fil
# Du behöver ha filerna `yolov3.weights` och `yolov3.cfg` nedladdade i samma mapp som programmet
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Config-filen definierar YOLO-nätverkets arkitektur och parametrar. Beskriver hur modellens lager ska struktureras, hur många filter ska användas, kernel size, etc.

# Ladda in COCO-datasetet, som YOLO-modellen är tränad på
# Varje rad i 'coco.names' motsvarar ett class name/kategori
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Ladda in en bild att testa object detection på
image = cv2.imread("sample_image.jpg")

max_display_width = 800
max_display_height = 800
original_height, original_width = image.shape[:2]
scaling_factor = min(max_display_width / original_width, max_display_height / original_height)
new_width = int(original_width * scaling_factor)
new_height = int(original_height * scaling_factor)

target_size = (new_width, new_height)

resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# Förbered bilden för YOLO-modellen genom att konvertera till en "blob"
# Blob är en 4D-bild med 4 egenskaper (batch size, channels, height, width). Modellen förväntar sig bilddata på detta format.
blob = cv2.dnn.blobFromImage(resized_image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Ta fram namnen på YOLOs Output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Kör en forward pass (d.v.s kör input igenom nätverket, modellen gör detections) och få output
detections = net.forward(output_layers)

# Skapa listor som ska hålla i object detection results
boxes = []  # Koordinater för bounding boxes. Varje box kommer representeras som en lista med plats (x, y), och storlek (w, h): [x, y, w, h]
confidences = [] # Confidence scores för varje detection, d.v.s hur mycket vi ska lita på de olika upptäckterna av objekt
class_ids = [] # Klass-IDn för varje detected object

confidence_threshold = 0.7 # Filtrera ut detections med mindre sannolikhet än detta värde
nms_threshold = 0.3 # Lägre värde ger färre överlappande boxar

# Gå igenom varje upptäckt object
for detection in detections:
    for obj in detection:
        scores = obj[5:] # De första 5 element är information om bounding boxen, resten är sannolikheten för olika klasser (d.v.s vilket typ av objekt vi hittat)
        class_id = np.argmax(scores)    # Ta fram den klass/kategori/typ av objekt som har högst score (högst sannolikhet att stämma)
        confidence = scores[class_id]   # Ta fram konfidensen för denna klass

        if confidence > confidence_threshold: # Filtrera ut svaga detections
            # Ta fram bounding boxens koordinater i förhållande till bildens dimensioner/storlek
            center_x = int(obj[0] * new_width)
            center_y = int(obj[1] * new_height)
            w = int(obj[2] * new_width)
            h = int(obj[3] * new_height)

            # Beräkna bounding boxens övre vänstra (top-left) hörn.
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Lägg til resultatet till listorna
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Applicera Non-Maximum Suppression (NMS) för att eliminera överflödiga överlappande boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=confidence_threshold, nms_threshold=nms_threshold)

# Rita bounding boxes and labels på den ursprungliga bilden
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = (0, 255, 0) # Vill att bounding boxes ska ha grön färg
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(resized_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Visa den resulterande bilden
cv2.imshow("Object Detection", resized_image)
cv2.waitKey(0) # Väntar på knapp-tryck innan fönstret stängs
cv2.destroyAllWindows()
