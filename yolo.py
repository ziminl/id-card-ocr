


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



net = cv2.dnn.readNetFromDarknet("yolov.cfg", "yolov.weights")



layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



image = cv2.imread("image.jpg")



height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Minimum confidence threshold
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

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        roi = image[y:y + h, x:x + w]
        cv2.imwrite("id_card.jpg", roi)
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.imshow(pil_image)
plt.axis('off')
#plt.show()



id_card_image = Image.open("id_card.jpg")



plt.imshow(id_card_image)
plt.axis('off')
plt.show()


