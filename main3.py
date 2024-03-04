import torch
import torchvision
import wget
from torchvision import transforms as T

import cv2
from PIL import Image

# Download the SSD model
url = 'https://www.dropbox.com/s/8r6f0e05lcpxea6/ssd_mobilenet_v3_large_coco.pth?dl=1'
filename = wget.download(url)


# Load the model
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=91)
checkpoint = torch.load(filename, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

# Define the classes
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Define the transform
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the video
video_capture = cv2.VideoCapture('traffic.mp4')

# Initialize the vehicle counter
vehicle_num = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Transform the frame
    img = Image.fromarray(frame)
    img = transform(img)

    # Detect the objects
    with torch.no_grad():
        pred = model([img])

    # Get the bounding boxes, labels, and scores
    bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]

    # Count the vehicles with score > 0.5
    num = torch.argwhere(scores > 0.5).shape[0]
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i] - 1]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        frame = cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                            cv2.LINE_AA)
        vehicle_num += 1

    cv2.putText(frame, "Vehicles: " + str(vehicle_num), (20, 80), 0, 5, (100, 200, 0), 5)
    resized_frame = cv2.resize(frame, (640, 480))  # Resize the frame to 640x480
    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Total vehicle count:", vehicle_num)

video_capture.release()
cv2.destroyAllWindows()
