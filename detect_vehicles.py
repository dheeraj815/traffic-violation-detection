import cv2
from ultralytics import YOLO
import os

# === Step 1: Load video ===
video_path = r"C:\Users\Dheeraj Muley\OneDrive\Desktop\MyAIProject\traffic-violation-detection\traffic\trafficvideo.mp4"

if not os.path.exists(video_path):
    print("❌ Video file not found.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Couldn't open video.")
    exit()

print("✅ Video loaded successfully.")

# === Step 2: Load YOLOv8 model ===
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

# === Step 3: Process each frame ===
while True:
    success, frame = cap.read()
    if not success:
        print("✅ Video processing complete.")
        break

    # Run vehicle detection
    results = model(frame)[0]

    # Classes for vehicle-related objects (COCO dataset)
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Draw detected boxes
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display result
    cv2.imshow("🚗 Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
