import cv2
import os
from ultralytics import YOLO

# === 1. Load video ===
video_path = r"C:\Users\Dheeraj Muley\OneDrive\Desktop\MyAIProject\traffic-violation-detection\traffic\trafficvideo.mp4"

print("🎥 Checking path:", video_path)

if not os.path.exists(video_path):
    print("❌ Video file NOT found at:", video_path)
    exit()
else:
    print("✅ Video file found.")

# ✅ Define cap before using it!
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ OpenCV could NOT open the video.")
    exit()
else:
    print("🎬 Video loaded successfully.")

# === 2. Load YOLOv8 model ===
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for speed

# === 3. Loop through video frames and detect vehicles ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video.")
        break

    # Run detection
    results = model(frame)[0]

    # Loop through detected boxes
    for r in results.boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        # Only detect vehicles (car, motorcycle, bus, truck)
        if cls in [2, 3, 5, 7]:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show video frame
    cv2.imshow("Vehicle Detection", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
