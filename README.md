# 🚦 AI-Powered Traffic Signal Violation Detection System

## 📌 Description
This project detects **traffic signal violations in real-time** using AI. It uses **YOLOv8** for vehicle detection, **OpenCV** for video processing, and **SORT** for tracking vehicles that cross a red signal in a restricted zone. The system simulates traffic lights and logs violations with timestamp and vehicle ID.

## 🎯 Features
- 🚗 Detects and tracks vehicles using YOLOv8 + SORT
- 🚦 Traffic light simulation (Red, Yellow, Green)
- 🔺 Custom polygon-based restricted zone
- 📸 Logs violations with vehicle ID, time, and frame
- 📂 Saves violation images in an output folder

## 🛠️ Tech Stack
- Python 3.x
- OpenCV
- YOLOv8 (`ultralytics`)
- SORT Tracking Algorithm
- Numpy, Pandas
- (Optional) Email/SMS alert system

## 🚀 How to Run

1. **Install Dependencies**
```bash
pip install -r requirements.txt
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy
python traffic_violation_detection.py
