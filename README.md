# BIRD_COUNT__WEIGHT_PROJECT
This project uses YOLOv8 for real-time object detection and ByteTrack for multi-object tracking in videos. It is designed to detect,track,and visualize birds in poultry videos. 
Each bird is assigned a unique ID, allowing for monitoring of movement across frames.The output is a video with bounding boxes and object IDs displayed on each detected bird

Features
Detect birds in video frames using YOLOv8.
Track birds across frames using OpenCV’s MultiTracker (CSRT).
Annotate the video with bounding boxes and bird IDs.
Estimate bird weight using bounding box area as a proxy.
Provides JSON output with per-frame counts, IDs, and weight estimates.

Requirements
Python 3.8+
FastAPI
Uvicorn
OpenCV
Ultralytics YOLOv8

Install dependencies:
pip install -r requirements.txt

How to Run the API
1.Start the FastAPI server: uvicorn app:app --reload
2.Health check: GET http://127.0.0.1:8000/health
3.Analyze a video: POST http://127.0.0.1:8000/analyze_video


With form-data:
video → video file (mp4)
fps_sample → frame sampling (default: 5)
conf_thresh → confidence threshold (default: 0.05)

Implementation Details:-
Detection: YOLOv8 detects birds on sampled frames (fps_sample).
Tracking: OpenCV MultiTracker tracks birds between detection frames to maintain IDs.
Weight Estimation: Approximated using bounding box area (width * height * scaling_factor).
iou_thresh → IoU threshold (default: 0.5)

Sample video Link :- https://drive.google.com/file/d/1bb5AbrVPpmc3a0Siv13ud_k9ir0Xr_mi/view?usp=sharing
Colab Link :- https://colab.research.google.com/drive/1vSVsWDl3fyQQ4PZE_-KvlhZxLRoHBqVj?usp=sharing
Annotted video Link :- https://drive.google.com/file/d/1AtGPj9gBz-2zQJXnkFxcKF9bozG0NrHn/view?usp=sharing

BirdCountWeightProject/
│
├─ app.py
├─ requirements.txt
├─ yolov8n.pt
├─ SampleVideos/
│ └─ video_test.mp4
├─ annotated_demo.avi
├─ sample_output.json
└─ README.md
