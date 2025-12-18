# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# import shutil
# import os
# import cv2
# from ultralytics import YOLO
# import uuid

# app = FastAPI()

# # Load YOLO model
# model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in your folder

# # Utility to save uploaded file
# def save_upload_file(upload_file: UploadFile, destination: str) -> str:
#     with open(destination, "wb") as buffer:
#         shutil.copyfileobj(upload_file.file, buffer)
#     return destination

# @app.get("/health")
# def health_check():
#     return {"status": "OK"}

# @app.post("/analyze_video")
# async def analyze_video(
#     video: UploadFile = File(...),
#     fps_sample: int = Form(5),
#     conf_thresh: float = Form(0.3),
#     iou_thresh: float = Form(0.5),
# ):
#     # Save uploaded video
#     video_filename = f"temp_{uuid.uuid4().hex}.mp4"
#     save_upload_file(video, video_filename)

#     # OpenCV video reader
#     cap = cv2.VideoCapture(video_filename)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Video writer for annotated output
#     # out_filename = f"annotated_{uuid.uuid4().hex}.mp4"
#     # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     # out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
#     # Video writer for annotated output (Windows AVI format works better)
#     out_filename = f"annotated_{uuid.uuid4().hex}.avi"
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))


#     frame_count = 0
#     counts_over_time = []
#     bird_ids = {}

#     # Initialize MultiTracker (legacy)
#     trackers = cv2.legacy.MultiTracker_create()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Sample frames
#         if frame_count % fps_sample == 0:
#             # YOLO detection
#             results = model(frame, conf=conf_thresh)
#             boxes = []

#             for r in results:
#                 for box in r.boxes.xyxy:
#                     x1, y1, x2, y2 = map(int, box)
#                     boxes.append((x1, y1, x2 - x1, y2 - y1))

#             # Reset trackers for sampled frames
#             trackers = cv2.legacy.MultiTracker_create()
#             for bbox in boxes:
#                 trackers.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

#             # Update bird IDs
#             bird_ids = {f"ID_{i}": bbox[0] * bbox[1] for i, bbox in enumerate(boxes)}

#         else:
#             # Update trackers
#             success, boxes = trackers.update(frame)
#             if success:
#                 boxes = [tuple(map(int, b)) for b in boxes]
#             else:
#                 boxes = []

#         # Annotate frame
#         for i, bbox in enumerate(boxes):
#             x, y, w, h = bbox
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID_{i}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Count birds and weight proxy (area)
#         counts_over_time.append({
#             "frame": frame_count,
#             "count": len(boxes),
#             "birds": {f"ID_{i}": w*h for i, (x, y, w, h) in enumerate(boxes)}
#         })

#         out.write(frame)
#         frame_count += 1

#     cap.release()
#     out.release()

#     # Return JSON response
#     response = {
#         "counts": counts_over_time,
#         "tracks_sample": list(bird_ids.items())[:5],
#         "weight_estimates": {k: v for k, v in bird_ids.items()},  # area proxy
#         "artifact": out_filename
#     }

#     # Remove temp video
#     os.remove(video_filename)

#     return JSONResponse(content=response)
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# import shutil
# import os
# import cv2
# from ultralytics import YOLO
# import uuid

# app = FastAPI()

# # Load YOLO model
# model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in your folder

# # Utility to save uploaded file
# def save_upload_file(upload_file: UploadFile, destination: str) -> str:
#     with open(destination, "wb") as buffer:
#         shutil.copyfileobj(upload_file.file, buffer)
#     return destination

# @app.get("/health")
# def health_check():
#     return {"status": "OK"}

# @app.post("/analyze_video")
# async def analyze_video(
#     video: UploadFile = File(...),
#     fps_sample: int = Form(5),
#     conf_thresh: float = Form(0.3),
#     iou_thresh: float = Form(0.5),
# ):
#     # Save uploaded video
#     video_filename = f"temp_{uuid.uuid4().hex}.mp4"
#     save_upload_file(video, video_filename)

#     # OpenCV video reader
#     cap = cv2.VideoCapture(video_filename)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Video writer for annotated output (Windows AVI format works better)
#     out_filename = f"annotated_{uuid.uuid4().hex}.avi"
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

#     frame_count = 0
#     counts_over_time = []
#     bird_ids = {}

#     # Initialize MultiTracker (legacy)
#     trackers = cv2.legacy.MultiTracker_create()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Sample frames for YOLO detection
#         if frame_count % fps_sample == 0:
#             # YOLO detection
#             results = model(frame, conf=conf_thresh)
#             boxes = []

#             for r in results:
#                 for box in r.boxes.xyxy:
#                     x1, y1, x2, y2 = map(int, box)
#                     boxes.append((x1, y1, x2 - x1, y2 - y1))

#             # Reset trackers for sampled frames
#             trackers = cv2.legacy.MultiTracker_create()
#             for bbox in boxes:
#                 trackers.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

#             # Update bird IDs
#             bird_ids = {f"ID_{i}": bbox[0] * bbox[1] for i, bbox in enumerate(boxes)}

#         else:
#             # Update trackers
#             success, boxes = trackers.update(frame)
#             if success:
#                 boxes = [tuple(map(int, b)) for b in boxes]
#             else:
#                 boxes = []

#         # Annotate frame
#         for i, bbox in enumerate(boxes):
#             x, y, w, h = bbox
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID_{i}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Count birds and weight estimation
#         counts_over_time.append({
#             "frame": frame_count,
#             "count": len(boxes),
#             "birds": {f"ID_{i}": {
#                 "area": w*h,
#                 "weight_g": round(w*h*0.001, 2)  # simple scaling factor
#             } for i, (x, y, w, h) in enumerate(boxes)}
#         })

#         out.write(frame)
#         frame_count += 1

#     cap.release()
#     out.release()

#     # Return JSON response
#     response = {
#         "counts": counts_over_time,
#         "tracks_sample": list(bird_ids.items())[:5],
#         "weight_estimates": {k: round(v*0.001, 2) for k, v in bird_ids.items()},  # approximate weight
#         "artifact": out_filename
#     }

#     # Remove temp video
#     os.remove(video_filename)

#     return JSONResponse(content=response)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
from ultralytics import YOLO
import uuid

app = FastAPI()

# Load YOLO model
model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in your folder

# Utility to save uploaded file
def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    fps_sample: int = Form(5),
    conf_thresh: float = Form(0.05),
    iou_thresh: float = Form(0.5),
):
    # Save uploaded video
    video_filename = f"temp_{uuid.uuid4().hex}.mp4"
    save_upload_file(video, video_filename)

    # OpenCV video reader
    cap = cv2.VideoCapture(video_filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video writer for annotated output (Windows AVI format works better)
    out_filename = f"annotated_{uuid.uuid4().hex}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

    frame_count = 0
    counts_over_time = []
    bird_ids = {}

    # Initialize MultiTracker (legacy)
    trackers = cv2.legacy.MultiTracker_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames for YOLO detection
        if frame_count % fps_sample == 0:
            # YOLO detection
            results = model(frame, conf=conf_thresh)
            boxes = []

            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2 - x1, y2 - y1))

            # Reset trackers for sampled frames
            trackers = cv2.legacy.MultiTracker_create()
            for bbox in boxes:
                trackers.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

            # Update bird IDs and approximate weight
            bird_ids = {f"ID_{i}": w*h for i, (x, y, w, h) in enumerate(boxes)}

        else:
            # Update trackers
            success, boxes = trackers.update(frame)
            if success:
                boxes = [tuple(map(int, b)) for b in boxes]
            else:
                boxes = []

        # Annotate frame with ID and weight
        for i, bbox in enumerate(boxes):
            x, y, w, h = bbox
            area = w*h
            weight_g = round(area * 0.001, 2)  # simple scaling factor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID_{i} W:{weight_g}g", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Count birds and weight estimation
        counts_over_time.append({
            "frame": frame_count,
            "count": len(boxes),
            "birds": {f"ID_{i}": {"area": w*h, "weight_g": round(w*h*0.001, 2)}
                      for i, (x, y, w, h) in enumerate(boxes)}
        })

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Return JSON response
    response = {
        "counts": counts_over_time,
        "tracks_sample": list(bird_ids.items())[:5],
        "weight_estimates": {k: round(v*0.001, 2) for k, v in bird_ids.items()},  # approximate weight
        "artifact": out_filename
    }

    # Remove temp video
    os.remove(video_filename)

    return JSONResponse(content=response)
