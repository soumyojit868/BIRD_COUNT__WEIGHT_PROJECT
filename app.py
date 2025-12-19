<<<<<<< HEAD
=======
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

>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
from ultralytics import YOLO
import uuid

app = FastAPI()

<<<<<<< HEAD
# -------------------------
# Load YOLO model (light version)
# -------------------------
model = YOLO("yolov8n.pt")  # make sure yolov8n.pt is in your project folder

# -------------------------
# Helper to save uploaded file
# -------------------------
=======
# Load YOLO model
model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in your folder

# Utility to save uploaded file
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination

<<<<<<< HEAD
# -------------------------
# Health check endpoint
# -------------------------
=======
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
@app.get("/health")
def health_check():
    return {"status": "OK"}

<<<<<<< HEAD
# -------------------------
# Video analysis endpoint
# -------------------------
=======
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    fps_sample: int = Form(5),
<<<<<<< HEAD
    conf_thresh: float = Form(0.02),   # low threshold for small/far birds
    iou_thresh: float = Form(0.5),
    max_weight_g: float = Form(4000.0),  # realistic max poultry weight
    min_box_area: int = Form(500),       # filter tiny boxes
=======
    conf_thresh: float = Form(0.05),
    iou_thresh: float = Form(0.5),
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
):
    # Save uploaded video
    video_filename = f"temp_{uuid.uuid4().hex}.mp4"
    save_upload_file(video, video_filename)

    # OpenCV video reader
    cap = cv2.VideoCapture(video_filename)
<<<<<<< HEAD
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "Cannot open video"})

=======
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

<<<<<<< HEAD
    # Video writer for annotated output
=======
    # Video writer for annotated output (Windows AVI format works better)
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
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

<<<<<<< HEAD
        boxes = []

        # -------------------------
        # Run YOLO detection every fps_sample frames
        # -------------------------
        if frame_count % fps_sample == 0:
            results = model.predict(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=1280,       # higher resolution to detect small/far birds
                classes=[14],     # COCO class 14 = bird
                augment=True,     # improves small object detection
                verbose=False
            )
=======
        # Sample frames for YOLO detection
        if frame_count % fps_sample == 0:
            # YOLO detection
            results = model(frame, conf=conf_thresh)
            boxes = []
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2

            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
<<<<<<< HEAD
                    w, h = x2 - x1, y2 - y1
                    if w*h >= min_box_area:  # filter tiny boxes
                        boxes.append((x1, y1, w, h))

            # Reset trackers
=======
                    boxes.append((x1, y1, x2 - x1, y2 - y1))

            # Reset trackers for sampled frames
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
            trackers = cv2.legacy.MultiTracker_create()
            for bbox in boxes:
                trackers.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

<<<<<<< HEAD
            # Save bird IDs for first frame
            bird_ids = {f"ID_{i}": w*h for i, (x, y, w, h) in enumerate(boxes)}

        else:
            # Update trackers for non-sampled frames
            success, boxes_tracked = trackers.update(frame)
            if success:
                boxes = [tuple(map(int, b)) for b in boxes_tracked]
            else:
                boxes = []

        # -------------------------
        # Compute realistic relative weights
        # -------------------------
        areas = [w*h for (x, y, w, h) in boxes]
        max_area = max(areas) if areas else 1

        bird_data = {}
        for i, bbox in enumerate(boxes):
            x, y, w, h = bbox
            area = w*h
            weight_g = round((area / max_area) * max_weight_g, 2)

            # Draw rectangle and weight
=======
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
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID_{i} W:{weight_g}g", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

<<<<<<< HEAD
            bird_data[f"ID_{i}"] = {
                "bbox": (x, y, w, h),
                "area": area,
                "weight_g": weight_g
            }

        # -------------------------
        # Save counts and weights
        # -------------------------
        counts_over_time.append({
            "frame": frame_count,
            "count": len(boxes),
            "birds": bird_data
=======
        # Count birds and weight estimation
        counts_over_time.append({
            "frame": frame_count,
            "count": len(boxes),
            "birds": {f"ID_{i}": {"area": w*h, "weight_g": round(w*h*0.001, 2)}
                      for i, (x, y, w, h) in enumerate(boxes)}
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
        })

        out.write(frame)
        frame_count += 1

<<<<<<< HEAD
        # Optional: show progress
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}")

    # Release resources
    cap.release()
    out.release()
    os.remove(video_filename)  # remove temporary video

    # -------------------------
    # Return JSON response
    # -------------------------
    response = {
        "frames_processed": frame_count,
        "sample_summary": counts_over_time[:5],
        "output_video": out_filename
    }

    return JSONResponse(content=response)

=======
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
>>>>>>> 5854835a972bc5ec32970e10e5899149297cf1c2
