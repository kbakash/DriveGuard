
import cv2
import csv
from pathlib import Path
import mediapipe as mp
import numpy as np

# Keep your existing dataset path here
DATASET = Path(
    r"C:\Users\Akash K B\Downloads\YawDD.rar\YawDD\YawDD dataset"
)

VIDEOS = sorted((DATASET / "Mirror").rglob("*.avi"))
OUTPUT_FILE = "yawdd_features.csv"

print("Total videos found:", len(VIDEOS))

LEFT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 13, 14, 0, 17]


def get_label(filename):
    name = filename.lower()

    if "talking&yawning" in name or "talkingandyawning" in name:
        return "Talking&Yawning"
    elif "yawning" in name:
        return "Yawning"
    elif "talking" in name:
        return "Talking"
    elif "normal" in name:
        return "Normal"

    return None


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def ear(points):
    return (
        distance(points[1], points[5])
        + distance(points[2], points[4])
    ) / (2 * distance(points[0], points[3]) + 1e-6)


def mar(points):
    width = distance(points[0], points[1])

    if width < 1e-6:
        return 0.0

    vertical = (
        distance(points[2], points[3])
        + distance(points[4], points[5])
    ) / 2.0

    return vertical / width


rows = []
mesh = mp.solutions.face_mesh

with mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    for video in VIDEOS:
        label = get_label(video.name)

        if label is None:
            continue

        cap = cv2.VideoCapture(str(video))
        frame_no = 0
        detected = 0

        # Reset temporal history for every video
        previous_ear = None
        previous_mar = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_no += 1

            # Process every 5th frame
            if frame_no % 5 != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                # Do not calculate changes across a missed detection
                previous_ear = None
                previous_mar = None
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            points = [(p.x * w, p.y * h) for p in landmarks]

            eye_points = [points[j] for j in LEFT_EYE]
            mouth_points = [points[j] for j in MOUTH]

            ear_value = ear(eye_points)
            mar_value = mar(mouth_points)

            if not (0.08 <= ear_value <= 0.60):
                previous_ear = None
                previous_mar = None
                continue

            if not (0.0 <= mar_value <= 1.5):
                previous_ear = None
                previous_mar = None
                continue

            # First valid frame has no previous measurement
            if previous_ear is None or previous_mar is None:
                ear_change = 0.0
                mar_change = 0.0
            else:
                ear_change = ear_value - previous_ear
                mar_change = mar_value - previous_mar

            rows.append([
                video.name,
                frame_no,
                ear_value,
                mar_value,
                ear_change,
                mar_change,
                label
            ])

            previous_ear = ear_value
            previous_mar = mar_value
            detected += 1

        cap.release()
        print(f"{video.name}: {detected} frames processed")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "video",
        "frame",
        "EAR",
        "MAR",
        "EAR_change",
        "MAR_change",
        "label"
    ])

    writer.writerows(rows)

print(f"\nSaved {len(rows)} feature rows to {OUTPUT_FILE}")
print("Features: EAR, MAR, EAR_change, MAR_change")
