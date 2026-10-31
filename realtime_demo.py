
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque
from math import hypot

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "driveguard_gru.keras"
NORMALIZATION_PATH = "driveguard_normalization.npz"

SEQUENCE_LENGTH = 30

# Starting values; calibrate for your camera and face.
EAR_THRESHOLD = 0.20
CLOSED_WARNING_SECONDS = 1.5

# Standard six-point EAR landmark groups.
# One group for each eye.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 291, 13, 14, 0, 17]

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model(MODEL_PATH)

saved = np.load(NORMALIZATION_PATH)
mean = saved["mean"]
std = np.maximum(saved["std"], 1e-6)
classes = saved["classes"]

sequence = deque(maxlen=SEQUENCE_LENGTH)

previous_ear = None
previous_mar = None

prediction_label = "Collecting frames..."
prediction_confidence = 0.0

eyes_closed_since = None
eyes_closed = False

# ---------------- FEATURE FUNCTIONS ----------------

def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def point(landmarks, index, width, height):
    p = landmarks[index]
    return (p.x * width, p.y * height)


def calculate_ear(landmarks, indices, width, height):
    p = [
        point(landmarks, i, width, height)
        for i in indices
    ]

    horizontal = distance(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0

    vertical_1 = distance(p[1], p[5])
    vertical_2 = distance(p[2], p[4])

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks, width, height):
    p = [
        point(landmarks, i, width, height)
        for i in MOUTH
    ]

    mouth_width = distance(p[0], p[1])
    if mouth_width < 1e-6:
        return 0.0

    vertical = (
        distance(p[2], p[3])
        + distance(p[4], p[5])
    ) / 2.0

    return vertical / mouth_width


# ---------------- START WEBCAM ----------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Webcam unavailable. Check camera permissions or try camera index 1."
    )

mp_face_mesh = mp.solutions.face_mesh

try:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            now = time.monotonic()
            eye_status = "Face not detected"
            warning = ""

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                left_ear = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )
                right_ear = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )

                # Average both eyes to reduce one-eye measurement noise.
                ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(landmarks, width, height)

                # Eye-closure state with sustained-closure timing.
                if ear < EAR_THRESHOLD:
                    if eyes_closed_since is None:
                        eyes_closed_since = now

                    eyes_closed = True
                    closed_duration = now - eyes_closed_since
                    eye_status = f"EYES CLOSED ({closed_duration:.1f}s)"

                    if closed_duration >= CLOSED_WARNING_SECONDS:
                        warning = "WARNING: PROLONGED EYE CLOSURE"
                else:
                    eyes_closed_since = None
                    eyes_closed = False
                    eye_status = "EYES OPEN"

                # Existing GRU feature pipeline.
                if (
                    0.08 <= ear <= 0.60
                    and 0.0 <= mar <= 1.5
                ):
                    ear_change = (
                        0.0 if previous_ear is None
                        else ear - previous_ear
                    )
                    mar_change = (
                        0.0 if previous_mar is None
                        else mar - previous_mar
                    )

                    sequence.append([
                        ear, mar, ear_change, mar_change
                    ])

                    previous_ear = ear
                    previous_mar = mar

                    if len(sequence) == SEQUENCE_LENGTH:
                        x = np.asarray(
                            sequence, dtype=np.float32
                        )[None, :, :]

                        x = (x - mean) / std
                        probabilities = model.predict(
                            x, verbose=0
                        )[0]

                        best = int(np.argmax(probabilities))
                        prediction_label = str(classes[best])
                        prediction_confidence = float(
                            probabilities[best]
                        )
                else:
                    sequence.clear()
                    previous_ear = None
                    previous_mar = None
                    prediction_label = "Collecting valid frames..."
                    prediction_confidence = 0.0

                cv2.putText(
                    frame,
                    f"Left EAR: {left_ear:.3f}  Right EAR: {right_ear:.3f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                )

            else:
                sequence.clear()
                previous_ear = None
                previous_mar = None
                eyes_closed_since = None
                eyes_closed = False
                prediction_label = "Face not detected"
                prediction_confidence = 0.0

            if len(sequence) < SEQUENCE_LENGTH:
                prediction_text = (
                    f"Behaviour: collecting {len(sequence)}"
                    f"/{SEQUENCE_LENGTH}"
                )
            else:
                prediction_text = (
                    f"Behaviour: {prediction_label} "
                    f"({prediction_confidence * 100:.1f}%)"
                )

            cv2.putText(
                frame, eye_status, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255) if eyes_closed else (0, 220, 0), 2
            )

            cv2.putText(
                frame, prediction_text, (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2
            )

            if warning:
                cv2.putText(
                    frame, warning, (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2
                )

            cv2.imshow("DriveGuard - Eye Closure + Behaviour", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
