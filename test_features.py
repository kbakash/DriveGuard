
import cv2
import mediapipe as mp
import numpy as np

# Yawning video for testing
VIDEO_PATH = (
    r"C:\Users\Akash K B\Downloads\YawDD.rar\YawDD\YawDD dataset"
    r"\Mirror\Female_mirror\1-FemaleNoGlasses-Normal.avi"
)
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# Mouth corners and inner-lip landmarks
MOUTH = [61, 291, 13, 14]


def calculate_mar(points):
    width = distance(points[0], points[1])
    vertical = distance(points[2], points[3])

    return vertical / (width + 1e-6)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video:", VIDEO_PATH)
    raise SystemExit

with mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    for i in range(16):
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame:", i + 1)
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            print(f"Frame {i + 1}: Face not detected")
            continue

        h, w = frame.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark

        points = [
            (p.x * w, p.y * h)
            for p in landmarks
        ]

        mouth = [points[idx] for idx in MOUTH]

        width = distance(mouth[0], mouth[1])
        vertical = distance(mouth[2], mouth[3])
        value = calculate_mar(mouth)

        print(
            f"Frame {i + 1}: "
            f"Width={width:.2f}, "
            f"Vertical={vertical:.2f}, "
            f"MAR={value:.4f}"
        )

        # Draw the four selected landmarks
        for idx in MOUTH:
            x, y = points[idx]

            cv2.circle(
                frame,
                (int(x), int(y)),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                str(idx),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Display the 16th frame only
        if i == 15:
            cv2.imshow("Mouth landmark check", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
