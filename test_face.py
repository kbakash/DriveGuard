import cv2
import mediapipe as mp
from pathlib import Path

dataset = Path(
    r"C:\Users\Akash K B\Downloads\YawDD.rar\YawDD\YawDD dataset"
)

video = next(dataset.rglob("*.avi"))
cap = cv2.VideoCapture(str(video))

mp_face = mp.solutions.face_mesh

with mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    ret, frame = cap.read()

    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    face,
                    mp_face.FACEMESH_CONTOURS
                )

            cv2.imshow("Face Landmarks", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

cap.release()