import cv2
from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\Akash K B\Downloads\YawDD.rar\YawDD\YawDD dataset"
)

video_path = next(DATASET_PATH.rglob("*.avi"))

cap = cv2.VideoCapture(str(video_path))

print("Video:", video_path.name)
print("Opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

ret, frame = cap.read()

if ret:
    cv2.imshow("YawDD Test Video", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not read video frame")

cap.release()