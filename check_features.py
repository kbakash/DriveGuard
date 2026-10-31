
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("yawdd_features.csv")

# Select one video from each behavior
examples = {
    "Normal": "1-FemaleNoGlasses-Normal.avi",
    "Talking": "1-FemaleNoGlasses-Talking.avi",
    "Yawning": "1-FemaleNoGlasses-Yawning.avi",
}

plt.figure(figsize=(12, 8))

for i, (label, video) in enumerate(examples.items(), start=1):
    data = df[df["video"] == video].sort_values("frame")

    plt.subplot(3, 1, i)
    plt.plot(data["frame"], data["EAR"], label="EAR")
    plt.plot(data["frame"], data["MAR"], label="MAR")
    plt.title(label)
    plt.ylabel("Feature value")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.xlabel("Frame")
plt.tight_layout()
plt.show()
