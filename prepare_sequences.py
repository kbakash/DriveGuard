
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURATION
# =========================
CSV_FILE = "yawdd_features.csv"
OUTPUT_FILE = "driveguard_sequences.npz"

SEQUENCE_LENGTH = 30
STRIDE = 5
RANDOM_STATE = 42

FEATURES = [
    "EAR",
    "MAR",
    "EAR_change",
    "MAR_change"
]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

required_columns = ["video", "frame", "label"] + FEATURES
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df.dropna(subset=required_columns).copy()
df = df.sort_values(["video", "frame"]).reset_index(drop=True)

# =========================
# EXTRACT PARTICIPANT ID
# Example:
# 1-FemaleNoGlasses-Normal.avi -> 1
# 15-FemaleGlasses-Talking.avi -> 15
# =========================
def get_participant(video):
    filename = str(video).replace("\\", "/").split("/")[-1]
    match = re.match(r"^(\d+)-", filename)

    if match:
        return match.group(1)

    raise ValueError(
        f"Cannot extract participant ID from filename: {filename}"
    )

df["participant"] = df["video"].apply(get_participant)

# =========================
# CHECK VIDEO LABELS
# =========================
label_counts = df.groupby("video")["label"].nunique()

if (label_counts > 1).any():
    bad_videos = label_counts[label_counts > 1].index.tolist()
    raise ValueError(f"Videos with multiple labels: {bad_videos}")

print("Unique participants:", df["participant"].nunique())
print("\nVideos per class:")
print(df.groupby("label")["video"].nunique())

# =========================
# CREATE OVERLAPPING SEQUENCES
# 30 frames per sequence.
# Move forward by 5 frames each time.
# =========================
X = []
y = []
groups = []

for video, video_df in df.groupby("video", sort=False):
    video_df = video_df.sort_values("frame")

    features = video_df[FEATURES].to_numpy(dtype=np.float32)
    label = video_df["label"].iloc[0]
    participant = video_df["participant"].iloc[0]

    # Skip sequences containing invalid values.
    for start in range(
        0,
        len(features) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):
        sequence = features[start:start + SEQUENCE_LENGTH]

        if not np.isfinite(sequence).all():
            continue

        X.append(sequence)
        y.append(label)
        groups.append(participant)

if not X:
    raise ValueError("No sequences created. Check your feature CSV.")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=str)
groups = np.asarray(groups, dtype=str)

print("\nTotal sequences:", len(X))
print("Sequence shape:", X.shape)

print("\nSequences per class:")
print(pd.Series(y).value_counts())

# =========================
# SPLIT BY PARTICIPANT
# Approximately 70% train,
# 15% validation, 15% test.
# =========================
participants = np.unique(groups)

train_ids, temp_ids = train_test_split(
    participants,
    test_size=0.30,
    random_state=RANDOM_STATE
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=RANDOM_STATE
)

train_mask = np.isin(groups, train_ids)
val_mask = np.isin(groups, val_ids)
test_mask = np.isin(groups, test_ids)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# =========================
# CONSISTENT LABEL ENCODING
# =========================
classes = np.unique(y)

class_to_id = {
    label: index for index, label in enumerate(classes)
}

y_train = np.array(
    [class_to_id[label] for label in y_train],
    dtype=np.int32
)
y_val = np.array(
    [class_to_id[label] for label in y_val],
    dtype=np.int32
)
y_test = np.array(
    [class_to_id[label] for label in y_test],
    dtype=np.int32
)

# =========================
# VERIFY PARTICIPANT SPLIT
# =========================
train_participants = set(groups[train_mask])
val_participants = set(groups[val_mask])
test_participants = set(groups[test_mask])

assert train_participants.isdisjoint(val_participants)
assert train_participants.isdisjoint(test_participants)
assert val_participants.isdisjoint(test_participants)

print("\nParticipants:")
print("Train:", len(train_participants))
print("Validation:", len(val_participants))
print("Test:", len(test_participants))

print("\nFinal sequence shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nClass mapping:")
print({
    label: class_to_id[label] for label in classes
})

print("\nParticipant overlap:")
print("Train/validation:", len(train_participants & val_participants))
print("Train/test:", len(train_participants & test_participants))
print("Validation/test:", len(val_participants & test_participants))

# =========================
# SAVE DATASET
# =========================
np.savez_compressed(
    OUTPUT_FILE,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    classes=classes,
    train_participants=np.array(sorted(train_participants)),
    val_participants=np.array(sorted(val_participants)),
    test_participants=np.array(sorted(test_participants))
)

print(f"\nDataset saved to {OUTPUT_FILE}")
