
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIGURATION
# =========================
DATA_FILE = "driveguard_sequences.npz"
MODEL_FILE = "driveguard_gru.keras"
NORMALIZATION_FILE = "driveguard_normalization.npz"

SEQUENCE_LENGTH = 30
NUM_FEATURES = 4
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================
# LOAD DATA
# =========================
data = np.load(DATA_FILE, allow_pickle=False)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.int32)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.int32)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int32)

classes = data["classes"].astype(str)
num_classes = len(classes)

print("=" * 50)
print("       DRIVEGUARD GRU TRAINING")
print("=" * 50)

print("\nDataset shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Classes:", classes)

# =========================
# VALIDATE DATA
# =========================
assert X_train.ndim == 3
assert X_train.shape[1:] == (SEQUENCE_LENGTH, NUM_FEATURES)
assert X_val.shape[1:] == (SEQUENCE_LENGTH, NUM_FEATURES)
assert X_test.shape[1:] == (SEQUENCE_LENGTH, NUM_FEATURES)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name} class counts:")
    print({
        classes[i]: int(np.sum(labels == i))
        for i in range(num_classes)
    })

# =========================
# NORMALIZE
# Use training data only
# =========================
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)
std = np.maximum(std, 1e-6)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# =========================
# CLASS WEIGHTS
# =========================
present_classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=present_classes,
    y=y_train
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(present_classes, weights)
}

print("\nClass weights:", class_weights)

# =========================
# BUILD GRU MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, NUM_FEATURES)
    ),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# =========================
# LOAD BEST MODEL
# =========================
model = tf.keras.models.load_model(MODEL_FILE)

# =========================
# TEST ON UNSEEN PARTICIPANTS
# =========================
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

print("\n" + "=" * 50)
print("          FINAL TEST RESULTS")
print("=" * 50)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f} "
      f"({test_accuracy * 100:.2f}%)")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(num_classes),
    target_names=classes,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(num_classes)
))

# =========================
# SAVE NORMALIZATION
# =========================
np.savez(
    NORMALIZATION_FILE,
    mean=mean.reshape(-1),
    std=std.reshape(-1),
    classes=classes
)

print(f"\nSaved best model: {MODEL_FILE}")
print(f"Saved normalization: {NORMALIZATION_FILE}")
