
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# 1. Load test data and normalization parameters
data = np.load("driveguard_sequences.npz")
norm = np.load("driveguard_normalization.npz")

X_test = data["X_test"]
y_test_text = data["y_test"]

mean = norm["mean"]
std = norm["std"]
classes = norm["classes"]

# 2. Normalize test data using training statistics
X_test = (X_test - mean) / (std + 1e-6)

# 3. Encode actual labels
encoder = LabelEncoder()
encoder.classes_ = classes

y_test = encoder.transform(y_test_text)

# 4. Load trained GRU model
model = tf.keras.models.load_model("driveguard_gru.keras")

# 5. Predict test labels
probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

# 6. Print test accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print("       GRU MODEL EVALUATION")
print("=" * 50)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 7. Print classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)

# 8. Generate confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(classes))
)

print("Confusion Matrix:")
print(cm)

# 9. Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(cmap="Blues", values_format="d")
plt.xticks(rotation=25, ha="right")
plt.title("GRU Behavior Classification")
plt.tight_layout()
plt.show()
