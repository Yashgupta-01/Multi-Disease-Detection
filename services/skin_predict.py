import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("models/skin_cancer.keras")

def predict_skin(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    # pred is probability of Malignant (sigmoid output)
    label = "Malignant" if pred > 0.5 else "Benign"
    confidence = float(pred) if label == "Malignant" else float(1 - pred)

    return {
        "disease": "Skin Cancer",
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }


# ============================================================
# ⚠️  SKIN CANCER MODEL IS BROKEN — NEEDS RETRAINING
# ============================================================
# Problem: Model predicts everything as Benign.
# Cause: HAM10000 has ~6:1 class imbalance (benign >> malignant)
# Fix: Retrain with class_weight to penalize missing malignant cases
#
# Add this to your skin_cancer.ipynb training cell:
#
#   from sklearn.utils.class_weight import compute_class_weight
#   import numpy as np
#
#   labels = train_generator.classes
#   class_weights = compute_class_weight(
#       class_weight='balanced',
#       classes=np.unique(labels),
#       y=labels
#   )
#   class_weight_dict = dict(enumerate(class_weights))
#   print("Class weights:", class_weight_dict)
#   # Expected output: {0: ~0.58, 1: ~2.1}  (malignant gets higher weight)
#
#   history = model.fit(
#       train_generator,
#       validation_data=val_generator,
#       epochs=20,
#       class_weight=class_weight_dict,   # <-- THIS LINE IS THE FIX
#       callbacks=[early_stopping, reduce_lr]
#   )
#
# After retraining, save as: models/skin_cancer.keras
# ============================================================