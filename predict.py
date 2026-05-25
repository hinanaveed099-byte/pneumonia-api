import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# =====================================================
# SETTINGS
# =====================================================
IMG_SIZE = 224

# =====================================================
# LOAD TRAINED MODEL
# =====================================================
model = load_model("best_pneumonia_model.keras")

print("\n====================================")
print("✅ MODEL LOADED SUCCESSFULLY")
print("====================================")

# =====================================================
# CLAHE PREPROCESSING
# =====================================================
def clahe_preprocessing(img):

    img = np.array(img)

    # RGB → GRAYSCALE
    gray = cv2.cvtColor(

        img,

        cv2.COLOR_RGB2GRAY

    )

    # CLAHE
    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8)

    )

    clahe_img = clahe.apply(gray)

    # GRAYSCALE → RGB
    clahe_img = cv2.cvtColor(

        clahe_img,

        cv2.COLOR_GRAY2RGB

    )

    # FLOAT32
    clahe_img = clahe_img.astype(np.float32)

    # EfficientNet preprocessing
    clahe_img = preprocess_input(

        clahe_img

    )

    return clahe_img

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_pneumonia(image_path):

    # =================================================
    # LOAD IMAGE
    # =================================================
    img = cv2.imread(image_path)

    if img is None:

        print("\n❌ IMAGE NOT FOUND")
        return

    # =================================================
    # ORIGINAL IMAGE
    # =================================================
    original_img = cv2.cvtColor(

        img,

        cv2.COLOR_BGR2RGB

    )

    # =================================================
    # RESIZE IMAGE
    # =================================================
    resized_img = cv2.resize(

        original_img,

        (IMG_SIZE, IMG_SIZE)

    )

    # =================================================
    # PREPROCESS IMAGE
    # =================================================
    processed_img = clahe_preprocessing(

        resized_img

    )

    # =================================================
    # ADD BATCH DIMENSION
    # =================================================
    processed_img = np.expand_dims(

        processed_img,

        axis=0

    )

    # =================================================
    # MODEL PREDICTION
    # =================================================
    prediction = model.predict(processed_img)[0][0]

    # =================================================
    # RESULT ANALYSIS
    # =================================================
    print("\n====================================")
    print("🩺 AI PNEUMONIA ANALYSIS REPORT")
    print("====================================")

    # =================================================
    # PNEUMONIA CASE
    # =================================================
    if prediction > 0.5:

        confidence = prediction * 100

        # Approx infection percentage
        infection = min(confidence - 15, 100)

        # =================================================
        # SEVERITY ANALYSIS
        # =================================================
        if confidence < 70:

            severity = "🟡 MILD"

            advice = "Consult doctor and use prescribed medicines."

        elif confidence < 85:

            severity = "🟠 MODERATE"

            advice = "Medical consultation recommended."

        else:

            severity = "🔴 SEVERE"

            advice = "Immediate medical attention required."

        # =================================================
        # PRINT RESULTS
        # =================================================
        print(f"🦠 Prediction        : PNEUMONIA")

        print(f"📊 Model Confidence  : {confidence:.2f}%")

        print(f"🫁 Lung Infection    : {infection:.2f}%")

        print(f"⚠️ Severity Level    : {severity}")

        print(f"💊 Suggestion        : {advice}")

        result_text = f"PNEUMONIA | {severity}"

    # =================================================
    # NORMAL CASE
    # =================================================
    else:

        confidence = (1 - prediction) * 100

        infection = max(0, 100 - confidence)

        severity = "🟢 NORMAL"

        advice = "No visible pneumonia detected."

        print(f"🫁 Prediction        : NORMAL")

        print(f"📊 Model Confidence  : {confidence:.2f}%")

        print(f"🫁 Lung Infection    : {infection:.2f}%")

        print(f"⚠️ Severity Level    : {severity}")

        print(f"💊 Suggestion        : {advice}")

        result_text = "NORMAL"

    print("====================================")

    # =================================================
    # DISCLAIMER
    # =================================================
    print("\n⚠️ DISCLAIMER:")
    print("This AI analysis is approximate and should not replace professional medical diagnosis.")

    # =================================================
    # DISPLAY IMAGE
    # =================================================
    plt.figure(figsize=(7, 7))

    plt.imshow(original_img)

    plt.title(

        result_text,

        fontsize=15

    )

    plt.axis("off")

    plt.show()

# =====================================================
# IMAGE PATH
# =====================================================

image_path = r"C:\Users\Dubai Laptops\PycharmProjects\PythonProject1\Pneumonia_Project\download (1).jfif"

# =====================================================
# RUN PREDICTION
# =====================================================

predict_pneumonia(image_path)