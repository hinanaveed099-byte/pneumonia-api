from flask import Flask, request, jsonify

import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# =========================================
# LOAD MODEL
# =========================================
model = tf.keras.models.load_model(
    "best_pneumonia_model.keras"
)

IMG_SIZE = 224

# =========================================
# PREPROCESS FUNCTION
# =========================================
def preprocess_image(img):

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    clahe_img = clahe.apply(gray)

    clahe_img = cv2.cvtColor(
        clahe_img,
        cv2.COLOR_GRAY2RGB
    )

    clahe_img = preprocess_input(
        clahe_img.astype(np.float32)
    )

    clahe_img = np.expand_dims(
        clahe_img,
        axis=0
    )

    return clahe_img

# =========================================
# API ROUTE
# =========================================
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:

        return jsonify({

            "error": "No image uploaded"

        })

    file = request.files['file']

    image_bytes = file.read()

    npimg = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )

    processed = preprocess_image(img)

    prediction = model.predict(processed)[0][0]

    # =====================================
    # PREDICTION LOGIC
    # =====================================
    if prediction > 0.5:

        confidence = float(prediction * 100)

        label = "PNEUMONIA"

        infection = float(confidence - 15)

        if confidence < 70:

            severity = "MILD"

            advice = "Consult doctor and use medicines."

        elif confidence < 85:

            severity = "MODERATE"

            advice = "Medical consultation recommended."

        else:

            severity = "SEVERE"

            advice = "Immediate medical attention required."

    else:

        confidence = float((1 - prediction) * 100)

        label = "NORMAL"

        infection = 0.0

        severity = "NORMAL"

        advice = "No visible pneumonia detected."

    # =====================================
    # JSON RESPONSE
    # =====================================
    return jsonify({

        "prediction": label,

        "confidence": round(confidence, 2),

        "lung_infection": round(infection, 2),

        "severity": severity,

        "suggestion": advice

    })

# =========================================
# RUN APP
# =========================================
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)