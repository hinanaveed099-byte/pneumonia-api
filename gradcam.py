import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================================
# SETTINGS
# =========================================
MODEL_PATH = "best_pneumonia_model.keras"

IMAGE_PATH = r"C:\Users\Dubai Laptops\PycharmProjects\PythonProject1\Pneumonia_Project\pneumonia.jfif"

IMG_SIZE = (224, 224)

# =========================================
# CLASS NAMES
# =========================================
classes = ['NORMAL', 'PNEUMONIA']

# =========================================
# LOAD MODEL
# =========================================
model = tf.keras.models.load_model(MODEL_PATH)

print("\n====================================")
print("✅ MODEL LOADED SUCCESSFULLY")
print("====================================")

# =========================================
# LOAD IMAGE
# =========================================
img = image.load_img(

    IMAGE_PATH,

    target_size=IMG_SIZE

)

img_array = image.img_to_array(img)

# =========================================
# CLAHE PREPROCESSING
# =========================================
gray = cv2.cvtColor(

    img_array.astype(np.uint8),

    cv2.COLOR_RGB2GRAY

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

# =========================================
# PREPROCESS INPUT
# =========================================
input_img = preprocess_input(

    clahe_img.astype(np.float32)

)

input_img = np.expand_dims(

    input_img,

    axis=0

)

# =========================================
# PREDICTION
# =========================================
prediction = model.predict(input_img)[0][0]

# =========================================
# CLASSIFICATION
# =========================================
if prediction > 0.5:

    predicted_label = "PNEUMONIA"

    confidence = prediction * 100

else:

    predicted_label = "NORMAL"

    confidence = (1 - prediction) * 100

# =========================================
# SEVERITY LEVEL
# =========================================
if confidence < 70:

    severity = "MILD"

elif confidence < 85:

    severity = "MODERATE"

else:

    severity = "SEVERE"

# =========================================
# FIND LAST CONVOLUTION LAYER
# =========================================
last_conv_layer_name = None

for layer in reversed(model.layers):

    if len(layer.output.shape) == 4:

        last_conv_layer_name = layer.name

        break

print(f"\n✅ LAST CONVOLUTION LAYER: {last_conv_layer_name}")

last_conv_layer = model.get_layer(

    last_conv_layer_name

)

# =========================================
# CREATE GRAD MODEL
# =========================================
grad_model = tf.keras.models.Model(

    inputs=model.inputs,

    outputs=[

        last_conv_layer.output,

        model.output

    ]

)

# =========================================
# GRAD-CAM
# =========================================
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(input_img)

    loss = predictions[:, 0]

# =========================================
# GET GRADIENTS
# =========================================
grads = tape.gradient(

    loss,

    conv_outputs

)

# =========================================
# GLOBAL AVERAGE POOLING
# =========================================
pooled_grads = tf.reduce_mean(

    grads,

    axis=(0, 1, 2)

)

# =========================================
# FEATURE MAPS
# =========================================
conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

# =========================================
# RELU
# =========================================
heatmap = np.maximum(

    heatmap,

    0

)

# =========================================
# NORMALIZE
# =========================================
heatmap /= (np.max(heatmap) + 1e-8)

# =========================================
# STRONG FOCUS ENHANCEMENT
# =========================================
heatmap = np.power(

    heatmap,

    3.0

)

# =========================================
# REMOVE LOW ATTENTION AREAS
# =========================================
heatmap[heatmap < 0.75] = 0

# =========================================
# SHARPEN PNEUMONIA REGION
# =========================================
heatmap = cv2.GaussianBlur(

    heatmap,

    (3,3),

    0

)

# =========================================
# NORMALIZE AGAIN
# =========================================
heatmap = heatmap / (

    np.max(heatmap) + 1e-8

)

# =========================================
# AFFECTED AREA
# =========================================
affected_area = np.sum(

    heatmap > 0.75

)

total_area = heatmap.shape[0] * heatmap.shape[1]

affected_percent = (

    affected_area / total_area

) * 100

# =========================================
# RESIZE HEATMAP
# =========================================
heatmap = cv2.resize(

    heatmap,

    IMG_SIZE

)

heatmap = np.uint8(

    255 * heatmap

)

# =========================================
# RED HOTSPOT
# =========================================
heatmap = cv2.applyColorMap(

    heatmap,

    cv2.COLORMAP_JET

)

# =========================================
# ORIGINAL IMAGE
# =========================================
original = cv2.imread(IMAGE_PATH)

original = cv2.resize(

    original,

    IMG_SIZE

)

# =========================================
# OVERLAY
# =========================================
overlay = cv2.addWeighted(

    original,

    0.75,

    heatmap,

    0.45,

    0

)

# =========================================
# CONVERT TO GRAY
# =========================================
gray_heatmap = cv2.cvtColor(

    heatmap,

    cv2.COLOR_BGR2GRAY

)

# =========================================
# STRICT THRESHOLD
# =========================================
_, thresh = cv2.threshold(

    gray_heatmap,

    220,

    255,

    cv2.THRESH_BINARY

)

# =========================================
# FIND CONTOURS
# =========================================
contours, _ = cv2.findContours(

    thresh,

    cv2.RETR_EXTERNAL,

    cv2.CHAIN_APPROX_SIMPLE

)

# =========================================
# DRAW FOCUS AREA ONLY
# =========================================
for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 80:

        x, y, w, h = cv2.boundingRect(cnt)

        # RED RECTANGLE
        cv2.rectangle(

            overlay,

            (x, y),

            (x+w, y+h),

            (0, 0, 255),

            2

        )

        # FOCUS LABEL
        cv2.putText(

            overlay,

            "PNEUMONIA REGION",

            (x, y-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0,0,255),

            2

        )

# =========================================
# CREATE DASHBOARD
# =========================================
dashboard = np.zeros((720, 1200, 3), dtype=np.uint8)

dashboard[:] = (15,15,20)

# =========================================
# HEADER
# =========================================
cv2.putText(

    dashboard,

    "AI PNEUMONIA ANALYSIS REPORT",

    (300, 40),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    (255,255,255),

    2

)

# =========================================
# LEFT PANEL
# =========================================
cv2.rectangle(

    dashboard,

    (30,80),

    (550,650),

    (70,70,70),

    2

)

cv2.putText(

    dashboard,

    "AI FOCUSED PNEUMONIA REGION",

    (120,115),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.7,

    (255,255,255),

    2

)

# =========================================
# LARGE IMAGE
# =========================================
large_overlay = cv2.resize(

    overlay,

    (450,500)

)

dashboard[140:640, 60:510] = large_overlay

# =========================================
# RIGHT PANEL
# =========================================
cv2.rectangle(

    dashboard,

    (620,80),

    (1150,650),

    (70,70,70),

    2

)

# =========================================
# RESULT
# =========================================
cv2.putText(

    dashboard,

    f"Prediction : {predicted_label}",

    (680,160),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.9,

    (0,255,255),

    2

)

cv2.putText(

    dashboard,

    f"Confidence : {confidence:.2f}%",

    (680,240),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.9,

    (255,255,255),

    2

)

cv2.putText(

    dashboard,

    f"Severity : {severity}",

    (680,320),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.9,

    (0,140,255),

    2

)

cv2.putText(

    dashboard,

    f"Lung Infection : {affected_percent:.2f}%",

    (680,400),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.9,

    (0,255,120),

    2

)

# =========================================
# MEDICAL SUGGESTION
# =========================================
if severity == "MILD":

    advice = "Use medicines and consult doctor."

elif severity == "MODERATE":

    advice = "Medical consultation recommended."

else:

    advice = "Immediate doctor consultation required."

cv2.putText(

    dashboard,

    "AI Suggestion:",

    (680,500),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.8,

    (255,255,255),

    2

)

cv2.putText(

    dashboard,

    advice,

    (680,550),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.65,

    (180,180,180),

    2

)

# =========================================
# DISCLAIMER
# =========================================
cv2.putText(

    dashboard,

    "AI-assisted medical analysis only.",

    (380,695),

    cv2.FONT_HERSHEY_SIMPLEX,

    0.5,

    (180,180,180),

    1

)

# =========================================
# SAVE OUTPUT
# =========================================
save_name = f"{predicted_label}_Focused_GradCAM.jpg"

cv2.imwrite(

    save_name,

    dashboard

)

# =========================================
# SHOW OUTPUT
# =========================================
cv2.imshow(

    "AI Pneumonia Focused Analysis",

    dashboard

)

cv2.waitKey(0)

cv2.destroyAllWindows()

print("\n====================================")
print(f"✅ RESULT SAVED AS: {save_name}")
print("====================================")