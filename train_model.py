import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras import layers, models

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight

# =====================================================
# SETTINGS
# =====================================================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25

# =====================================================
# DATASET PATHS
# =====================================================
TRAIN_PATH = r"C:\Users\Dubai Laptops\PycharmProjects\PythonProject1\Pneumonia_Project\pneumonia dataset\train"

TEST_PATH = r"C:\Users\Dubai Laptops\PycharmProjects\PythonProject1\Pneumonia_Project\pneumonia dataset\test"

# =====================================================
# CLAHE PREPROCESSING
# =====================================================
def clahe_preprocessing(img):

    # Convert image to numpy array
    img = np.array(img)

    # Ensure uint8 datatype
    if img.dtype != np.uint8:

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        else:
            img = img.astype(np.uint8)

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
# TRAIN + VALIDATION GENERATOR
# =====================================================
train_datagen = ImageDataGenerator(

    preprocessing_function=clahe_preprocessing,

    validation_split=0.20,

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    brightness_range=[0.9, 1.1],

    fill_mode='nearest'

)

# =====================================================
# TEST GENERATOR
# =====================================================
test_datagen = ImageDataGenerator(

    preprocessing_function=clahe_preprocessing

)

# =====================================================
# TRAIN DATA
# =====================================================
train_data = train_datagen.flow_from_directory(

    TRAIN_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training',

    shuffle=True

)

# =====================================================
# VALIDATION DATA (VIRTUAL SPLIT)
# =====================================================
val_data = train_datagen.flow_from_directory(

    TRAIN_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='validation',

    shuffle=True

)

# =====================================================
# TEST DATA
# =====================================================
test_data = test_datagen.flow_from_directory(

    TEST_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    shuffle=False

)

# =====================================================
# CLASS INDICES
# =====================================================
print("\n==============================")
print("CLASS INDICES")
print("==============================")

print(train_data.class_indices)

# =====================================================
# CLASS WEIGHTS
# =====================================================
class_weights = compute_class_weight(

    class_weight='balanced',

    classes=np.unique(train_data.classes),

    y=train_data.classes

)

class_weights = dict(
    enumerate(class_weights)
)

print("\n==============================")
print("CLASS WEIGHTS")
print("==============================")

print(class_weights)

# =====================================================
# STEPS
# =====================================================
train_steps = max(
    train_data.samples // BATCH_SIZE,
    1
)

val_steps = max(
    val_data.samples // BATCH_SIZE,
    1
)

# =====================================================
# LOAD PRETRAINED MODEL
# =====================================================
base_model = EfficientNetB0(

    input_shape=(224, 224, 3),

    include_top=False,

    weights='imagenet'

)

# =====================================================
# FREEZE MOST LAYERS
# =====================================================
for layer in base_model.layers[:-20]:

    layer.trainable = False

# =====================================================
# CUSTOM HEAD
# =====================================================
x = base_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation='relu'
)(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(
    128,
    activation='relu'
)(x)

x = layers.Dropout(0.3)(x)

output = layers.Dense(
    1,
    activation='sigmoid'
)(x)

# =====================================================
# FINAL MODEL
# =====================================================
model = models.Model(

    inputs=base_model.input,

    outputs=output

)

# =====================================================
# COMPILE MODEL
# =====================================================
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00003
    ),

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.Precision(
            name='precision'
        ),

        tf.keras.metrics.Recall(
            name='recall'
        )

    ]

)

# =====================================================
# CALLBACKS
# =====================================================
callbacks = [

    ModelCheckpoint(

        "best_pneumonia_model.keras",

        monitor='val_accuracy',

        save_best_only=True,

        mode='max',

        verbose=1

    ),

    EarlyStopping(

        monitor='val_accuracy',

        patience=8,

        restore_best_weights=True,

        verbose=1

    ),

    ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.3,

        patience=3,

        min_lr=1e-7,

        verbose=1

    )

]

# =====================================================
# MODEL SUMMARY
# =====================================================
print("\n======================================")
print("🧠 MODEL SUMMARY")
print("======================================")

model.summary()

# =====================================================
# START TRAINING
# =====================================================
print("\n======================================")
print("🚀 TRAINING STARTED")
print("======================================")

# =====================================================
# TRAIN MODEL
# =====================================================
history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    steps_per_epoch=train_steps,

    validation_steps=val_steps,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)

# =====================================================
# TEST MODEL
# =====================================================
print("\n======================================")
print("🧪 TESTING MODEL")
print("======================================")

test_loss, test_acc, test_precision, test_recall = model.evaluate(
    test_data
)

print("\n====================================")
print(f"✅ TEST ACCURACY : {test_acc*100:.2f}%")
print(f"✅ TEST PRECISION: {test_precision*100:.2f}%")
print(f"✅ TEST RECALL   : {test_recall*100:.2f}%")
print("====================================")

# =====================================================
# SAVE FINAL MODEL
# =====================================================
model.save(
    "final_pneumonia_model.keras"
)

print("\n====================================")
print("💾 MODEL SAVED")
print("====================================")

# =====================================================
# FINAL ANALYSIS
# =====================================================
final_train_acc = history.history['accuracy'][-1]

final_val_acc = history.history['val_accuracy'][-1]

gap = abs(
    final_train_acc -
    final_val_acc
)

print("\n====================================")
print("🧠 FINAL MODEL ANALYSIS")
print("====================================")

print(f"🔥 Train Accuracy      : {final_train_acc*100:.2f}%")

print(f"🔥 Validation Accuracy : {final_val_acc*100:.2f}%")

print(f"🔥 Accuracy Gap        : {gap*100:.2f}%")

# =====================================================
# OVERFITTING CHECK
# =====================================================
if gap < 0.05:

    print("\n✅ MODEL IS WELL BALANCED")

elif gap < 0.10:

    print("\n⚠️ SLIGHT OVERFITTING")

else:

    print("\n❌ HIGH OVERFITTING DETECTED")

print("\n🔥 PNEUMONIA MODEL READY")

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = clahe_preprocessing(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(img)[0][0]

    print("\n====================================")

    if prediction > 0.5:

        confidence = prediction * 100

        print(f"🦠 Prediction : PNEUMONIA")
        print(f"📊 Confidence : {confidence:.2f}%")

    else:

        confidence = (1 - prediction) * 100

        print(f"🫁 Prediction : NORMAL")
        print(f"📊 Confidence : {confidence:.2f}%")

    print("====================================")


# =====================================================
# EXAMPLE PREDICTION
# =====================================================
# predict_image(r"test_image.jpeg")