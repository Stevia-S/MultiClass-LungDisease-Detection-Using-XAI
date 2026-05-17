# ============================================
# IMPORTS
# ============================================

import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.applications import (
    ResNet50,
    VGG16
)

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Concatenate
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

# ============================================
# CONFIG
# ============================================

DATA_DIR = "datasets"

CATEGORIES = [
    "COVID",
    "Normal",
    "Pneumonia"
]

IMG_SIZE = 128

BATCH_SIZE = 8

EPOCHS = 10

# ============================================
# PREPROCESSING
# ============================================

def preprocess_ct(img):

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

    enhanced = clahe.apply(gray)

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    return enhanced

# ============================================
# LOAD DATA
# ============================================

def load_data():

    data = []
    labels = []

    for idx, category in enumerate(CATEGORIES):

        folder = os.path.join(
            DATA_DIR,
            category
        )

        print(f"Loading {category} images...")

        for file in os.listdir(folder):

            path = os.path.join(
                folder,
                file
            )

            img = cv2.imread(path)

            if img is None:
                continue

            img = preprocess_ct(img)

            img = preprocess_input(img)

            data.append(img)

            labels.append(idx)

    return (
        np.array(data, dtype=np.float32),
        np.array(labels)
    )

# ============================================
# LOAD DATASET
# ============================================

print("Loading dataset...")

X, y = load_data()

print("Dataset Loaded")

print("X Shape:", X.shape)

print("Y Shape:", y.shape)

# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================
# BUILD FUSION MODEL
# ============================================

def build_model():

    input_tensor = Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # ========================================
    # RESNET50
    # ========================================

    resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=input_tensor
    )

    # ========================================
    # VGG16
    # ========================================

    vgg = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=input_tensor
    )

    # freeze pretrained layers
    resnet.trainable = False
    vgg.trainable = False

    # ========================================
    # FEATURE EXTRACTION
    # ========================================

    res_feat = GlobalAveragePooling2D()(
        resnet.output
    )

    vgg_feat = GlobalAveragePooling2D()(
        vgg.output
    )

    # ========================================
    # FEATURE FUSION
    # ========================================

    combined = Concatenate()(
        [res_feat, vgg_feat]
    )

    # ========================================
    # CLASSIFICATION HEAD
    # ========================================

    x = Dense(
        256,
        activation="relu"
    )(combined)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    output = Dense(
        len(CATEGORIES),
        activation="softmax"
    )(x)

    # ========================================
    # FINAL MODEL
    # ========================================

    model = Model(
        inputs=input_tensor,
        outputs=output
    )

    model.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ============================================
# CREATE MODEL
# ============================================

model = build_model()

model.summary()

# ============================================
# TRAINING
# ============================================

early_stop = EarlyStopping(
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

# ============================================
# SAVE MODEL
# ============================================

model.save("lung_model.h5")

print("✅ Model Saved")

# ============================================
# EVALUATION
# ============================================

y_pred = np.argmax(
    model.predict(X_test),
    axis=1
)

print("\n📊 Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CATEGORIES
    )
)