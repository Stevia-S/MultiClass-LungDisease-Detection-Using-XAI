import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- CONFIG ----
DATA_DIR = "datasets"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5
CATEGORIES = ["COVID", "Normal", "Pneumonia"]

# ---- DATA GENERATOR ----
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ---- CUSTOM CNN ----
def build_cnn():
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),

        Conv2D(64,(3,3),activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),

        Conv2D(128,(3,3),activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.3),
        Dense(3,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# ---- TRANSFER MODELS ----
def build_transfer_model(base):
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(3,activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# ---- BUILD MODELS ----
cnn_model = build_cnn()
resnet_model = build_transfer_model(ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3)))
vgg_model = build_transfer_model(VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3)))
mobile_model = build_transfer_model(MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3)))

# ---- TRAIN MODELS ----
print("Training CNN...")
cnn_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("Training ResNet...")
resnet_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("Training VGG...")
vgg_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

print("Training MobileNet...")
mobile_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ---- GRAD CAM ----
def grad_cam(model, img_array):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer = layer
            break

    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap,0) / (np.max(heatmap)+1e-8)
    return heatmap

# ---- HEATMAP DISPLAY ----
def show_heatmap(img, heatmap, title):
    heatmap = cv2.resize(heatmap, (IMG_SIZE,IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay[:,:,::-1])
    plt.title(title)
    plt.axis("off")
    plt.show()

# ---- INFECTION AREA ----
def infection_score(heatmap, percentile=80):
    threshold = np.percentile(heatmap, percentile)
    infected = np.sum(heatmap >= threshold)
    total = heatmap.size
    return (infected / total) * 100

# ---- LOAD SAMPLE IMAGE ----
img_path = val_gen.filepaths[0]
img = cv2.imread(img_path)
img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
img_array = np.expand_dims(img/255.0, axis=0)

# ---- PREDICT FUNCTION ----
def predict_and_show(model, name):
    loss, acc = model.evaluate(val_gen, verbose=0)

    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]

    print(f"\n{name} RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"Prediction: {CATEGORIES[pred_class]}")
    print(f"Confidence: {confidence:.2f}")

    heatmap = grad_cam(model, img_array)
    inf_pct = infection_score(heatmap)

    print(f"Affected Area: {inf_pct:.2f}%")

    show_heatmap(img, heatmap, f"{name} Heatmap")

# ---- FINAL OUTPUT ----
print("\n--- RESULTS ---")

predict_and_show(resnet_model, "ResNet50")
predict_and_show(vgg_model, "VGG16")
predict_and_show(mobile_model, "MobileNetV2")
predict_and_show(cnn_model, "Custom CNN")

# ---- MEDICAL DISCLAIMER ----
print("\n--- MEDICAL DISCLAIMER ---")
print("This system is for research and educational purposes only.")
print("It is NOT a medical diagnostic tool.")
print("Always consult a certified radiologist or doctor for clinical decisions.")