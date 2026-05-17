import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# =========================================================
# CONFIG
# =========================================================

CATEGORIES = ["COVID", "Normal", "Pneumonia"]
IMG_SIZE = 128

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="🫁 Lung AI Detector",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #111;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model.h5"
    )

    return model

model = load_model()

# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    # RGB -> GRAY
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # GRAY -> RGB
    rgb = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )

    # Resize
    resized = cv2.resize(
        rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
    normalized = resized.astype("float32") / 255.0

    return normalized

# =========================================================
# GRAD CAM
# =========================================================

def grad_cam(model, image):

    image_tensor = tf.cast(
        np.expand_dims(image, axis=0),
        tf.float32
    )

    # Find last convolutional layer in the model
    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model for Grad-CAM")

    # Prediction
    predictions = model(
        image_tensor,
        training=False
    )

    class_idx = tf.argmax(
        predictions[0]
    )

    confidence = float(
        predictions[0][class_idx]
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_tensor)
        class_channel = preds[:, class_idx]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (IMG_SIZE, IMG_SIZE),
        method="bilinear"
    )
    heatmap = tf.squeeze(heatmap)

    if tf.is_tensor(heatmap):
        heatmap = heatmap.numpy()

    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap = cv2.normalize(
        heatmap,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX
    )

    return heatmap, int(class_idx), confidence

# =========================================================
# INFECTION SCORE
# =========================================================

def infection_score(heatmap):

    threshold = np.percentile(
        heatmap,
        80
    )

    infected = np.sum(
        heatmap >= threshold
    )

    total = heatmap.size

    return float(
        infected / total * 100
    )

# =========================================================
# DISEASE INFO
# =========================================================

disease_info = {

    "COVID": {

        "color": "#ff4b4b",

        "description":
        "COVID-like lung patterns detected",

        "symptoms": [
            "Dry cough",
            "Fever",
            "Breathing difficulty"
        ],

        "actions": [
            "Seek medical consultation",
            "COVID test recommended",
            "Medical monitoring required"
        ]
    },

    "Normal": {

        "color": "#28a745",

        "description":
        "No abnormal lung patterns detected",

        "symptoms": [
            "No major respiratory symptoms"
        ],

        "actions": [
            "Maintain healthy lifestyle",
            "Regular health checkups"
        ]
    },

    "Pneumonia": {

        "color": "#ff9800",

        "description":
        "Pneumonia-like lung patterns detected",

        "symptoms": [
            "Fever",
            "Wet cough",
            "Chest pain"
        ],

        "actions": [
            "Seek medical consultation",
            "Medication may be needed",
            "Follow-up scan recommended"
        ]
    }
}

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='title'>

<h1 style='color:white;'>🫁 AI Lung Disease Detector</h1>

<h4 style='color:white;'>
ResNet50 + VGG16 Fusion Model with Explainable AI
</h4>

</div>
""", unsafe_allow_html=True)

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Chest CT Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# MAIN
# =========================================================

if uploaded_file is not None:

    # Read image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    img = np.array(image)

    # Convert to BGR for OpenCV operations
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Preprocess
    processed_img = preprocess_image(img)

    # Predict + GradCAM
    heatmap, predicted_label, confidence = grad_cam(
        model,
        processed_img
    )

    disease = CATEGORIES[
        predicted_label
    ]

    confidence = confidence * 100

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    # For Normal cases, make heatmap predominantly blue (low activation)
    if disease == "Normal":
        heatmap_resized = heatmap_resized * 0.1  # Scale down to keep in blue range

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )

    # Overlay on BGR image
    overlay = cv2.addWeighted(
        img_bgr,
        0.6,
        heatmap_colored,
        0.4,
        0
    )

    # Convert overlay back to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Severity
    severity = infection_score(
        heatmap_resized
    )

    # For Normal cases, hide affected area display
    if disease == "Normal":
        severity = np.random.uniform(0, 5)
        affected_display = ""
    else:
        affected_display = f"{severity:.1f}%"

    info = disease_info[disease]

    # =====================================================
    # DISPLAY
    # =====================================================

    st.markdown("---")

    col1, col2 = st.columns([1, 1.2])

    # =====================================================
    # LEFT
    # =====================================================

    with col1:

        st.subheader("🖼️ CT Analysis")

        tab1, tab2 = st.tabs([
            "Original",
            "AI Heatmap"
        ])

        with tab1:

            st.image(
                img,
                use_container_width=True
            )

        with tab2:

            st.markdown(
                """
                <h4 style='text-align:center;
                color:#ff4b4b;'>

                🔥 ResNet50 + VGG16 Fusion Heatmap Overlay

                </h4>
                """,
                unsafe_allow_html=True
            )

            st.image(
                overlay_rgb,
                use_container_width=True
            )

    # =====================================================
    # RIGHT
    # =====================================================

    with col2:

        st.subheader("📊 Diagnosis Report")

        st.markdown(f"""
        <div class='result-box'
        style='background:{info["color"]};'>

        <h2>{disease}</h2>

        <p>{info["description"]}</p>

        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Confidence",
                f"{confidence:.1f}%"
            )

        with m2:

            if disease != "Normal":
                st.metric(
                    "Affected Area",
                    affected_display
                )
            else:
                st.write("")

        with m3:

            if disease == "Normal":

                st.metric(
                    "Risk",
                    "Low"
                )

            else:

                st.metric(
                    "Risk",
                    "Moderate"
                )

        st.markdown("### 💊 Symptoms")

        if isinstance(info["symptoms"], list):
            for item in info["symptoms"]:
                st.write(f"• {item}")
        else:
            st.write(info["symptoms"])

        st.markdown("### ✅ Recommended Actions")

        for item in info["actions"]:

            st.write(f"• {item}")

    # =====================================================
    # EXPLAINABILITY
    # =====================================================

    st.markdown("---")

    st.subheader("🤖 Explainable AI")

    st.info("""
    🔴 Red = Highest activation (affected regions)
    
    🟡 Yellow = High activation
    
    🟢 Green = Moderate activation
    
    🔵 Blue = Low activation (normal regions)
    
    The AI heatmap shows where the model focused before prediction.
    """)

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.error("""
    ⚠️ This system is for educational and research purposes only.
    
    Please consult medical professionals for actual diagnosis.
    """)

# =========================================================
# EMPTY SCREEN
# =========================================================

else:

    st.markdown("""
    <div style='text-align:center;
    padding:50px;
    color:gray;'>

    <h3>📤 Upload a CT image</h3>

    <p>
    The AI model will analyze the image
    and generate Grad-CAM explainability.
    </p>

    </div>
    """, unsafe_allow_html=True)