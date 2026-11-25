"""
=====================================================================
 CULTURAL HERITAGE SITE CLASSIFIER - STREAMLIT APP
=====================================================================

Run:
    streamlit run app.py

Required files in the same folder:

    best_model.h5
    class_names.txt
    model_info.txt
    style.css       (optional)

Model trained using:
    User-provided augmented Heritage dataset

CNN architectures compared:
    1. MobileNetV2
    2. EfficientNetB0
    3. ResNet50V2
    4. DenseNet121
    5. Xception

The best-performing model is saved as:
    best_model.h5
=====================================================================
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Cultural Heritage AI",
    page_icon="🏛️",
    layout="wide"
)


# ================================================================
# CUSTOM CSS
# ================================================================

def load_css(css_path):

    if os.path.exists(css_path):

        with open(
            css_path,
            "r",
            encoding="utf-8"
        ) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

    else:

        st.info(
            "style.css not found. "
            "Running with default Streamlit styling."
        )


load_css("style.css")


# ================================================================
# FILE PATHS
# ================================================================

# ================================================================
# FILE PATHS
# ================================================================

MODEL_PATH = "best_model11.h5"
CLASSES_PATH = "class_names11.txt"
INFO_PATH = "modelinf.txt"


# ================================================================
# CHECK REQUIRED FILES
# ================================================================

required_files = [

    MODEL_PATH,
    CLASSES_PATH,
    INFO_PATH

]


missing_files = [

    file

    for file in required_files

    if not os.path.exists(file)

]


if missing_files:

    st.error(
        "Required files are missing:"
    )

    for file in missing_files:

        st.write(
            f"❌ {file}"
        )

    st.info(
        "Place these files in the same folder as app.py."
    )

    st.stop()


# ================================================================
# LOAD MODEL + CLASS NAMES + MODEL INFO
# ================================================================

@st.cache_resource
def load_everything():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # ------------------------------------------------------------
    # Class names
    # ------------------------------------------------------------

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = [

            line.strip()

            for line in f

            if line.strip()

        ]


    # ------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------

    info = {}

    with open(
        INFO_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if "=" in line:

                key, value = line.strip().split(
                    "=",
                    1
                )

                info[key] = value


    return (
        model,
        class_names,
        info
    )


model, CLASS_NAMES, MODEL_INFO = load_everything()


# ================================================================
# MODEL INFORMATION
# ================================================================

IMG_SIZE = int(
    MODEL_INFO.get(
        "input_size",
        224
    )
)

ARCH = MODEL_INFO.get(
    "architecture",
    "Unknown"
)

TEST_ACCURACY = MODEL_INFO.get(
    "test_accuracy",
    "N/A"
)


# ================================================================
# PREPROCESSING
# ================================================================

PREPROCESS_MAP = {

    "MobileNetV2":
        tf.keras.applications.mobilenet_v2.preprocess_input,

    "EfficientNetB0":
        tf.keras.applications.efficientnet.preprocess_input,

    "ResNet50V2":
        tf.keras.applications.resnet_v2.preprocess_input,

    "DenseNet121":
        tf.keras.applications.densenet.preprocess_input,

    "Xception":
        tf.keras.applications.xception.preprocess_input

}


preprocess_fn = PREPROCESS_MAP.get(
    ARCH
)


if preprocess_fn is None:

    st.warning(
        f"Unknown architecture '{ARCH}'. "
        "Using standard 0-1 scaling."
    )

    preprocess_fn = lambda x: x / 255.0


# ================================================================
# HERITAGE SITE INFORMATION
# ================================================================

SITE_INFO = {

    "Angkor_Wat": (
        "Khmer",
        "Angkor Wat, Cambodia"
    ),

    "Cahokia_Mounds": (
        "Mississippian",
        "Cahokia Mounds, USA"
    ),

    "Chan_Chan": (
        "Chimú",
        "Chan Chan, Peru"
    ),

    "Chichen_Itza": (
        "Maya-Toltec",
        "Chichen Itza, Mexico"
    ),

    "Giza_Necropolis": (
        "Ancient Egyptian",
        "Giza Necropolis, Egypt"
    ),

    "Karnak_Temple_Complex": (
        "Ancient Egyptian",
        "Karnak Temple Complex, Egypt"
    ),

    "Knossos_Palace": (
        "Minoan",
        "Palace of Knossos, Greece"
    ),

    "Machu_Picchu": (
        "Inca",
        "Machu Picchu, Peru"
    ),

    "Mohenjo_daro": (
        "Indus Valley",
        "Mohenjo-daro, Pakistan"
    ),

    "Persepolis": (
        "Achaemenid Persian",
        "Persepolis, Iran"
    ),

    "Petra": (
        "Nabataean",
        "Petra, Jordan"
    ),

    "Pompeii": (
        "Roman",
        "Pompeii, Italy"
    ),

    "Sanchi_Stupa_Complex": (
        "Ancient Indian",
        "Sanchi Stupa Complex, India"
    ),

    "Teotihuacan": (
        "Mesoamerican",
        "Teotihuacan, Mexico"
    ),

    # IMPORTANT:
    # Correct name from your new dataset
    "Ur_Tell_el_Muqayyar": (
        "Mesopotamian",
        "Ur (Tell el-Muqayyar), Iraq"
    )

}


# ================================================================
# HEADER
# ================================================================

st.title(
    "🏛️ Cultural Heritage AI"
)

st.subheader(
    "AI-Based Heritage Site Image Classification"
)

st.write(
    "Upload an image of a cultural heritage site "
    "and the trained CNN will predict the most likely "
    "heritage class."
)


# ================================================================
# MODEL INFORMATION
# ================================================================

st.divider()

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "CNN Architecture",
        ARCH
    )


with col2:

    st.metric(
        "Test Accuracy",
        TEST_ACCURACY
    )


with col3:

    st.metric(
        "Heritage Classes",
        len(CLASS_NAMES)
    )


# ================================================================
# IMAGE UPLOAD
# ================================================================

st.divider()

st.header(
    "📷 Upload Heritage Image"
)


uploaded_file = st.file_uploader(

    "Choose an image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]

)


# ================================================================
# PREDICTION
# ================================================================

if uploaded_file is not None:

    # ------------------------------------------------------------
    # Open image
    # ------------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )


    # ------------------------------------------------------------
    # Display image
    # ------------------------------------------------------------

    col_img, col_result = st.columns(
        [1, 1]
    )


    with col_img:

        st.image(

            image,

            caption="Uploaded Heritage Image",

            use_container_width=True

        )


    # ------------------------------------------------------------
    # Preprocess image
    # ------------------------------------------------------------

    img_resized = image.resize(

        (
            IMG_SIZE,
            IMG_SIZE
        )

    )


    img_array = np.array(

        img_resized,

        dtype=np.float32

    )


    # IMPORTANT:
    # Same preprocessing used during training

    img_array = preprocess_fn(
        img_array
    )


    img_array = np.expand_dims(

        img_array,

        axis=0

    )


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    with st.spinner(
        "🔍 Analyzing heritage image..."
    ):

        predictions = model.predict(

            img_array,

            verbose=0

        )[0]


    # ------------------------------------------------------------
    # Top predictions
    # ------------------------------------------------------------

    top_indices = np.argsort(

        predictions

    )[::-1][:5]


    best_index = top_indices[0]

    best_class = CLASS_NAMES[
        best_index
    ]

    confidence = float(
        predictions[
            best_index
        ]
    )


    # ------------------------------------------------------------
    # Heritage metadata
    # ------------------------------------------------------------

    civilization, location = SITE_INFO.get(

        best_class,

        (
            "Information unavailable",
            "Information unavailable"
        )

    )


    # ============================================================
    # RESULT
    # ============================================================

    with col_result:

        st.success(
            "Prediction completed!"
        )

        st.markdown(
            "### 🏛️ Predicted Heritage Site"
        )

        st.markdown(
            f"## {best_class.replace('_', ' ')}"
        )

        st.metric(

            "Confidence",

            f"{confidence * 100:.2f}%"

        )

        st.write(
            f"**Civilization:** {civilization}"
        )

        st.write(
            f"**Location:** {location}"
        )


    # ============================================================
    # CONFIDENCE MESSAGE
    # ============================================================

    st.divider()

    if confidence >= 0.80:

        st.success(
            "🟢 High-confidence prediction"
        )

    elif confidence >= 0.50:

        st.warning(
            "🟡 Moderate-confidence prediction"
        )

    else:

        st.error(
            "🔴 Low-confidence prediction"
        )


    # ============================================================
    # TOP 5 PREDICTIONS
    # ============================================================

    st.header(
        "📊 Top 5 Predictions"
    )


    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        class_name = CLASS_NAMES[
            idx
        ]

        probability = float(
            predictions[idx]
        )


        col_a, col_b = st.columns(
            [4, 1]
        )


        with col_a:

            st.write(
                f"**{rank}. "
                f"{class_name.replace('_', ' ')}**"
            )

            st.progress(
                probability
            )


        with col_b:

            st.write(
                f"**{probability * 100:.2f}%**"
            )


    # ============================================================
    # HERITAGE INFORMATION CARD
    # ============================================================

    st.divider()

    st.header(
        "🏺 Heritage Information"
    )


    info_col1, info_col2 = st.columns(
        2
    )


    with info_col1:

        st.write(
            "**Site:**"
        )

        st.write(
            best_class.replace(
                "_",
                " "
            )
        )


    with info_col2:

        st.write(
            "**Civilization / Culture:**"
        )

        st.write(
            civilization
        )


    st.write(
        f"**Location:** {location}"
    )


# ================================================================
# DATASET / MODEL INFORMATION
# ================================================================

st.divider()

with st.expander(
    "ℹ️ About the Model"
):

    st.write(
        f"**Architecture:** {ARCH}"
    )

    st.write(
        f"**Input Image Size:** "
        f"{IMG_SIZE} × {IMG_SIZE}"
    )

    st.write(
        f"**Number of Classes:** "
        f"{len(CLASS_NAMES)}"
    )

    st.write(
        f"**Test Accuracy:** "
        f"{TEST_ACCURACY}"
    )

    st.write(
        "**Training Dataset:** "
        "User-provided augmented heritage dataset"
    )

    st.write(
        "**Transfer Learning:** ImageNet"
    )

    st.write(
        "**CNN Models Compared:** "
        "MobileNetV2, EfficientNetB0, ResNet50V2, "
        "DenseNet121 and Xception"
    )


# ================================================================
# CLASS LIST
# ================================================================

with st.expander(
    "🏛️ Supported Heritage Sites"
):

    for i, class_name in enumerate(
        CLASS_NAMES,
        start=1
    ):

        st.write(
            f"{i}. "
            f"{class_name.replace('_', ' ')}"
        )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "🏛️ Cultural Heritage AI | "
    "Deep Learning + Transfer Learning + Computer Vision"
)