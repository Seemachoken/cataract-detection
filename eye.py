import os
import json
from urllib.request import urlopen
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models are downloaded from Hugging Face on Render
MODEL_DIR = os.path.join(
    "/tmp",
    "cataract_models"
)

CNN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn_model.h5"
)

MOBILENET_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenet_model.h5"
)

# Hugging Face model URLs
CNN_MODEL_URL = (
    "https://huggingface.co/seema-choken/"
    "cataract-detection-models/resolve/main/cnn_model.h5"
)

MOBILENET_MODEL_URL = (
    "https://huggingface.co/seema-choken/"
    "cataract-detection-models/resolve/main/mobilenet_model.h5"
)

# Render needs 0.0.0.0
HOST = "0.0.0.0"

# Render automatically provides PORT.
# When running locally, it will use 8000.
PORT = int(
    os.environ.get(
        "PORT",
        8000
    )
)

IMG_SIZE = (224, 224)


# ============================================================
# MODEL ACCURACIES
# ============================================================

MODEL_ACCURACIES = {
    "CNN": 92.82,
    "MobileNetV2": 96.41,
    "ResNet50": 92.62,
    "EfficientNetB0": 48.11,
    "Hybrid CNN + MobileNetV2": 96.61,
}


# ============================================================
# HYBRID MODEL WEIGHTS
# ============================================================

CNN_WEIGHT = 0.20
MOBILENET_WEIGHT = 0.80


# ============================================================
# GLOBAL MODELS
# ============================================================

cnn_model = None
mobilenet_model = None


# ============================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

def download_model_if_needed(
    model_path,
    model_url
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Do not download again if model already exists
    if (
        os.path.exists(model_path)
        and
        os.path.getsize(model_path) > 0
    ):

        print(
            "Model already downloaded:",
            model_path
        )

        return


    print()
    print(
        "Downloading model from Hugging Face..."
    )

    print(
        model_url
    )

    print()


    try:

        with urlopen(
            model_url,
            timeout=600
        ) as response:

            with open(
                model_path,
                "wb"
            ) as file:

                while True:

                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:

                        break

                    file.write(
                        chunk
                    )


        print(
            "Model downloaded successfully:",
            model_path
        )


    except Exception as e:

        print(
            "Model download error:",
            e
        )

        if os.path.exists(
            model_path
        ):

            try:

                os.remove(
                    model_path
                )

            except Exception:

                pass

        raise


# ============================================================
# LOAD MODELS
# ============================================================

def load_project_models():

    global cnn_model
    global mobilenet_model


    # ---------------- Download CNN ----------------

    print(
        "Preparing CNN model..."
    )

    download_model_if_needed(
        CNN_MODEL_PATH,
        CNN_MODEL_URL
    )


    # ---------------- Download MobileNetV2 ----------------

    print(
        "Preparing MobileNetV2 model..."
    )

    download_model_if_needed(
        MOBILENET_MODEL_PATH,
        MOBILENET_MODEL_URL
    )


    # ---------------- CNN ----------------

    try:

        cnn_model = load_model(
            CNN_MODEL_PATH,
            compile=False
        )

        print(
            "CNN model loaded successfully."
        )

    except Exception as e:

        cnn_model = None

        print(
            "CNN model error:",
            e
        )


    # ---------------- MobileNetV2 ----------------

    try:

        mobilenet_model = load_model(
            MOBILENET_MODEL_PATH,
            compile=False
        )

        print(
            "MobileNetV2 model loaded successfully."
        )

    except Exception as e:

        mobilenet_model = None

        print(
            "MobileNetV2 model error:",
            e
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_bytes):

    from io import BytesIO

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(
        array,
        axis=0
    )


# ============================================================
# GET MODEL PROBABILITY
# ============================================================

def get_probability(
    model,
    image_array
):

    prediction = model.predict(
        image_array,
        verbose=0
    )

    value = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return max(
        0.0,
        min(
            1.0,
            value
        )
    )


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_bytes):

    if (
        cnn_model is None
        or
        mobilenet_model is None
    ):

        raise RuntimeError(
            "CNN or MobileNetV2 model could not be loaded. "
            "Please check the Hugging Face model files."
        )

    image_array = prepare_image(
        image_bytes
    )

    # CNN prediction
    cnn_score = get_probability(
        cnn_model,
        image_array
    )

    # MobileNetV2 prediction
    mobilenet_score = get_probability(
        mobilenet_model,
        image_array
    )

    # Hybrid prediction
    hybrid_score = (
        CNN_WEIGHT * cnn_score
        +
        MOBILENET_WEIGHT * mobilenet_score
    )

    if hybrid_score > 0.5:

        label = "Normal"
        confidence = hybrid_score

    else:

        label = "Cataract"
        confidence = 1.0 - hybrid_score

    return {
        "label": label,
        "confidence": round(
            confidence,
            4
        ),
        "cnn_score": round(
            cnn_score,
            4
        ),
        "mobilenet_score": round(
            mobilenet_score,
            4
        ),
        "hybrid_score": round(
            hybrid_score,
            4
        ),
        "model": "Hybrid CNN + MobileNetV2",
        "model_accuracy": MODEL_ACCURACIES[
            "Hybrid CNN + MobileNetV2"
        ],
    }


# ============================================================
# GET UPLOADED FILE
# ============================================================

def get_uploaded_file(handler):

    content_type = handler.headers.get(
        "Content-Type",
        ""
    )

    content_length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    if content_length <= 0:

        raise ValueError(
            "No image was uploaded."
        )

    body = handler.rfile.read(
        content_length
    )

    if "multipart/form-data" not in content_type:

        raise ValueError(
            "Please upload using multipart/form-data."
        )

    if "boundary=" not in content_type:

        raise ValueError(
            "Upload boundary was not found."
        )

    boundary = content_type.split(
        "boundary=",
        1
    )[1].strip()

    if (
        boundary.startswith('"')
        and
        boundary.endswith('"')
    ):

        boundary = boundary[1:-1]

    boundary_bytes = (
        "--" + boundary
    ).encode(
        "utf-8"
    )

    for part in body.split(
        boundary_bytes
    ):

        if b'name="file"' not in part:

            continue

        header_end = part.find(
            b"\r\n\r\n"
        )

        if header_end == -1:

            continue

        file_data = part[
            header_end + 4:
        ]

        file_data = file_data.rstrip(
            b"\r\n-"
        )

        if file_data:

            return file_data

    raise ValueError(
        "No file field was found in the upload."
    )


# ============================================================
# HTML PAGE
# ============================================================

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Cataract Detection | Medical AI</title>

<style>

*{
    box-sizing:border-box;
    margin:0;
    padding:0
}

body{
    font-family:Inter,Arial,sans-serif;
    background:#f7f9fc;
    color:#172033;
    min-height:100vh
}

.navbar{
    height:72px;
    background:rgba(255,255,255,.96);
    border-bottom:1px solid #e7ebf2;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 7%;
    position:sticky;
    top:0;
    z-index:10
}

.logo{
    display:flex;
    align-items:center;
    gap:10px;
    font-size:20px;
    font-weight:800
}

.logo-icon{
    width:38px;
    height:38px;
    border-radius:12px;
    display:grid;
    place-items:center;
    background:linear-gradient(135deg,#5b5cf0,#8c63ff);
    color:white
}

.nav-tag{
    color:#68738a;
    font-size:13px;
    font-weight:600
}

.hero{
    text-align:center;
    padding:60px 20px 35px
}

.badge{
    display:inline-block;
    background:#eef0ff;
    color:#5557d9;
    padding:8px 15px;
    border-radius:999px;
    font-size:12px;
    font-weight:800;
    letter-spacing:.5px;
    margin-bottom:18px
}

.hero h1{
    font-size:clamp(34px,5vw,58px);
    line-height:1.08;
    margin-bottom:16px;
    letter-spacing:-1.8px
}

.gradient-text{
    background:linear-gradient(90deg,#5557df,#8b5cf6);
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent
}

.hero p{
    max-width:700px;
    margin:auto;
    color:#69758b;
    font-size:16px;
    line-height:1.7
}

.container{
    width:min(1120px,92%);
    margin:0 auto 70px
}

.main-grid{
    display:grid;
    grid-template-columns:1.1fr .9fr;
    gap:24px
}

.card{
    background:white;
    border:1px solid #e8ecf3;
    border-radius:24px;
    box-shadow:0 15px 45px rgba(32,43,70,.07)
}

.upload-card,
.result-card,
.accuracy-section,
.steps{
    padding:28px
}

.section-title{
    font-size:20px;
    font-weight:800;
    margin-bottom:7px
}

.section-subtitle{
    color:#788399;
    font-size:14px;
    margin-bottom:22px
}

.dropzone{
    min-height:290px;
    border:2px dashed #cdd3e3;
    border-radius:20px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    padding:30px;
    text-align:center;
    cursor:pointer;
    background:#fbfcff
}

.dropzone:hover,
.dropzone.dragover{
    border-color:#7774ef;
    background:#f6f5ff
}

.upload-icon{
    width:66px;
    height:66px;
    border-radius:20px;
    display:grid;
    place-items:center;
    background:#efefff;
    color:#5b5cf0;
    font-size:29px;
    margin-bottom:16px
}

.dropzone h3{
    font-size:17px;
    margin-bottom:7px
}

.dropzone p{
    color:#8a94a7;
    font-size:13px;
    line-height:1.6
}

#fileInput{
    display:none
}

.preview{
    display:none;
    margin-top:18px;
    border-radius:18px;
    overflow:hidden;
    background:#f5f7fb;
    text-align:center;
    padding:15px
}

.preview img{
    max-width:100%;
    max-height:300px;
    border-radius:14px;
    object-fit:contain
}

.actions{
    display:flex;
    gap:12px;
    margin-top:18px
}

button{
    border:none;
    cursor:pointer;
    border-radius:13px;
    padding:13px 20px;
    font-size:14px;
    font-weight:800
}

.analyze-btn{
    flex:1;
    color:white;
    background:linear-gradient(135deg,#5b5ce9,#845cf6)
}

.analyze-btn:disabled{
    opacity:.5;
    cursor:not-allowed
}

.clear-btn{
    background:#eef1f6;
    color:#4d586d
}

.loading{
    display:none;
    text-align:center;
    padding:15px;
    color:#656fdc;
    font-weight:700;
    font-size:14px
}

.result-box{
    min-height:240px;
    border-radius:20px;
    background:linear-gradient(145deg,#fafbff,#f5f6ff);
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    padding:25px;
    text-align:center
}

.result-icon{
    width:72px;
    height:72px;
    border-radius:50%;
    display:grid;
    place-items:center;
    font-size:30px;
    background:#eceeff;
    color:#5c5fe2;
    margin-bottom:15px
}

.result-label{
    font-size:28px;
    font-weight:900;
    margin-bottom:6px
}

.result-confidence{
    color:#68738a;
    font-size:14px
}

.confidence-wrap{
    width:100%;
    margin-top:20px
}

.confidence-label{
    display:flex;
    justify-content:space-between;
    color:#68738a;
    font-size:12px;
    font-weight:700;
    margin-bottom:7px
}

.progress{
    height:9px;
    background:#e8ebf2;
    border-radius:99px;
    overflow:hidden
}

.progress-bar{
    height:100%;
    width:0%;
    background:linear-gradient(90deg,#5b5ce9,#8a5cf6);
    border-radius:inherit
}

.score-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:10px;
    width:100%;
    margin-top:18px
}

.score{
    background:white;
    border:1px solid #e8ebf3;
    border-radius:13px;
    padding:12px 8px
}

.score small{
    display:block;
    color:#8992a4;
    font-size:10px;
    margin-bottom:4px
}

.score strong{
    font-size:14px
}

.accuracy-section{
    margin-top:24px
}

.accuracy-grid{
    display:grid;
    grid-template-columns:repeat(5,1fr);
    gap:12px;
    margin-top:20px
}

.accuracy-item{
    padding:18px 12px;
    background:#fafbff;
    border:1px solid #e8ebf3;
    border-radius:16px;
    text-align:center
}

.accuracy-value{
    font-size:23px;
    font-weight:900;
    color:#5a5cdf
}

.accuracy-name{
    color:#707b90;
    font-size:11px;
    line-height:1.4;
    margin-top:6px
}

.steps{
    margin-top:24px
}

.steps-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:15px;
    margin-top:20px
}

.step{
    padding:20px;
    border-radius:17px;
    background:#fafbff;
    border:1px solid #e8ebf3
}

.step-number{
    width:34px;
    height:34px;
    border-radius:10px;
    background:#eceeff;
    color:#5b5cdf;
    display:grid;
    place-items:center;
    font-weight:900;
    margin-bottom:13px
}

.step h3{
    font-size:15px;
    margin-bottom:7px
}

.step p{
    color:#7c8799;
    font-size:13px;
    line-height:1.6
}

.disclaimer{
    margin-top:24px;
    padding:18px 20px;
    background:#fff9ec;
    border:1px solid #f3e0b4;
    border-radius:16px;
    color:#755f2d;
    font-size:12px;
    line-height:1.7
}

.error{
    display:none;
    margin-top:15px;
    padding:13px 15px;
    border-radius:12px;
    background:#fff0f0;
    color:#b53b3b;
    border:1px solid #f2caca;
    font-size:13px;
    line-height:1.5
}

footer{
    text-align:center;
    padding:25px;
    color:#929bad;
    font-size:12px
}

@media(max-width:850px){

    .main-grid{
        grid-template-columns:1fr
    }

    .accuracy-grid{
        grid-template-columns:repeat(2,1fr)
    }

    .steps-grid{
        grid-template-columns:1fr
    }
}

@media(max-width:520px){

    .navbar{
        padding:0 5%
    }

    .nav-tag{
        display:none
    }

    .hero{
        padding-top:40px
    }

    .upload-card,
    .result-card,
    .accuracy-section,
    .steps{
        padding:20px
    }

    .actions{
        flex-direction:column
    }

    .score-grid{
        grid-template-columns:1fr
    }
}

</style>

</head>

<body>

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            ✦
        </div>

        <span>
            Cataract AI
        </span>

    </div>

    <div class="nav-tag">
        AI-Powered Eye Screening Research Prototype
    </div>

</nav>

<section class="hero">

    <div class="badge">
        MEDICAL AI • RESEARCH PROTOTYPE
    </div>

    <h1>
        Smart
        <span class="gradient-text">
            Cataract Detection
        </span>
    </h1>

    <p>
        Upload a fundus-eye image and use the trained Hybrid CNN + MobileNetV2 model to generate a research prediction.
    </p>

</section>

<main class="container">

<div class="main-grid">

<section class="card upload-card">

    <div class="section-title">
        Upload Eye Image
    </div>

    <div class="section-subtitle">
        Use a clear fundus-eye image in JPG, JPEG, or PNG format.
    </div>

    <label
        class="dropzone"
        id="dropzone"
    >

        <input
            id="fileInput"
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        >

        <div class="upload-icon">
            ↑
        </div>

        <h3>
            Click to upload or drag & drop
        </h3>

        <p>
            Supported file types: JPG, JPEG, PNG
        </p>

    </label>

    <div
        class="preview"
        id="preview"
    >

        <img
            id="previewImage"
            alt="Eye image preview"
        >

    </div>

    <div
        class="error"
        id="errorBox"
    ></div>

    <div
        class="loading"
        id="loading"
    >
        Analyzing image with the trained model...
    </div>

    <div class="actions">

        <button
            class="analyze-btn"
            id="analyzeBtn"
            disabled
        >
            Analyze Image
        </button>

        <button
            class="clear-btn"
            id="clearBtn"
        >
            Clear
        </button>

    </div>

</section>

<section class="card result-card">

    <div class="section-title">
        Prediction Result
    </div>

    <div class="section-subtitle">
        Result from the Hybrid CNN + MobileNetV2 model.
    </div>

    <div class="result-box">

        <div
            class="result-icon"
            id="resultIcon"
        >
            ◉
        </div>

        <div
            class="result-label"
            id="resultLabel"
        >
            Waiting for image
        </div>

        <div
            class="result-confidence"
            id="resultConfidence"
        >
            Upload an eye image to begin.
        </div>

        <div class="confidence-wrap">

            <div class="confidence-label">

                <span>
                    Confidence
                </span>

                <span id="confidenceText">
                    0%
                </span>

            </div>

            <div class="progress">

                <div
                    class="progress-bar"
                    id="progressBar"
                ></div>

            </div>

        </div>

        <div class="score-grid">

            <div class="score">

                <small>
                    CNN Score
                </small>

                <strong id="cnnScore">
                    —
                </strong>

            </div>

            <div class="score">

                <small>
                    MobileNetV2
                </small>

                <strong id="mobilenetScore">
                    —
                </strong>

            </div>

            <div class="score">

                <small>
                    Hybrid Score
                </small>

                <strong id="hybridScore">
                    —
                </strong>

            </div>

        </div>

    </div>

</section>

</div>

<section class="card accuracy-section">

    <div class="section-title">
        Model Performance
    </div>

    <div class="section-subtitle">
        Measured test accuracies from this project.
    </div>

    <div class="accuracy-grid">

        <div class="accuracy-item">

            <div class="accuracy-value">
                92.82%
            </div>

            <div class="accuracy-name">
                CNN
            </div>

        </div>

        <div class="accuracy-item">

            <div class="accuracy-value">
                96.41%
            </div>

            <div class="accuracy-name">
                MobileNetV2
            </div>

        </div>

        <div class="accuracy-item">

            <div class="accuracy-value">
                92.62%
            </div>

            <div class="accuracy-name">
                ResNet50
            </div>

        </div>

        <div class="accuracy-item">

            <div class="accuracy-value">
                48.11%
            </div>

            <div class="accuracy-name">
                EfficientNetB0
            </div>

        </div>

        <div class="accuracy-item">

            <div class="accuracy-value">
                96.61%
            </div>

            <div class="accuracy-name">
                Hybrid CNN + MobileNetV2
            </div>

        </div>

    </div>

</section>

<section class="card steps">

    <div class="section-title">
        How It Works
    </div>

    <div class="section-subtitle">
        Simple three-step research workflow.
    </div>

    <div class="steps-grid">

        <div class="step">

            <div class="step-number">
                1
            </div>

            <h3>
                Upload
            </h3>

            <p>
                Select a clear fundus-eye image from your computer.
            </p>

        </div>

        <div class="step">

            <div class="step-number">
                2
            </div>

            <h3>
                AI Analysis
            </h3>

            <p>
                The backend preprocesses the image and runs the trained models.
            </p>

        </div>

        <div class="step">

            <div class="step-number">
                3
            </div>

            <h3>
                Result
            </h3>

            <p>
                The system displays the predicted class and confidence.
            </p>

        </div>

    </div>

</section>

<div class="disclaimer">

    <strong>
        Medical Disclaimer:
    </strong>

    This application is a research/academic prototype and is not a medical diagnosis tool. Do not use the result to make medical decisions. Please consult a qualified eye-care professional for diagnosis and treatment.

</div>

</main>

<footer>

    Cataract Detection • BTech AIML Academic Project • Research Prototype

</footer>

<script>

const fileInput =
    document.getElementById(
        "fileInput"
    );

const dropzone =
    document.getElementById(
        "dropzone"
    );

const preview =
    document.getElementById(
        "preview"
    );

const previewImage =
    document.getElementById(
        "previewImage"
    );

const analyzeBtn =
    document.getElementById(
        "analyzeBtn"
    );

const clearBtn =
    document.getElementById(
        "clearBtn"
    );

const loading =
    document.getElementById(
        "loading"
    );

const errorBox =
    document.getElementById(
        "errorBox"
    );

let selectedFile = null;


dropzone.addEventListener(
    "click",
    function () {

        fileInput.click();

    }
);


fileInput.addEventListener(
    "change",
    function (event) {

        if (
            event.target.files.length > 0
        ) {

            setFile(
                event.target.files[0]
            );

        }

    }
);


dropzone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropzone.classList.add(
            "dragover"
        );

    }
);


dropzone.addEventListener(
    "dragleave",
    function () {

        dropzone.classList.remove(
            "dragover"
        );

    }
);


dropzone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropzone.classList.remove(
            "dragover"
        );

        if (
            event.dataTransfer.files.length > 0
        ) {

            setFile(
                event.dataTransfer.files[0]
            );

        }

    }
);


function setFile(file) {

    hideError();

    const validTypes = [
        "image/jpeg",
        "image/png"
    ];


    if (
        !validTypes.includes(
            file.type
        )
    ) {

        showError(
            "Please select a JPG, JPEG, or PNG image."
        );

        return;

    }


    selectedFile = file;


    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            previewImage.src =
                event.target.result;

            preview.style.display =
                "block";

        };


    reader.readAsDataURL(
        file
    );


    analyzeBtn.disabled =
        false;


    document.getElementById(
        "resultLabel"
    ).textContent =
        "Ready to analyze";


    document.getElementById(
        "resultConfidence"
    ).textContent =
        "Click Analyze Image to run the model.";

}


clearBtn.addEventListener(
    "click",
    clearAll
);


function clearAll() {

    selectedFile = null;

    fileInput.value = "";

    preview.style.display =
        "none";

    previewImage.src =
        "";

    analyzeBtn.disabled =
        true;

    loading.style.display =
        "none";

    hideError();


    document.getElementById(
        "resultLabel"
    ).textContent =
        "Waiting for image";


    document.getElementById(
        "resultConfidence"
    ).textContent =
        "Upload an eye image to begin.";


    document.getElementById(
        "confidenceText"
    ).textContent =
        "0%";


    document.getElementById(
        "progressBar"
    ).style.width =
        "0%";


    document.getElementById(
        "cnnScore"
    ).textContent =
        "—";


    document.getElementById(
        "mobilenetScore"
    ).textContent =
        "—";


    document.getElementById(
        "hybridScore"
    ).textContent =
        "—";

}


analyzeBtn.addEventListener(
    "click",
    async function () {

        if (!selectedFile) {

            showError(
                "Please upload an image first."
            );

            return;

        }


        hideError();

        loading.style.display =
            "block";

        analyzeBtn.disabled =
            true;


        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );

            }


            showResult(
                data
            );


        } catch (error) {

            showError(
                error.message ||
                "Could not connect to the prediction server."
            );


        } finally {

            loading.style.display =
                "none";

            analyzeBtn.disabled =
                false;

        }

    }
);


function showResult(data) {

    const confidence =
        Number(
            data.confidence || 0
        );


    const percentage =
        Math.round(
            confidence * 100
        );


    document.getElementById(
        "resultLabel"
    ).textContent =
        data.label;


    document.getElementById(
        "resultConfidence"
    ).textContent =
        "Model confidence: " +
        percentage +
        "%";


    document.getElementById(
        "confidenceText"
    ).textContent =
        percentage +
        "%";


    document.getElementById(
        "progressBar"
    ).style.width =
        percentage +
        "%";


    document.getElementById(
        "cnnScore"
    ).textContent =
        (
            Number(
                data.cnn_score
            ) * 100
        ).toFixed(1) +
        "%";


    document.getElementById(
        "mobilenetScore"
    ).textContent =
        (
            Number(
                data.mobilenet_score
            ) * 100
        ).toFixed(1) +
        "%";


    document.getElementById(
        "hybridScore"
    ).textContent =
        (
            Number(
                data.hybrid_score
            ) * 100
        ).toFixed(1) +
        "%";


    document.getElementById(
        "resultIcon"
    ).textContent =
        data.label.toLowerCase() ===
        "cataract"
            ? "!"
            : "✓";

}


function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";

}


function hideError() {

    errorBox.textContent =
        "";

    errorBox.style.display =
        "none";

}

</script>

</body>

</html>
'''


# ============================================================
# HTTP HANDLER
# ============================================================

class CataractHandler(
    BaseHTTPRequestHandler
):


    # --------------------------------------------------------
    # SEND JSON RESPONSE
    # --------------------------------------------------------

    def send_json(
        self,
        status_code,
        data
    ):

        response = json.dumps(
            data
        ).encode(
            "utf-8"
        )


        self.send_response(
            status_code
        )


        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )


        self.send_header(
            "Content-Length",
            str(len(response))
        )


        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )


        self.end_headers()


        self.wfile.write(
            response
        )


    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    def do_OPTIONS(self):

        self.send_response(
            204
        )


        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )


        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS"
        )


        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )


        self.end_headers()


    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        path = urlparse(
            self.path
        ).path


        # Main page
        if path in (
            "/",
            "/index.html"
        ):

            page = HTML_PAGE.encode(
                "utf-8"
            )


            self.send_response(
                200
            )


            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )


            self.send_header(
                "Content-Length",
                str(len(page))
            )


            self.end_headers()


            self.wfile.write(
                page
            )


            return


        # Health check
        if path == "/health":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "models_loaded": (
                        cnn_model is not None
                        and
                        mobilenet_model is not None
                    )
                }
            )


            return


        # Unknown page
        self.send_json(
            404,
            {
                "error": "Page not found."
            }
        )


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        path = urlparse(
            self.path
        ).path


        if path != "/predict":

            self.send_json(
                404,
                {
                    "error": "Endpoint not found."
                }
            )

            return


        try:

            image_bytes = get_uploaded_file(
                self
            )


            result = predict_image(
                image_bytes
            )


            self.send_json(
                200,
                result
            )


        except Exception as e:

            print(
                "Prediction error:",
                e
            )


            self.send_json(
                500,
                {
                    "error": str(e)
                }
            )


    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    def log_message(
        self,
        format_string,
        *args
    ):

        print(
            "[SERVER]",
            format_string % args
        )


# ============================================================
# START SERVER
# ============================================================

def start_server():

    print(
        "Loading project models..."
    )


    load_project_models()


    print(
        "Starting server..."
    )


    server = HTTPServer(
        (
            HOST,
            PORT
        ),
        CataractHandler
    )


    print()
    print(
        "=" * 60
    )

    print(
        "CATARACT DETECTION WEB APP"
    )

    print(
        "=" * 60
    )

    print(
        "Server Host:",
        HOST
    )

    print(
        "Server Port:",
        PORT
    )

    print(
        "Prediction API: /predict"
    )

    print(
        "Health Check: /health"
    )

    print(
        "=" * 60
    )

    print(
        "Server started successfully."
    )

    print(
        "Waiting for requests..."
    )

    print()


    try:

        server.serve_forever()


    except KeyboardInterrupt:

        print(
            "\nServer stopped."
        )


    finally:

        server.server_close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start_server()