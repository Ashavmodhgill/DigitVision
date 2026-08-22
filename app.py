from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "digitvision_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        features = self.conv_layers(images)
        return self.fc_layer(features.view(features.size(0), -1))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = CNN().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def prepare_image(image):
    grayscale = ImageOps.grayscale(image).resize((28, 28), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    if pixels.mean() > 127:
        pixels = 255 - pixels
    pixels = Image.fromarray(pixels).resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(pixels, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor.sub(0.5).div(0.5).to(DEVICE), pixels


def predict(model, image):
    tensor, processed = prepare_image(image)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return int(probabilities.argmax()), probabilities, processed


st.set_page_config(
    page_title="DigitVision | Handwritten Digit AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#14201d; --muted:#71807a; --paper:#eef2e9; --mint:#d8f0df; --lime:#c7ef67; --coral:#f07e62; --line:#d7ded4; }
    .stApp { background-color:var(--paper); color:var(--ink); background-image:linear-gradient(rgba(20,32,29,.045) 1px, transparent 1px),linear-gradient(90deg, rgba(20,32,29,.045) 1px, transparent 1px); background-size:32px 32px; }
    [data-testid="stSidebar"] { background:#14201d; border-right:1px solid #30453c; }
    [data-testid="stSidebar"] * { color:#edf4e9 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size:1.5rem; letter-spacing:-.04em; }
    .block-container { max-width: 1240px; padding: 3.8rem 3.5rem 4.5rem; }
    h1, h2, h3, p, label, button { font-family:'Space Grotesk', sans-serif; }
    h1 { font-size: clamp(2.8rem, 6vw, 5.8rem); line-height:.91; letter-spacing:-.075em; margin:0; color:var(--ink); max-width:760px; }
    h2 { font-size:1.35rem; letter-spacing:-.04em; }
    .eyebrow { color:var(--coral); font:500 .72rem 'DM Mono', monospace; letter-spacing:.08em; text-transform:uppercase; margin-bottom:1.25rem; }
    .lede { color:var(--muted); font-size:1.05rem; max-width:560px; line-height:1.55; margin-top:1.4rem; }
    .hero { position:relative; padding:1rem 0 3.2rem; border-bottom:1px solid var(--line); margin-bottom:2.2rem; animation:rise .7s ease both; }
    .hero:after { content:'01'; position:absolute; right:2rem; bottom:2.5rem; color:#d7dfd2; font:500 7rem 'DM Mono',monospace; line-height:1; z-index:0; }
    .hero > * { position:relative; z-index:1; }
    .badge { display:inline-flex; align-items:center; gap:.55rem; background:var(--ink); border-radius:999px; padding:.5rem .85rem; color:var(--lime); font:500 .7rem 'DM Mono', monospace; margin-top:1.5rem; }
    .badge:first-letter { color:var(--coral); }
    .panel { background:rgba(255,255,255,.84); border:1px solid var(--line); border-radius:10px; padding:1.4rem; min-height:430px; box-shadow:0 18px 45px rgba(25,42,31,.08); animation:rise .7s .1s ease both; }
    .panel-title { display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; color:var(--ink); font-weight:600; }
    .panel-title span { color:var(--muted); font:400 .72rem 'DM Mono', monospace; }
    .result { background:var(--ink); color:#f7f9f0; border-radius:10px; padding:2rem; min-height:430px; display:flex; flex-direction:column; justify-content:space-between; box-shadow:0 22px 50px rgba(20,32,29,.2); animation:rise .7s .18s ease both; }
    .result .label { color:var(--lime); font:500 .75rem 'DM Mono', monospace; text-transform:uppercase; white-space:nowrap; }
    .digit { font-size:11rem; font-weight:700; line-height:.9; letter-spacing:-.1em; color:#f7f9f0; margin:.8rem 0; }
    .confidence { color:#c4d1c4; font:400 .84rem 'DM Mono', monospace; }
    .metric { border-top:1px solid #43524b; padding-top:1rem; display:flex; justify-content:space-between; color:#c4d1c4; font:400 .78rem 'DM Mono', monospace; }
    .metric strong { color:#f7f9f0; font-weight:500; }
    .section-label { color:var(--muted); font:500 .7rem 'DM Mono', monospace; letter-spacing:.08em; text-transform:uppercase; margin:2rem 0 .8rem; }
    .confidence-board { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:.7rem 1.4rem; padding:1.2rem 1.4rem 1.4rem; background:rgba(255,255,255,.72); border:1px solid var(--line); border-radius:10px; box-shadow:0 14px 35px rgba(25,42,31,.06); }
    .confidence-row { display:grid; grid-template-columns:2rem 1fr auto; align-items:center; gap:.7rem; min-width:0; color:var(--muted); font:500 .72rem 'DM Mono', monospace; }
    .confidence-row.is-top { color:var(--ink); }
    .confidence-digit { display:grid; place-items:center; width:2rem; height:2rem; border:1px solid var(--line); border-radius:5px; background:#f8faf5; color:var(--ink); }
    .is-top .confidence-digit { border-color:var(--ink); background:var(--ink); color:var(--lime); }
    .confidence-track { height:.45rem; overflow:hidden; border-radius:999px; background:#dce4da; }
    .confidence-fill { height:100%; min-width:2px; border-radius:inherit; background:#aab9ae; }
    .is-top .confidence-fill { background:var(--coral); }
    .confidence-value { min-width:4.4rem; text-align:right; color:var(--ink); }
    @media (max-width: 700px) { .confidence-board { grid-template-columns:1fr; } }
    .stButton > button { min-height:3rem; border-radius:6px; border:1px solid var(--ink); background:var(--ink); color:#fff; font-family:'Space Grotesk',sans-serif; font-weight:600; transition:transform .2s ease, background .2s ease; }
    .stButton > button:hover { background:#314840; border-color:#314840; transform:translateY(-2px); }
    [data-testid="stFileUploader"] { border:1px dashed #a9b8ac; border-radius:6px; padding:.5rem; background:rgba(255,255,255,.45); }
    [data-testid="stFileUploader"] section { padding:0; }
    iframe { border-radius:7px; box-shadow:inset 0 0 0 1px rgba(255,255,255,.15); }
    .stProgress > div > div { background-color:var(--lime); }
    .stProgress > div { background:#dce4da; }
    [data-testid="stSidebar"] .stCodeBlock { border:1px solid #3c5148; }
    @keyframes rise { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
    @media (max-width: 700px) { .block-container { padding:2rem 1.2rem 3rem; } .hero:after { right:0; font-size:4.5rem; bottom:2rem; } .digit { font-size:8rem; } .result { padding:1.4rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()

with st.sidebar:
    st.markdown("## ✦ DigitVision")
    st.caption("A compact CNN trained on MNIST")
    st.divider()
    st.markdown("**Model status**")
    if model is None:
        st.error("Weights not found")
        st.caption("Run the final notebook cell to export the trained model.")
    else:
        st.success("Model ready")
    st.markdown("**Architecture**")
    st.caption("3 convolutional blocks\n\n128 feature maps\n\n10 output classes")
    st.divider()
    st.caption("Inference device")
    st.code(str(DEVICE).upper(), language=None)

st.markdown('<div class="hero"><div class="eyebrow">MNIST / REAL-TIME INFERENCE</div><h1>What digit<br>did you draw?</h1><p class="lede">Give the model a handwritten digit and watch it turn pixels into a confident prediction.</p><div class="badge">● 99.24% test accuracy</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.08, .92], gap="large")
with left:
    st.markdown('<div class="panel"><div class="panel-title">Input canvas <span>28 × 28 normalized</span></div>', unsafe_allow_html=True)
    if st_canvas is None:
        st.warning("Install the requirements to enable the drawing canvas.")
        uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
        source_image = Image.open(uploaded) if uploaded else None
    else:
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#182522",
            height=360,
            width=360,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        source_image = Image.fromarray(canvas.image_data.astype("uint8")) if canvas.image_data is not None else None
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Or use an image</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload another digit image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if upload is not None:
        source_image = Image.open(upload)
    predict_clicked = st.button("Classify digit  →", use_container_width=True, disabled=model is None)

with right:
    st.markdown('<div class="result"><div><div class="label">Prediction</div>', unsafe_allow_html=True)
    if predict_clicked and source_image is not None:
        digit, probabilities, processed = predict(model, source_image)
        st.markdown(f'<div class="digit">{digit}</div><div class="confidence">{probabilities[digit] * 100:.2f}% confidence</div>', unsafe_allow_html=True)
        st.image(processed, width=110, caption="Processed input")
        st.markdown(f'<div class="metric"><span>Top alternative</span><strong>{int(np.argsort(probabilities)[-2])} · {probabilities[np.argsort(probabilities)[-2]] * 100:.2f}%</strong></div>', unsafe_allow_html=True)
    elif model is None:
        st.markdown('<div class="digit">—</div><div class="confidence">Awaiting exported model weights</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="digit">?</div><div class="confidence">Draw a digit, then classify it</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Class confidence</div>', unsafe_allow_html=True)
if predict_clicked and source_image is not None and model is not None:
    predicted_digit, probabilities, _ = predict(model, source_image)
    confidence_rows = []
    for index in np.argsort(probabilities)[::-1]:
        probability = float(probabilities[index])
        top_class = " is-top" if index == predicted_digit else ""
        confidence_rows.append(
            f'<div class="confidence-row{top_class}"><span class="confidence-digit">{index}</span>'
            f'<span class="confidence-track"><span class="confidence-fill" style="width:{probability * 100:.2f}%"></span></span>'
            f'<span class="confidence-value">{probability * 100:.2f}%</span></div>'
        )
    st.markdown(f'<div class="confidence-board">{"".join(confidence_rows)}</div>', unsafe_allow_html=True)
else:
    st.caption("Prediction probabilities will appear here after inference.")
