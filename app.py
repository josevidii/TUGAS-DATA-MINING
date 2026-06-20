import streamlit as st
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.transforms as tv_transforms
import librosa
import numpy as np
import tempfile
import os

from torchvision import models

# ====================================
# CONFIG
# ====================================

st.set_page_config(
    page_title="ESC-50 Audio Classification",
    page_icon="🎵",
    layout="centered"
)

SAMPLE_RATE = 44100
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
TOP_DB      = 80

# ====================================
# LABELS & CATEGORIES
# ====================================

CLASS_NAMES = [
    'Dog', 'Rooster', 'Pig', 'Cow', 'Frog',
    'Cat', 'Hen', 'Insects', 'Sheep', 'Crow',
    'Rain', 'Sea waves', 'Crackling fire', 'Crickets', 'Chirping birds',
    'Water drops', 'Wind', 'Pouring water', 'Toilet flush', 'Thunderstorm',
    'Crying baby', 'Sneezing', 'Clapping', 'Breathing', 'Coughing',
    'Footsteps', 'Laughing', 'Brushing teeth', 'Snoring', 'Drinking - sipping',
    'Door knock', 'Mouse click', 'Keyboard typing', 'Door - wood creaks', 'Can opening',
    'Washing machine', 'Vacuum cleaner', 'Clock alarm', 'Clock tick', 'Glass breaking',
    'Helicopter', 'Chainsaw', 'Siren', 'Car horn', 'Engine',
    'Train', 'Church bells', 'Airplane', 'Fireworks', 'Hand saw'
]

CATEGORY_MAP = {
    'Dog': ('Animals', '🐾'), 'Rooster': ('Animals', '🐾'), 'Pig': ('Animals', '🐾'),
    'Cow': ('Animals', '🐾'), 'Frog': ('Animals', '🐾'), 'Cat': ('Animals', '🐾'),
    'Hen': ('Animals', '🐾'), 'Insects': ('Animals', '🐾'), 'Sheep': ('Animals', '🐾'),
    'Crow': ('Animals', '🐾'),
    'Rain': ('Natural Soundscapes', '🌿'), 'Sea waves': ('Natural Soundscapes', '🌿'),
    'Crackling fire': ('Natural Soundscapes', '🌿'), 'Crickets': ('Natural Soundscapes', '🌿'),
    'Chirping birds': ('Natural Soundscapes', '🌿'), 'Water drops': ('Natural Soundscapes', '🌿'),
    'Wind': ('Natural Soundscapes', '🌿'), 'Pouring water': ('Natural Soundscapes', '🌿'),
    'Toilet flush': ('Natural Soundscapes', '🌿'), 'Thunderstorm': ('Natural Soundscapes', '🌿'),
    'Crying baby': ('Human Sounds', '🗣️'), 'Sneezing': ('Human Sounds', '🗣️'),
    'Clapping': ('Human Sounds', '🗣️'), 'Breathing': ('Human Sounds', '🗣️'),
    'Coughing': ('Human Sounds', '🗣️'), 'Footsteps': ('Human Sounds', '🗣️'),
    'Laughing': ('Human Sounds', '🗣️'), 'Brushing teeth': ('Human Sounds', '🗣️'),
    'Snoring': ('Human Sounds', '🗣️'), 'Drinking - sipping': ('Human Sounds', '🗣️'),
    'Door knock': ('Interior & Domestic', '🏠'), 'Mouse click': ('Interior & Domestic', '🏠'),
    'Keyboard typing': ('Interior & Domestic', '🏠'), 'Door - wood creaks': ('Interior & Domestic', '🏠'),
    'Can opening': ('Interior & Domestic', '🏠'), 'Washing machine': ('Interior & Domestic', '🏠'),
    'Vacuum cleaner': ('Interior & Domestic', '🏠'), 'Clock alarm': ('Interior & Domestic', '🏠'),
    'Clock tick': ('Interior & Domestic', '🏠'), 'Glass breaking': ('Interior & Domestic', '🏠'),
    'Helicopter': ('Exterior & Urban', '🏙️'), 'Chainsaw': ('Exterior & Urban', '🏙️'),
    'Siren': ('Exterior & Urban', '🏙️'), 'Car horn': ('Exterior & Urban', '🏙️'),
    'Engine': ('Exterior & Urban', '🏙️'), 'Train': ('Exterior & Urban', '🏙️'),
    'Church bells': ('Exterior & Urban', '🏙️'), 'Airplane': ('Exterior & Urban', '🏙️'),
    'Fireworks': ('Exterior & Urban', '🏙️'), 'Hand saw': ('Exterior & Urban', '🏙️'),
}

CATEGORY_COLORS = {
    'Animals':             '#6366f1',
    'Natural Soundscapes': '#10b981',
    'Human Sounds':        '#f59e0b',
    'Interior & Domestic': '#3b82f6',
    'Exterior & Urban':    '#ef4444',
}

BAR_COLORS = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#ddd6fe']

# ====================================
# CSS — satu blok bersih
# ====================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Base ---- */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main > div {
    background: #eef0fb !important;
    font-family: 'Inter', sans-serif !important;
}

/* ---- Remove default streamlit padding/gap ---- */
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }
[data-testid="stVerticalBlock"] > div { gap: 0.6rem !important; }

/* ---- Header ---- */
.app-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
}
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #64748b;
    font-size: 0.92rem;
    margin-top: 0.35rem;
}

/* ========================================
   UPLOAD SECTION — card via st.container
   ======================================== */

/* ---- Card wrapper (st.container border=True) ---- */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stVerticalBlockBorderWrapper"] > div > div,
[data-testid="stVerticalBlockBorderWrapper"] > div > div > div {
    background: #ffffff !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1.5px solid #e0e7ff !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.10), 0 1px 4px rgba(0,0,0,0.04) !important;
    overflow: hidden !important;
    padding: 0 !important;
    transition: box-shadow 0.25s ease, transform 0.25s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 12px 40px rgba(99,102,241,0.16), 0 2px 8px rgba(0,0,0,0.06) !important;
    transform: translateY(-2px) !important;
}

/* inner padding */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.4rem 1.6rem !important;
}

/* section label inside card */
.card-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.9rem;
}

/* ---- File uploader strip ---- */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #f5f7ff !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #818cf8 !important;
    background: #f0f2ff !important;
}
[data-testid="stFileUploaderFile"] {
    background: #f0f2ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderFileName"] {
    font-weight: 600 !important;
    color: #3730a3 !important;
}
[data-testid="stFileUploaderFileSize"] { color: #94a3b8 !important; }

/* ---- Divider inside card ---- */
.card-divider {
    height: 1px;
    background: #f1f5f9;
    margin: 1rem 0;
}

/* ---- Audio player ---- */
[data-testid="stAudio"],
[data-testid="stAudio"] > div,
[data-testid="stAudio"] > div > div {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stAudio"] audio {
    width: 100%;
    border-radius: 8px;
    accent-color: #6366f1;
    background: #ffffff !important;
}

/* ---- Predict button ---- */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 3px 12px rgba(99,102,241,0.3) !important;
    transition: opacity 0.15s, transform 0.15s !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ---- Spinner ---- */
[data-testid="stSpinner"] p { color: #6366f1 !important; font-weight: 500; }

/* ---- Prediction result card (HTML) ---- */
.pred-card {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 1.2rem;
}
.pred-icon { font-size: 2.8rem; line-height: 1; }
.pred-label {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 0.15rem;
}
.pred-name { font-size: 1.55rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
.pred-cat {
    font-size: 0.78rem; font-weight: 600;
    padding: 0.2rem 0.65rem; border-radius: 999px;
    display: inline-block; margin-top: 0.3rem;
}

/* ---- Confidence ---- */
.conf-section { margin-bottom: 1.2rem; }
.section-label {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 0.4rem;
}
.conf-value { font-size: 1.9rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; }
.conf-track {
    background: #f1f5f9; border-radius: 999px;
    height: 10px; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #818cf8);
}

/* ---- Top 5 bars ---- */
.bar-row {
    background: #f8faff;
    border: 1px solid #e8ecff;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.45rem;
    transition: box-shadow 0.2s, transform 0.2s;
}
.bar-row:hover {
    box-shadow: 0 4px 16px rgba(99,102,241,0.12);
    transform: translateY(-1px);
}
.bar-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.bar-name { font-size: 0.88rem; font-weight: 600; color: #1e293b; }
.bar-badge {
    font-size: 0.68rem; font-weight: 600;
    padding: 0.12rem 0.5rem; border-radius: 999px;
    margin-left: 0.4rem;
}
.bar-pct { font-size: 0.85rem; font-weight: 700; color: #475569; }
.bar-track { background: #e8ecff; border-radius: 999px; height: 7px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ====================================
# TRANSFORMS
# ====================================

mel_transform    = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, n_mels=N_MELS,
                                    power=2.0, normalized=True)
to_db            = T.AmplitudeToDB(stype='power', top_db=TOP_DB)
compute_delta    = T.ComputeDeltas(win_length=9)
imagenet_norm    = tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
resize_transform = tv_transforms.Resize((224, 224))

# ====================================
# MODEL
# ====================================

@st.cache_resource
def load_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    ckpt = torch.load("best_model.pth", map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

model = load_model()

# ====================================
# PREPROCESS & PREDICT
# ====================================

def preprocess_audio(path):
    audio_np = librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    mel = mel_transform(waveform)
    mel = to_db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    d1  = compute_delta(mel)
    d2  = compute_delta(d1)
    x   = torch.cat([mel, d1, d2], dim=0)
    x   = resize_transform(x)
    x   = imagenet_norm(x)
    return x.unsqueeze(0)

def predict(path):
    x = preprocess_audio(path)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred_idx   = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    top_probs, top_idx = torch.topk(probs, 5)
    return CLASS_NAMES[pred_idx], confidence, top_probs, top_idx

# ====================================
# UI
# ====================================

st.markdown("""
<div class="app-header">
  <h1>🎵 ESC-50 Audio Classifier</h1>
  <p>Upload a WAV file to classify environmental sounds using ResNet50</p>
</div>
""", unsafe_allow_html=True)

# ---- Upload card (st.container renders as stVerticalBlockBorderWrapper) ----
with st.container(border=True):
    st.markdown('<div class="card-label">🎙️ Audio Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload WAV File", type=["wav"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
        st.audio(uploaded_file)
        st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
        predict_btn = st.button("🔍  Analyze Sound", use_container_width=True)
    else:
        predict_btn = False

# ---- Results ----
if uploaded_file is not None and predict_btn:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        try:
            label, confidence, top_probs, top_idx = predict(temp_path)
            cat, cat_emoji = CATEGORY_MAP.get(label, ('Unknown', '❓'))
            cat_color      = CATEGORY_COLORS.get(cat, '#6366f1')

            with st.container(border=True):
                # Prediction
                st.markdown(f"""
                <div class="pred-card">
                  <div class="pred-icon">{cat_emoji}</div>
                  <div>
                    <div class="pred-label">Predicted Sound</div>
                    <div class="pred-name">{label}</div>
                    <span class="pred-cat"
                          style="background:{cat_color}18;color:{cat_color};">{cat}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence
                st.markdown(f"""
                <div class="conf-section">
                  <div class="section-label">Confidence</div>
                  <div class="conf-value">{confidence:.1%}</div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{confidence*100:.1f}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Top 5
                st.markdown('<div class="section-label">Top 5 Predictions</div>',
                            unsafe_allow_html=True)

                for i, (p, idx) in enumerate(zip(top_probs, top_idx)):
                    name    = CLASS_NAMES[idx.item()]
                    pct     = p.item()
                    c, _    = CATEGORY_MAP.get(name, ('Unknown', ''))
                    c_color = CATEGORY_COLORS.get(c, '#6366f1')

                    st.markdown(f"""
                    <div class="bar-row">
                      <div class="bar-meta">
                        <div>
                          <span class="bar-name">{name}</span>
                          <span class="bar-badge"
                                style="background:{c_color}18;color:{c_color};">{c}</span>
                        </div>
                        <span class="bar-pct">{pct:.1%}</span>
                      </div>
                      <div class="bar-track">
                        <div class="bar-fill"
                             style="width:{pct*100:.1f}%;background:{BAR_COLORS[i]};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
        finally:
            os.unlink(temp_path)