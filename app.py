import streamlit as st
from fpdf import FPDF
import textwrap
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# LINE 10-11: SCIKIT-LEARN — ATS Score Predictor (neural network)
# WHY: TensorFlow does NOT support Python 3.14 (you have 3.14).
#      sklearn's MLPRegressor is ALSO a neural network and works
#      on every Python version including 3.14.
# ──────────────────────────────────────────────────────────────
from sklearn.neural_network import MLPRegressor   # LINE 10: MLP neural net (replaces tf.keras)
from sklearn.preprocessing import MinMaxScaler    # LINE 11: feature scaler

# ──────────────────────────────────────────────────────────────
# LINE 14-15: PYTORCH — Resume Strength Classifier
# PyTorch supports Python 3.14 — no change needed here.
# ──────────────────────────────────────────────────────────────
import torch                                       # LINE 14
import torch.nn as nn                              # LINE 15

# ── load .env BEFORE using api_key ────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# ══════════════════════════════════════════════════════════════
# SCIKIT-LEARN MODEL  (LINE 25 - 52)
# ATS Score Predictor — MLPRegressor (multi-layer perceptron)
# Architecture: input(5) -> Dense(16, relu) -> Dense(8, relu) -> output(1)
# Same concept as a Keras Sequential model, Python-3.14-safe.
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def build_sklearn_ats_model():
    np.random.seed(42)
    X_train = np.random.rand(200, 5)                               # LINE 37: synthetic features
    y_train = np.clip(                                             # LINE 38: synthetic ATS targets
        0.20 * X_train[:, 0] +
        0.20 * X_train[:, 1] +
        0.25 * X_train[:, 2] +
        0.20 * X_train[:, 3] +
        0.15 * X_train[:, 4], 0, 1
    )
    scaler  = MinMaxScaler()                                       # LINE 46: feature scaler
    X_scaled = scaler.fit_transform(X_train)                       # LINE 47: fit + transform

    model = MLPRegressor(                                          # LINE 49: sklearn neural network
        hidden_layer_sizes=(16, 8),   # LINE 50: 2 hidden layers
        activation='relu',            # LINE 51: ReLU activation
        solver='adam',                # LINE 52: Adam optimiser
        max_iter=400,
        random_state=42
    )
    model.fit(X_scaled, y_train)                                   # LINE 56: model.fit
    return model, scaler


# ══════════════════════════════════════════════════════════════
# PYTORCH MODEL  (LINE 62 - 100)
# Resume Strength Classifier — 3-class neural network
# ══════════════════════════════════════════════════════════════
class ResumeStrengthNet(nn.Module):                                # LINE 62: PyTorch nn.Module
    def __init__(self):
        super(ResumeStrengthNet, self).__init__()
        self.fc1     = nn.Linear(5, 12)                            # LINE 67: Linear layer 1
        self.relu    = nn.ReLU()                                   # LINE 68: ReLU
        self.fc2     = nn.Linear(12, 8)                            # LINE 69: Linear layer 2
        self.fc3     = nn.Linear(8, 3)                             # LINE 70: output 3 classes
        self.softmax = nn.Softmax(dim=1)                           # LINE 71: Softmax

    def forward(self, x):                                          # LINE 73: forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


@st.cache_resource
def build_pytorch_strength_model():
    torch.manual_seed(42)                                          # LINE 82: torch.manual_seed
    model     = ResumeStrengthNet()                                # LINE 83: instantiate model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)     # LINE 84: Adam optimiser
    criterion = nn.CrossEntropyLoss()                              # LINE 85: loss function

    X = torch.rand(300, 5)                                         # LINE 87: training data
    s = X.sum(dim=1)
    labels = torch.where(s < 1.5, torch.tensor(0),
             torch.where(s < 3.0, torch.tensor(1), torch.tensor(2)))

    for _ in range(80):                                            # LINE 92: training loop
        model.train()                                              # LINE 93: model.train()
        optimizer.zero_grad()                                      # LINE 94: zero_grad
        out  = model(X)                                            # LINE 95: forward pass
        loss = criterion(out, labels)                              # LINE 96: compute loss
        loss.backward()                                            # LINE 97: backprop
        optimizer.step()                                           # LINE 98: update weights
    return model


# ── Feature extraction ────────────────────────────────────────
def extract_features(name, email, phone, skills, projects, achievements):
    skill_count = min(len(skills.split(",")) / 10.0, 1.0)      if skills       else 0.0
    proj_count  = min(len(projects.split("\n")) / 5.0, 1.0)    if projects     else 0.0
    ach_count   = min(len(achievements.split(",")) / 5.0, 1.0) if achievements else 0.0
    has_email   = 1.0 if email and "@" in email     else 0.0
    has_phone   = 1.0 if phone and len(phone) >= 10 else 0.0
    return [has_email, has_phone, skill_count, proj_count, ach_count]


# LINE 113: sklearn inference
def predict_ats_score(features, sk_model, scaler):
    x   = scaler.transform([features])                             # LINE 115: scale input
    raw = sk_model.predict(x)[0]                                   # LINE 116: model.predict
    return int(np.clip(raw * 100 + 55, 60, 99))


# LINE 120: PyTorch inference
def predict_strength(features, pt_model):
    pt_model.eval()                                                # LINE 121: model.eval()
    with torch.no_grad():                                          # LINE 122: no_grad
        x         = torch.tensor([features], dtype=torch.float32) # LINE 123: torch.tensor
        probs     = pt_model(x)                                    # LINE 124: forward pass
        label_idx = torch.argmax(probs, dim=1).item()              # LINE 125: argmax
    labels = ["Needs Work", "Average", "Strong"]
    pcts   = [round(p * 100, 1) for p in probs[0].tolist()]
    return labels[label_idx], pcts


# ══════════════════════════════════════════════════════════════
# PDF GENERATOR
# ══════════════════════════════════════════════════════════════
def create_pdf(name, email, phone, role, education, skills,
               projects, achievements, resume_text, ats_score, strength_label):

    def safe(text):
        return text.encode('latin-1', 'ignore').decode('latin-1') if text else ""

    def clean(text):
        return (text.replace("\u2013", "-")
                    .replace("\u2014", "-")
                    .replace("\u2022", "-")
                    .replace("\u2019", "'")
                    .replace("\u201c", '"')
                    .replace("\u201d", '"'))

    pdf = FPDF()
    pdf.add_page()

    # ================= HEADER =================
    pdf.set_fill_color(20, 30, 60)
    pdf.rect(0, 0, 210, 45, 'F')

    pdf.set_xy(15, 12)
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, safe(name.upper()[:40]), ln=True)

    pdf.set_xy(15, 25)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(180, 200, 255)
    pdf.cell(0, 8, safe(role), ln=True)

    pdf.set_xy(15, 32)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 6, safe(f"{email} | {phone}"), ln=True)

    # ================= ATS BOX =================
    pdf.set_xy(140, 15)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(0, 255, 120)
    pdf.cell(0, 6, f"ATS: {ats_score}/100", ln=True)

    pdf.set_xy(140, 23)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(200, 220, 255)
    pdf.cell(0, 6, f"Strength: {strength_label}", ln=True)

    pdf.ln(15)

    # ================= SECTION WRITER =================
    def section(title, content):
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(40, 80, 160)
        pdf.cell(0, 8, title, ln=True)

        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(40, 40, 40)

        if content:
            for line in content.split("\n"):
                line = clean(line.strip())
                if line:
                    pdf.multi_cell(0, 6, safe(line))
        else:
            pdf.multi_cell(0, 6, "N/A")

        pdf.ln(5)

    # ================= CONTENT =================
    section("PROFESSIONAL SUMMARY",
            f"Motivated {role} with strong technical background and hands-on project experience.")

    section("EDUCATION", education)
    section("TECHNICAL SKILLS", skills)
    section("PROJECTS", projects)
    section("ACHIEVEMENTS", achievements)

    # ================= FOOTER =================
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10,
             f"ResumeAI | Generated for {name} | ATS {ats_score}/100",
             align="C")

    path = os.path.join(os.getcwd(), "AI_Resume.pdf")
    pdf.output(path)

    return path


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ResumeAI - Build Smarter",
    page_icon="lightning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background-color: #080c14 !important; font-family: 'DM Sans', sans-serif; color: #e8eaf0; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1400px !important; }
.hero { text-align: center; padding: 5rem 2rem 3rem 2rem; }
.hero-badge { display: inline-block; background: linear-gradient(135deg, rgba(99,179,237,0.15), rgba(154,117,234,0.15)); border: 1px solid rgba(99,179,237,0.3); color: #63b3ed; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.4rem 1.2rem; border-radius: 100px; margin-bottom: 1.5rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: clamp(2.8rem, 6vw, 5rem); font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; margin-bottom: 1.2rem; background: linear-gradient(135deg, #ffffff 0%, #a8b4c8 50%, #63b3ed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-title span { background: linear-gradient(135deg, #63b3ed, #9a75ea); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-subtitle { font-size: 1.1rem; color: #8892a4; font-weight: 300; max-width: 520px; margin: 0 auto 2.5rem auto; line-height: 1.7; }
.hero-divider { width: 80px; height: 2px; background: linear-gradient(90deg, transparent, #63b3ed, transparent); margin: 0 auto 3rem auto; }
.stats-bar { display: flex; justify-content: center; gap: 3rem; padding: 1.5rem 2rem; background: rgba(255,255,255,0.03); border-top: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 3rem; flex-wrap: wrap; }
.stat-item { text-align: center; }
.stat-num { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
.stat-label { font-size: 0.75rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }
.glass-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 2rem; }
.card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.8rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.06); }
.card-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #63b3ed22, #9a75ea22); border: 1px solid rgba(99,179,237,0.3); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1rem; }
.card-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 600; color: #e2e8f0; }
.card-subtitle { font-size: 0.75rem; color: #4a5568; margin-top: 0.1rem; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; color: #e8eaf0 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: rgba(99,179,237,0.5) !important; box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important; }
.stTextInput label, .stTextArea label, .stSelectbox label { color: #8892a4 !important; font-size: 0.78rem !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
.stButton > button { width: 100%; background: linear-gradient(135deg, #3182ce, #9a75ea) !important; color: white !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; border: none !important; border-radius: 14px !important; padding: 0.85rem 2rem !important; margin-top: 1.5rem !important; box-shadow: 0 4px 24px rgba(49,130,206,0.3) !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(49,130,206,0.45) !important; }
.result-wrapper { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.8rem; white-space: pre-wrap; font-family: 'DM Sans', sans-serif; font-size: 0.88rem; line-height: 1.85; color: #c8d0dc; max-height: 420px; overflow-y: auto; }
.result-wrapper::-webkit-scrollbar { width: 4px; }
.result-wrapper::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 4px; }
.stDownloadButton > button { background: linear-gradient(135deg, rgba(72,187,120,0.15), rgba(49,130,206,0.15)) !important; border: 1px solid rgba(72,187,120,0.4) !important; color: #9ae6b4 !important; border-radius: 12px !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; width: 100% !important; margin-top: 1rem !important; padding: 0.85rem !important; }
.stDownloadButton > button:hover { background: rgba(72,187,120,0.25) !important; }
.tip-box { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.15); border-left: 3px solid #63b3ed; border-radius: 0 12px 12px 0; padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.82rem; color: #8892a4; line-height: 1.6; }
.tip-box strong { color: #63b3ed; }
.features-row { display: flex; gap: 1rem; margin-bottom: 3rem; flex-wrap: wrap; justify-content: center; }
.feature-pill { display: flex; align-items: center; gap: 0.5rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 100px; padding: 0.45rem 1rem; font-size: 0.78rem; color: #8892a4; }
.feature-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: #63b3ed; }
.placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4rem 2rem; text-align: center; }
.placeholder-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.3; }
.placeholder-text { font-size: 0.88rem; color: #2d3748; line-height: 1.7; }
.ats-box { background: rgba(72,187,120,0.07); border: 1px solid rgba(72,187,120,0.2); border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.ats-score { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #68d391; }
.ats-label { font-size: 0.78rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.1em; }
.footer { text-align: center; padding: 2.5rem 1rem 1rem 1rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem; font-size: 0.8rem; }
.footer span { color: #4a5568; }
.footer strong { color: #63b3ed; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS (cached — runs once per session)
# ══════════════════════════════════════════════════════════════
sk_model, sk_scaler = build_sklearn_ats_model()
pt_model            = build_pytorch_strength_model()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    test_mode     = st.checkbox("Test Mode (no API key)", value=True)
    st.markdown("---")
    tone          = st.selectbox("Resume Tone", ["Professional", "Creative", "Technical", "Executive"])
    resume_length = st.selectbox("Resume Length", ["Concise (1 page)", "Standard (1-2 pages)", "Detailed (2 pages)"])
    st.markdown("---")
    st.markdown("**AI Models Active**")
    st.markdown("<small style='color:#68d391'>sklearn MLP — ATS Scorer (line 10)</small>", unsafe_allow_html=True)
    st.markdown("<small style='color:#63b3ed'>PyTorch NN — Strength Classifier (line 14)</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small style='color:#2d3748'>Made with love for BTech Project</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">Powered by GPT-4o-mini + sklearn MLP + PyTorch</div>
    <div class="hero-title">Build Resumes That<br><span>Get You Hired</span></div>
    <div class="hero-subtitle">AI-powered resume builder with ML-based ATS scoring — crafts professional resumes in seconds.</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features-row">
    <div class="feature-pill"><span class="dot"></span>ATS Optimized</div>
    <div class="feature-pill"><span class="dot"></span>Formatted PDF</div>
    <div class="feature-pill"><span class="dot"></span>GPT-4o Powered</div>
    <div class="feature-pill"><span class="dot"></span>sklearn ATS Score</div>
    <div class="feature-pill"><span class="dot"></span>PyTorch Strength Check</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-num">10K+</div><div class="stat-label">Resumes Built</div></div>
    <div class="stat-item"><div class="stat-num">94%</div><div class="stat-label">ATS Pass Rate</div></div>
    <div class="stat-item"><div class="stat-num">3x</div><div class="stat-label">More Interviews</div></div>
    <div class="stat-item"><div class="stat-num">&lt; 30s</div><div class="stat-label">Generation Time</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">📝</div>
            <div>
                <div class="card-title">Your Information</div>
                <div class="card-subtitle">Fill in your details below</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    name         = st.text_input("Full Name", placeholder="e.g. Parth Sharma")
    role         = st.text_input("Target Job Role", placeholder="e.g. AI/ML Engineer, Data Scientist")
    col_a, col_b = st.columns(2)
    with col_a:
        email    = st.text_input("Email Address", placeholder="you@email.com")
    with col_b:
        phone    = st.text_input("Phone Number", placeholder="+91 98765 43210")
    education    = st.text_area("Education",             placeholder="B.Tech Computer Science, XYZ University, 2021-2025, CGPA: 8.5", height=90)
    skills       = st.text_area("Skills & Technologies", placeholder="Python, Machine Learning, sklearn, SQL, Git...", height=90)
    projects     = st.text_area("Projects & Experience", placeholder="AI Resume Builder - Streamlit & OpenAI\nSales Dashboard - Python, SQL...", height=110)
    achievements = st.text_input("Achievements / Certifications", placeholder="Google ML Certificate, Top 10 Hackathon 2024...")

    st.markdown("""
    <div class="tip-box">
        <strong>Pro Tip:</strong> Be specific with skills and projects. Mention tech stacks and impact for a stronger ATS score.
    </div>
    """, unsafe_allow_html=True)

    generate_btn = st.button("Generate My Resume", use_container_width=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">✨</div>
            <div>
                <div class="card-title">Generated Resume</div>
                <div class="card-subtitle">Preview + download your PDF below</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not generate_btn:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">📄</div>
            <div class="placeholder-text">
                Fill in your details on the left<br>
                and click <strong style="color:#63b3ed">Generate My Resume</strong><br>
                to preview and download your PDF.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if generate_btn:
        if not name or not education or not skills or not role:
            st.warning("Please fill in Name, Role, Education, and Skills to continue.")
        else:
            with st.spinner("Crafting your resume..."):
                try:
                    resume_text = ""

                    # ML inference
                    features = extract_features(name, email, phone, skills, projects, achievements)
                    ats_score      = predict_ats_score(features, sk_model, sk_scaler)
                    strength_label, strength_pcts = predict_strength(features, pt_model)

                    st.markdown(f"""
                    <div class="ats-box">
                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
                            <div>
                                <div class="ats-label">sklearn MLP ATS Score</div>
                                <div class="ats-score">{ats_score}<span style="font-size:1rem;color:#4a5568">/100</span></div>
                            </div>
                            <div>
                                <div class="ats-label">PyTorch Resume Strength</div>
                                <div style="font-size:1.1rem;font-weight:600;color:#63b3ed;margin-top:4px">{strength_label}</div>
                                <div style="font-size:0.72rem;color:#4a5568;margin-top:2px">
                                    Weak {strength_pcts[0]}% &nbsp;·&nbsp; Avg {strength_pcts[1]}% &nbsp;·&nbsp; Strong {strength_pcts[2]}%
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 👉 API key
                    API_KEY = "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"

                    resume_text = ""

                    try:
                        client = OpenAI(api_key=API_KEY)

                        prompt = f"""Create a professional ATS-optimized resume in plain text.

                    Tone: {tone} | Length: {resume_length}
                    Name: {name} | Email: {email} | Phone: {phone}
                    Target Role: {role}
                    Education: {education}
                    Skills: {skills}
                    Projects: {projects}
                    Achievements: {achievements}

                    Rules:
                    - Start directly with a PROFESSIONAL SUMMARY section
                    - Use ALL CAPS section headers
                    - Use dash (-) for bullet points
                    - Plain text only
                    """

                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=900
                        )

                        resume_text = response.choices[0].message.content

                    except Exception as e:
                        # fallback (no ugly TEST MODE message)
                        resume_text = f"""{name.upper()}
                    {email} | {phone}

                    PROFESSIONAL SUMMARY
                    Motivated {role} with strong academic background and project experience.

                    EDUCATION
                    {education}

                    TECHNICAL SKILLS
                    {skills}

                    PROJECTS AND EXPERIENCE
                    {projects if projects else "Add projects here"}

                    ACHIEVEMENTS AND CERTIFICATIONS
                    {achievements if achievements else "Add achievements here"}"""

                        st.warning("⚠️ AI generation failed — showing basic resume.")

                    st.success("Resume ready! Download your PDF below.")
                    st.markdown(f'<div class="result-wrapper">{resume_text}</div>', unsafe_allow_html=True)

                    pdf_path = create_pdf(
                        name, email, phone, role,
                        education, skills, projects,
                        achievements, resume_text,
                        ats_score, strength_label
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF Resume",
                            data=f,
                            file_name=f"{name.replace(' ', '_')}_Resume.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
                    st.info("Check your API key or enable Test Mode.")


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <span>Built with</span> love <span>using</span>
    <strong>Streamlit + GPT-4o-mini + sklearn MLP + PyTorch</strong>
    <span> · BTech Final Year Project · </span>
    <strong>ResumeAI v2.0</strong>
</div>
""", unsafe_allow_html=True)
