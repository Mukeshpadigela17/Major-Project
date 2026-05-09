"""
Enhanced Streamlit App - LLM Ensemble Banking System (Simplified Version)
Includes multimodal analysis with text, image, and ML components
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ensemble_voter import EnsembleVoter
    from text_analyzer import TextAnalyzer
    from image_processor import ImageProcessor
    ENSEMBLE_AVAILABLE = True
except:
    ENSEMBLE_AVAILABLE = False

st.set_page_config(page_title="AI Banking Ensemble", layout="wide")

# Comprehensive Styling for Text Visibility
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 45%, #0d9488 100%);
    color: #f8fafc;
}

.stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #e0f2fe;
}

/* All text elements - ensure visibility on dark background */
label, .stText, .stMarkdown, .stCaption, p, span, div, strong, em, small, li, a {
    color: #f8fafc !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #e0f2fe !important;
}

/* Font sizes for headings */
h1 { font-size: 3rem !important; }
h2 { font-size: 2.4rem !important; }
h3 { font-size: 2rem !important; }

/* Input controls - dark text on light background */
input, textarea, select, option, .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stRadio {
    color: #0f172a !important;
    background: #fdfdfd !important;
    border-color: #ddd !important;
}

/* Labels for all inputs */
.stSlider label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stFileUploader label, .stRadio label {
    color: #f8fafc !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

/* File uploader container - make it MORE visible - override all white backgrounds */
[data-testid="stFileUploader"] {
    border: 3px dashed #0ea5e9 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    background: rgba(30, 58, 138, 0.5) !important;
    box-shadow: inset 0 0 12px rgba(14, 165, 233, 0.2) !important;
}

/* File uploader drag area - DARK background */
[data-testid="stFileUploader"] > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 10px !important;
    border: 2px dashed rgba(14, 165, 233, 0.4) !important;
}

/* Remove any white backgrounds from uploader */
[data-testid="stFileUploader"] * {
    background: transparent !important;
}

/* Placeholder text in file uploader */
[data-testid="stFileUploader"] .stText p {
    color: #a5f3fc !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* File uploader text styling */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {
    color: #e0f2fe !important;
}

/* Browse button inside file uploader */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    padding: 10px 20px !important;
    border: 2px solid #047857 !important;
    border-radius: 8px !important;
}

/* Text input placeholders */
.stTextArea textarea::placeholder,
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #94a3b8 !important;
    font-style: italic !important;
}

/* Focus states for better visibility */
.stNumberInput input:focus,
.stTextArea textarea:focus,
.stTextInput input:focus {
    border: 2px solid #0ea5e9 !important;
    box-shadow: 0 0 8px rgba(14, 165, 233, 0.3) !important;
}

/* Sidebar labels */
.sidebar .stSlider label, .sidebar .stNumberInput label, .sidebar .stRadio label {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Tab labels - ensure visible */
.stTabs [data-baseweb="tab-list"] button {
    color: #e0f2fe !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #fff !important;
    font-weight: 800 !important;
}

/* Card styling */
.section-card {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 12px 26px rgba(8, 8, 24, 0.35);
}

/* Tab title */
.tab-title {
    font-size: 2.6rem;
    font-weight: 900;
    color: #f8fafc;
    text-align: center;
}

/* Button styling - ENHANCED VISIBILITY */
.stButton button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    padding: 16px 24px !important;
    border: 3px solid #0369a1 !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s ease !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    box-shadow: 0 12px 24px rgba(2, 132, 199, 0.5) !important;
    transform: translateY(-2px) !important;
}

.stButton button:active {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    transform: translateY(0) !important;
}

/* Clear Chat button special styling */
[data-testid="column"] button[key="clear_chat_btn"] {
    background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important;
    border: 3px solid #dc2626 !important;
}

[data-testid="column"] button[key="clear_chat_btn"]:hover {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    box-shadow: 0 12px 24px rgba(239, 68, 68, 0.5) !important;
}

/* File uploader button styling */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 12px 20px !important;
    border: 3px solid #047857 !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 10px 20px rgba(16, 185, 129, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* File uploader container label text */
.stFileUploader label {
    color: #e0f2fe !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
}

/* Metrics and data displays */
[data-testid="metric-container"] {
    background: rgba(0, 0, 0, 0.2);
    padding: 20px;
    border-radius: 10px;
}

/* Tables */
.stDataFrame {
    color: #f8fafc !important;
}

/* Error and success messages */
.stSuccess, .stError, .stWarning, .stInfo {
    color: #f8fafc !important;
}

/* Expander headers */
.streamlit-expanderHeader {
    color: #f8fafc !important;
}

/* Sidebar styling - override defaults for visibility */
[data-testid="stSidebar"] {
    background: #f0f4f8 !important;
}

[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

.sidebar .stText, .sidebar .stMarkdown, .sidebar label {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Tab buttons */
button[role="tab"] {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}

button[role="tab"][aria-selected="true"] {
    color: #a5f3fc !important;
    border-bottom: 4px solid #38bdf8 !important;
}

/* Input container styling */
.input-container {
    background: rgba(30, 58, 138, 0.5);
    border: 2px solid rgba(148, 163, 184, 0.4);
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
}

.input-label {
    color: #e0f2fe !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    margin-bottom: 8px !important;
}

.input-field {
    width: 100% !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-weight: 600 !important;
}

.input-field:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1) !important;
}

/* Result container styling */
.result-container {
    background: rgba(15, 23, 42, 0.6);
    border: 2px solid rgba(34, 197, 94, 0.3);
    border-radius: 15px;
    padding: 25px;
    margin-top: 10px;
}

/* Ensure number inputs are visible */
.stNumberInput input, .stTextArea textarea {
    color: #0f172a !important;
    background: #ffffff !important;
    border: 2px solid #cbd5e1 !important;
}

.stNumberInput label, .stTextArea label, .stFileUploader label {
    color: #e0f2fe !important;
    font-weight: 700 !important;
}

/* Enhanced heading styling */
.prediction-heading {
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    color: #e0f2fe !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5) !important;
    margin: 20px 0 15px 0 !important;
    padding: 10px 0 !important;
    border-bottom: 3px solid #38bdf8 !important;
}

.section-heading {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #a5f3fc !important;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
    margin: 15px 0 12px 0 !important;
}

.metric-card {
    background: rgba(56, 189, 248, 0.1) !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin: 10px 0 !important;
}

.info-box {
    background: rgba(6, 182, 212, 0.15) !important;
    border-left: 4px solid #06b6d4 !important;
    padding: 12px 15px !important;
    border-radius: 8px !important;
    margin: 8px 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Advanced LLM Ensemble Banking System")
st.markdown("**AI-Powered Multimodal Customer Identification**")

# Load models
@st.cache_resource
def load_all():
    try:
        from pathlib import Path
        
        # Get the directory of the current script
        app_dir = Path(__file__).parent
        project_root = app_dir.parent
        
        with open(project_root / "models" / "customer_model.pkl", "rb") as f:
            ml_model = pickle.load(f)
        with open(project_root / "models" / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Compatibility: older scikit-learn persistent models may omit multi_class attr
        if hasattr(ml_model, 'coef_') and not hasattr(ml_model, 'multi_class'):
            ml_model.multi_class = 'ovr'

        ensemble = EnsembleVoter(model=ml_model, scaler=scaler) if ENSEMBLE_AVAILABLE else None
        text_analyzer = TextAnalyzer() if ENSEMBLE_AVAILABLE else None
        image_processor = ImageProcessor() if ENSEMBLE_AVAILABLE else None
        
        df = pd.read_csv(project_root / "data" / "bank.csv")
        df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
        df = pd.get_dummies(df, drop_first=True)
        X_cols = df.drop("deposit", axis=1).columns
        
        return ml_model, scaler, ensemble, text_analyzer, image_processor, X_cols
    except Exception as e:
        st.error(f"Error loading: {e}")
        return None, None, None, None, None, None

ml_model, scaler, ensemble, text_analyzer, image_processor, X_cols = load_all()


def safe_predict_proba(model, X):
    if hasattr(model, 'coef_') and not hasattr(model, 'multi_class'):
        model.multi_class = 'ovr'

    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)
        except Exception:
            pass

    if hasattr(model, 'decision_function'):
        df = model.decision_function(X)
        if df.ndim == 1:
            p = 1 / (1 + np.exp(-df))
            return np.vstack([1 - p, p]).T
        if df.ndim == 2:
            exp = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

    preds = model.predict(X)
    return np.vstack([1 - preds, preds]).T


if ml_model is None:
    st.error("❌ Run `python main.py` first to train models")
    st.stop()

# Session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Sidebar
st.sidebar.header("⚙️ Settings")
ml_wt = st.sidebar.slider("ML Weight", 0.0, 1.0, 0.5)
txt_wt = st.sidebar.slider("Text Weight", 0.0, 1.0, 0.25)
img_wt = st.sidebar.slider("Image Weight", 0.0, 1.0, 0.25)
threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.7)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Ensemble Predict", "📊 Dashboard", "🤖 Chatbot", "ℹ️ Info"])

# TAB 1: Ensemble Predictions
with tab1:
    st.markdown("<div class='tab-title'>🎯 Multimodal Predictions</div>", unsafe_allow_html=True)
    
    # Customer Input Section - Full Width
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("### 📋 Customer Information")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("<p class='input-label'>Age</p>", unsafe_allow_html=True)
        age = st.number_input("Age", 18, 100, 35, label_visibility="collapsed")
    
    with col_b:
        st.markdown("<p class='input-label'>Balance ($)</p>", unsafe_allow_html=True)
        balance = st.number_input("Balance ($)", 0, 250000, 5000, label_visibility="collapsed")
    
    with col_c:
        st.markdown("<p class='input-label'>Call Duration (s)</p>", unsafe_allow_html=True)
        duration = st.number_input("Call Duration (s)", 0, 5000, 300, label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("<p class='input-label'>📝 Customer Notes (optional)</p>", unsafe_allow_html=True)
    customer_text = st.text_area("Customer Notes", placeholder="Enter customer feedback or notes here...", height=80, label_visibility="collapsed")
    
    st.markdown("<p class='input-label'>📎 Upload Document (optional)</p>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload Document", type=["jpg", "png"], label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action Button
    predict_btn = st.button("🚀 Analyze Customer", use_container_width=True, key="analyze_btn")
    
    # Results Section - Full Width
    if predict_btn:
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.markdown("### ✨ Analysis Results")
        
        # ML prediction
        input_df = pd.DataFrame(columns=X_cols)
        input_df.loc[0] = 0
        input_df.at[0, "age"] = age
        input_df.at[0, "balance"] = balance
        input_df.at[0, "duration"] = duration
        
        ml_features = scaler.transform(input_df).flatten()
        
        # Text analysis
        text_features = None
        if customer_text and text_analyzer:
            try:
                text_result = text_analyzer.analyze_customer_profile(customer_text)
                text_features = text_result['text_features']
            except:
                pass
        
        # Image analysis
        image_features = None
        if uploaded_image and image_processor:
            try:
                with open(f"/tmp/{uploaded_image.name}", "wb") as f:
                    f.write(uploaded_image.getbuffer())
                img_result = image_processor.analyze_document(f"/tmp/{uploaded_image.name}")
                if img_result['success']:
                    image_features = img_result['image_features']
            except:
                pass
        
        # Ensemble prediction
        if ensemble:
            ensemble.set_weights(ml_wt, txt_wt, img_wt)
            pred = ensemble.predict(ml_features, text_features, image_features)
        else:
            ml_pred = safe_predict_proba(ml_model, ml_features.reshape(1, -1))[0][1]
            pred = {
                'final_prediction': ml_pred,
                'confidence': 0.5,
                'predicted_class': 1 if ml_pred > 0.5 else 0,
                'component_predictions': {'ml_model': ml_pred, 'text_analysis': 0.5, 'image_analysis': 0.5}
            }
        
        st.success("✅ Analysis Complete!")
        st.markdown("---")
        
        # Main Prediction - Enhanced Display
        st.markdown("<div class='prediction-heading'>🎯 Prediction Result</div>", unsafe_allow_html=True)
        
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        with col_pred1:
            pred_text = "🟢 WILL DEPOSIT" if pred['predicted_class'] == 1 else "🔴 WON'T DEPOSIT"
            st.markdown(f"<div class='metric-card'><p style='font-size:0.9rem; color:#a5f3fc;'>PREDICTION</p><p style='font-size:1.8rem; font-weight:900; color:#22c55e;'>{pred_text}</p></div>", unsafe_allow_html=True)
        
        with col_pred2:
            conf_color = "#22c55e" if pred['confidence'] > 0.7 else "#f59e0b" if pred['confidence'] > 0.5 else "#ef4444"
            st.markdown(f"<div class='metric-card'><p style='font-size:0.9rem; color:#a5f3fc;'>CONFIDENCE SCORE</p><p style='font-size:1.8rem; font-weight:900; color:{conf_color};'>{pred['confidence']*100:.1f}%</p></div>", unsafe_allow_html=True)
        
        with col_pred3:
            risk_level = "LOW" if pred['confidence'] > 0.75 else "MEDIUM" if pred['confidence'] > 0.6 else "HIGH"
            risk_color = "#22c55e" if risk_level == "LOW" else "#f59e0b" if risk_level == "MEDIUM" else "#ef4444"
            st.markdown(f"<div class='metric-card'><p style='font-size:0.9rem; color:#a5f3fc;'>RISK LEVEL</p><p style='font-size:1.8rem; font-weight:900; color:{risk_color};'>{risk_level}</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendation Box - Enhanced
        if pred['confidence'] > threshold:
            if pred['predicted_class'] == 1:
                st.markdown(f"<div style='background: rgba(34, 197, 94, 0.2); border: 2px solid #22c55e; border-radius: 12px; padding: 15px; margin: 15px 0;'><p style='font-size: 1.2rem; font-weight: 800; color: #22c55e; margin: 0;'>✨ HIGH VALUE CUSTOMER</p><p style='color: #e0f2fe; margin: 5px 0 0 0;'>Strong confidence level at {pred['confidence']*100:.1f}%. Recommend priority engagement.</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background: rgba(245, 158, 11, 0.2); border: 2px solid #f59e0b; border-radius: 12px; padding: 15px; margin: 15px 0;'><p style='font-size: 1.2rem; font-weight: 800; color: #f59e0b; margin: 0;'>📋 NEEDS FOLLOW-UP</p><p style='color: #e0f2fe; margin: 5px 0 0 0;'>Moderate confidence at {pred['confidence']*100:.1f}%. Recommend additional engagement.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; border-radius: 12px; padding: 15px; margin: 15px 0;'><p style='font-size: 1.2rem; font-weight: 800; color: #ef4444; margin: 0;'>❓ UNCERTAIN PREDICTION</p><p style='color: #e0f2fe; margin: 5px 0 0 0;'>Below threshold ({threshold*100:.0f}%). Recommend collecting more data before action.</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Component Breakdown - Enhanced
        st.markdown("<div class='section-heading'>📊 Component Analysis Breakdown</div>", unsafe_allow_html=True)
        comps = pred['component_predictions']
        
        # Create a more detailed component display
        comp1, comp2, comp3 = st.columns(3)
        
        with comp1:
            ml_score = comps['ml_model']
            ml_bar = "▄" * int(ml_score * 20)
            st.markdown(f"""
            <div class='info-box'>
            <p style='font-size:1.1rem; font-weight:700; color:#a5f3fc; margin:0 0 8px 0;'>🤖 ML Model</p>
            <p style='font-size:1.5rem; font-weight:900; color:#22c55e; margin:0 0 8px 0;'>{ml_score:.1%}</p>
            <p style='font-size:0.8rem; color:#cbd5e1; margin:0;'>Traditional Logistic Regression</p>
            </div>
            """, unsafe_allow_html=True)
        
        with comp2:
            text_score = comps['text_analysis']
            text_bar = "▄" * int(text_score * 20)
            st.markdown(f"""
            <div class='info-box'>
            <p style='font-size:1.1rem; font-weight:700; color:#a5f3fc; margin:0 0 8px 0;'>🔤 Text Analysis</p>
            <p style='font-size:1.5rem; font-weight:900; color:#06b6d4; margin:0 0 8px 0;'>{text_score:.1%}</p>
            <p style='font-size:0.8rem; color:#cbd5e1; margin:0;'>NLP Sentiment & Intent</p>
            </div>
            """, unsafe_allow_html=True)
        
        with comp3:
            img_score = comps['image_analysis']
            img_bar = "▄" * int(img_score * 20)
            st.markdown(f"""
            <div class='info-box'>
            <p style='font-size:1.1rem; font-weight:700; color:#a5f3fc; margin:0 0 8px 0;'>🖼️ Image Analysis</p>
            <p style='font-size:1.5rem; font-weight:900; color:#10b981; margin:0 0 8px 0;'>{img_score:.1%}</p>
            <p style='font-size:0.8rem; color:#cbd5e1; margin:0;'>Document & Visual</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Customer Feature Summary - Enhanced
        st.markdown("<div class='section-heading'>👤 Customer Profile Features</div>", unsafe_allow_html=True)
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            age_risk = "Low" if 25 <= age <= 65 else "High"
            st.markdown(f"""
            <div class='info-box'>
            <p style='font-size:0.9rem; color:#a5f3fc; font-weight:700;'>AGE</p>
            <p style='font-size:1.4rem; font-weight:900; color:#60a5fa;'>{age}</p>
            <p style='font-size:0.8rem; color:#cbd5e1;'>years old</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            balance_level = "High" if balance > 50000 else "Medium" if balance > 5000 else "Low"
            st.markdown(f"""
            <div class='info-box'>
            <p style='font-size:0.9rem; color:#a5f3fc; font-weight:700;'>BALANCE</p>
            <p style='font-size:1.4rem; font-weight:900; color:#10b981;'>${balance:,.0f}</p>
            <p style='font-size:0.8rem; color:#cbd5e1;'>{balance_level} Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            duration_level = "Long" if duration > 600 else "Medium" if duration > 180 else "Short"
            st.markdown(f"""
            <div class='info-box'>
            <p style='font-size:0.9rem; color:#a5f3fc; font-weight:700;'>CALL DURATION</p>
            <p style='font-size:1.4rem; font-weight:900; color:#f59e0b;'>{duration}s</p>
            <p style='font-size:0.8rem; color:#cbd5e1;'>{duration_level} Engagement</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Store prediction
        st.session_state.predictions.append({
            "age": age,
            "balance": balance,
            "duration": duration,
            "pred": pred['predicted_class'],
            "conf": pred['confidence']
        })
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Enter customer details above and click **Analyze Customer** to generate predictions")

# TAB 2: Dashboard
with tab2:
    st.markdown("<div class='tab-title'>📊 Dashboard</div>", unsafe_allow_html=True)
    
    if len(st.session_state.predictions) > 0:
        df_pred = pd.DataFrame(st.session_state.predictions)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df_pred))
        col2.metric("Avg Confidence", f"{df_pred['conf'].mean():.1%}")
        col3.metric("Deposit %", f"{df_pred['pred'].sum()/len(df_pred)*100:.0f}%")
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#0f172a')
            will = (df_pred['pred'] == 1).sum()
            wont = (df_pred['pred'] == 0).sum()
            ax.pie([will, wont], labels=['Will', 'Won\'t'], colors=['#22c55e', '#ef4444'], autopct='%1.1f%%', textprops={'color': 'white'})
            ax.set_title("Predictions", color='white')
            st.pyplot(fig)
        
        with col_b:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor='#0f172a')
            ax.scatter(df_pred['age'], df_pred['balance'], c=df_pred['conf'], cmap='RdYlGn', s=100)
            ax.set_xlabel('Age', color='white')
            ax.set_ylabel('Balance', color='white')
            ax.set_title("Age vs Balance", color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)
        
        st.dataframe(df_pred)
    else:
        st.info("No predictions yet. Make some in the Ensemble tab!")

# TAB 3: Chatbot
with tab3:
    st.markdown("<div class='tab-title'>🤖 Banking Bot</div>", unsafe_allow_html=True)
    
    if "chat" not in st.session_state:
        st.session_state.chat = []
    
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    
    # Chat header with Clear History button
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("<p class='section-heading'>💬 Chat Interface</p>", unsafe_allow_html=True)
    with header_col2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_chat_btn"):
            st.session_state.chat = []
            st.session_state.chat_input = ""
            st.rerun()
    
    st.markdown("---")
    
    # Input section with columns for better layout
    col_input, col_btn = st.columns([5, 1])
    
    with col_input:
        st.markdown("<p class='input-label'>💬 Ask about our banking products</p>", unsafe_allow_html=True)
        query = st.text_input("Question", placeholder="e.g., 'Tell me about deposits' or 'What loans do you offer?'", label_visibility="collapsed", key="chat_query_input")
    
    with col_btn:
        st.markdown("<p style='margin-top: 28px;'></p>", unsafe_allow_html=True)
        send_clicked = st.button("📤 Send", use_container_width=True, key="chat_send_btn")
    
    if send_clicked and query:
        # Determine bot response
        if any(w in query.lower() for w in ["deposit", "save", "fd", "fixed"]):
            resp = "💰 Fixed Deposits: Earn 5-7% annual interest with flexible tenure from 1-5 years!"
        elif any(w in query.lower() for w in ["loan", "borrow", "credit"]):
            resp = "🏦 Loans Available:\n• Personal Loans: 8-12% interest\n• Home Loans: 6-8% interest\n• Business Loans: 10-15% interest"
        elif any(w in query.lower() for w in ["account", "open", "create"]):
            resp = "📱 Account Opening is Simple:\n1. Provide ID verification\n2. Address proof\n3. Takes just 5 minutes online!\n4. Instant card activation"
        elif any(w in query.lower() for w in ["fee", "charge", "cost"]):
            resp = "💳 Charges:\n• No account maintenance fee\n• ATM withdrawals: Free (1000+ ATMs)\n• Fund transfer: Free for up to 10 per month\n• Debit card: Free"
        elif any(w in query.lower() for w in ["rate", "interest", "%"]):
            resp = "📊 Current Rates:\n• Savings Account: 3-4%\n• Current Account: No interest\n• Money Market: 5-6%\n• Certificates of Deposit: 5-7%"
        else:
            resp = "📞 Great question! I can help with deposits, loans, accounts, fees, and rates. Or contact our support team at support@bank.com for more details!"
        
        # Add to chat history
        st.session_state.chat.append(("You", query))
        st.session_state.chat.append(("Bot", resp))
        
        # Rerun to update the UI and clear text
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    st.markdown("<div style='max-height: 400px; overflow-y: auto; padding: 10px; border-radius: 10px; background: rgba(15, 23, 42, 0.6);'>", unsafe_allow_html=True)
    
    if len(st.session_state.chat) == 0:
        st.info("👋 Start a conversation! Ask me about deposits, loans, accounts, fees, or interest rates.")
    else:
        for sender, msg in st.session_state.chat:
            if sender == "Bot":
                st.markdown(f"<div style='background: rgba(14,165,233,0.25); padding:12px; border-radius:8px; margin:8px 0; border-left: 4px solid #0ea5e9;'><span style='font-weight:900; color:#60a5fa;'>🤖 Banking Bot:</span><p style='color:#e0f2fe; margin:5px 0 0 0; white-space: pre-wrap;'>{msg}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background: rgba(34,197,94,0.25); padding:12px; border-radius:8px; margin:8px 0; border-left: 4px solid #22c55e;'><span style='font-weight:900; color:#86efac;'>👤 You:</span><p style='color:#e0f2fe; margin:5px 0 0 0; white-space: pre-wrap;'>{msg}</p></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 4: Info
with tab4:
    st.markdown("<div class='tab-title'>ℹ️ About</div>", unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 Advanced LLM Ensemble System
    
    **Components:**
    - 🤖 **ML**: Logistic Regression on customer features
    - 🔤 **LLM**: Hugging Face Transformers for sentiment/intent
    - 📸 **Vision**: OpenCV + Pytesseract for document OCR
    
    **Features:**
    - ✅ Multimodal analysis (text + image + data)
    - ✅ Real-time predictions with confidence scores
    - ✅ Interactive dashboard with analytics
    - ✅ Smart chatbot for banking queries
    - ✅ Customizable ensemble weights
    
    **Status:** Production Ready ✨
    """)
