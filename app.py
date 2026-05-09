import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Banking System", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 45%, #0d9488 100%);
    color: #f8fafc;
}

.stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #e0f2fe;
}

label, .stText, .stMarkdown, .stCaption, p, span, div, strong, em, small, li, a, h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}

.stApp, [data-testid="stAppViewContainer"] {
    color: #f8fafc !important;
}

/* input controls must remain readable with dark text */
input, textarea, select, option, .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
    color: #0f172a !important;
    background: #fdfdfd !important;
    border-color: #ddd !important;
}

/* specific labels for sliders and inputs */
.stSlider label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stFileUploader label,
.stSlider .st-bf {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

/* ensure text in input boxes is visible on white bg */
.stNumberInput input, .stTextInput textarea, .stSlider input {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* keep very clear icon text and blank placeholder */
.stNumberInput div, .stTextInput div, .stSlider div {
    color: #0f172a !important;
}


/* explicit on-upload instructions and file browser text */
[data-testid="stFileUploader"] * {
    color: #0f172a !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #e0f2fe !important;
}

h1 { font-size: 3rem !important; }
h2 { font-size: 2.4rem !important; }
h3 { font-size: 2rem !important; }

.section-card {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 12px 26px rgba(8, 8, 24, 0.35);
}

.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #60a5fa;
    margin-bottom: 14px;
}

.tab-title {
    font-size: 2.6rem;
    font-weight: 900;
    color: #f8fafc;
    text-align: center;
    margin-bottom: 16px;
    text-shadow: 0 0 12px rgba(0,0,0,0.9);
}

.stButton>button {
    background: linear-gradient(90deg, #0ea5e9, #10b981);
    color: white;
    border: 0;
    border-radius: 12px;
    padding: 0.75em 1em;
}

.stSlider>div>div>div>div>input,
.stNumberInput>div>input,
textarea,
input[type="text"] {
    color: #0f172a;
    background: #ffffff;
    font-weight: 700;
}

.stButton>button:hover {
    box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.5);
}

.sidebar .stMarkdown h1, .sidebar .stMarkdown h2, .sidebar .stMarkdown h3 { color: #0f172a !important; }
.sidebar .stMarkdown, .sidebar .stSelectbox, .sidebar .stText, .sidebar .stNumberInput, .sidebar .stSlider {
    color: #0f172a !important;
}
.sidebar .css-1jdc9rk { background: #f8fafc !important; }
.sidebar .css-1avcm0n { color: #0f172a !important; }

@media (max-width: 900px) {
    .section-card { padding: 16px; }
}

/* sidebar visibility (override default white-on-white text) */
[data-testid="stSidebar"] * {
    color: #0f172a !important;
}
[data-testid="stSidebar"] {
    background: #f8fafc !important;
}

/* tab navigation style */
button[role="tab"] {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em !important;
    padding: 14px 20px !important;
    color: #e2e8f0 !important;
}

button[role="tab"][aria-selected="true"] {
    color: #a5f3fc !important;
    border-bottom: 4px solid #38bdf8 !important;
}

button[role="tab"]:hover {
    color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 AI Multimodal Banking System")
st.markdown("""
<div style='text-align:center; color:#bfdbfe; margin-top:-10px; margin-bottom:14px;'>
🎉 Welcome to your vibrant, interactive customer intelligence dashboard — all actions are designed carefully. 🎉
</div>
""", unsafe_allow_html=True)

# Sidebar interaction panel
st.sidebar.markdown("""
<div style='background:#ffffff; padding:12px; border-radius:12px; color:#0f172a;'>
<h2 style='margin-top:0; margin-bottom:8px;'>⚙️ Quick Controls</h2>
<ul style='color:#0f172a; margin-top:0; margin-left:16px; line-height:1.6;'>
<li><strong>Step 1:</strong> Fill customer input</li>
<li><strong>Step 2:</strong> Add a short message (keywords: loan, deposit, save)</li>
<li><strong>Step 3:</strong> Optional image upload</li>
<li><strong>Step 4:</strong> Run analyze and check Dashboard</li>
</ul>
<p style='font-size:0.95rem; margin-bottom:0;'>Use the tabs for streamlined interaction.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
active_scoring = st.sidebar.selectbox("Customer confidence threshold", [0.6, 0.7, 0.8, 0.9], index=1)
st.sidebar.info(f"High potential if score > {active_scoring}")

if st.sidebar.button("📄 Reset Session Data"):
    st.session_state.data = []
    st.session_state.chat = []
    st.sidebar.success("Session data reset. Reload page to restart UI state.")


# ---------------- LOAD ----------------
import os
from pathlib import Path

# Get the directory of the current script
app_dir = Path(__file__).parent
project_root = app_dir.parent

# Load models and data using absolute paths
model = joblib.load(project_root / "models" / "customer_model.pkl")
scaler = joblib.load(project_root / "models" / "scaler.pkl")

# Compatibility: older scikit-learn model persistence may not include multi_class attr
if not hasattr(model, "multi_class"):
    model.multi_class = "ovr"


def safe_predict_proba(model, X):
    if not hasattr(model, "multi_class"):
        model.multi_class = "ovr"

    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if df.ndim == 1:
            p = 1 / (1 + np.exp(-df))
            return np.vstack([1 - p, p]).T
        if df.ndim == 2:
            exp = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

    preds = model.predict(X)
    return np.vstack([1 - preds, preds]).T


df = pd.read_csv(project_root / "data" / "bank.csv")
df["deposit"] = df["deposit"].map({"yes":1,"no":0})
df = pd.get_dummies(df, drop_first=True)
X_columns = df.drop("deposit", axis=1).columns

# SESSION
if "data" not in st.session_state:
    st.session_state.data = []

if "chat" not in st.session_state:
    st.session_state.chat = []

# TABS
tab1, tab2, tab3 = st.tabs(["🏦 Customer Analysis", "📊 Dashboard", "🤖 Chatbot"])

# ================= TAB 1 =================
with tab1:

    st.markdown("<div class='tab-title'>🏦 Customer Analysis</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # -------- INPUT --------
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Step 1: Customer Input (use inputs carefully)</div>", unsafe_allow_html=True)

        st.markdown("<span style='color:#f8fafc; font-weight:700; font-size:1.1rem'>Age</span>", unsafe_allow_html=True)
        age = st.number_input("", 18, 100, 30, help="Estimate customer age", format="%d", label_visibility='collapsed')

        st.markdown("<span style='color:#f8fafc; font-weight:700; font-size:1.1rem'>Balance</span>", unsafe_allow_html=True)
        balance = st.number_input("", 0, 100000, 1000, help="Customer account balance", format="%d", label_visibility='collapsed')

        st.markdown("<span style='color:#f8fafc; font-weight:700; font-size:1.1rem'>Call Duration</span>", unsafe_allow_html=True)
        duration = st.number_input("", 0, 5000, 100, help="Duration of recent call (seconds)", format="%d", label_visibility='collapsed')

        st.subheader("💬 Step 2: Customer Message")
        text = st.text_area("Enter message", placeholder="e.g. I want to invest in a fixed deposit")

        st.subheader("🖼 Step 3: Upload Image")
        image = st.file_uploader("Choose image", type=["jpg","png"], help="Optional context image for multimodal signal")

        run = st.button("🚀 Analyze Customer")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- RESULTS --------
    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📈 Step 4: AI Prediction Output</div>", unsafe_allow_html=True)

        if run:
            input_df = pd.DataFrame(columns=X_columns)
            input_df.loc[0] = 0
            input_df.at[0,"age"] = age
            input_df.at[0,"balance"] = balance
            input_df.at[0,"duration"] = duration

            prob = safe_predict_proba(model, scaler.transform(input_df))[0][1]

            keywords = ["invest","deposit","save","fd"]
            sentiment_score = 1 if any(k in text.lower() for k in keywords) else 0
            image_score = 1 if image else 0

            final_score = (prob*0.6)+(sentiment_score*0.3)+(image_score*0.1)

            if final_score > active_scoring:
                st.success("💎 EXCELLENT potential customer")
            elif final_score > 0.6:
                st.info("✅ GOOD potential customer")
            else:
                st.warning("❗ LOW potential customer, handle carefully")

            st.metric("💰 Deposit Probability", f"{prob*100:.2f}%", delta=f"{final_score*100:.2f}%")

            fig, ax = plt.subplots(figsize=(4,3))
            ax.set_facecolor('#f8fafc')
            ax.bar(["ML","NLP","IMG"], [prob, sentiment_score, image_score], color=['#22c55e','#38bdf8','#a78bfa'])
            ax.set_ylim(0,1)
            ax.set_title("Model + Signal Breakdown", color='#fff')
            ax.tick_params(colors='#fff')
            st.pyplot(fig)

            st.session_state.data.append({
                "Age": age,
                "Balance": balance,
                "Duration": duration,
                "Score": final_score
            })
        else:
            st.info("Click the button above to run the model. Every option is validated carefully.")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 2 =================
with tab2:

    st.markdown("<div class='tab-title'>📊 Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Dashboard Overview</div>", unsafe_allow_html=True)

    if len(st.session_state.data) > 0:
        df_dash = pd.DataFrame(st.session_state.data)

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", len(df_dash), delta=f"{len(df_dash)-1} since start")
        c2.metric("Avg Balance", int(df_dash["Balance"].mean()), delta=f"{int(df_dash['Balance'].mean()-1000)}")
        c3.metric("Avg Score", round(df_dash["Score"].mean(),2), delta=f"{round(df_dash['Score'].mean()-0.65,2)}")

        colA, colB = st.columns(2)

        with colA:
            fig1, ax1 = plt.subplots(figsize=(3,3), facecolor='#0f172a')
            high = len(df_dash[df_dash["Score"] > 0.6])
            low = len(df_dash[df_dash["Score"] <= 0.6])
            wedges, texts, autotexts = ax1.pie([high, low], labels=["High","Low"], colors=['#22c55e','#ef4444'], autopct="%1.1f%%", textprops={'color':'white'})
            ax1.set_title("Potential Split", color='white')
            st.pyplot(fig1)

        with colB:
            fig2, ax2 = plt.subplots(figsize=(4,3), facecolor='#0f172a')
            ax2.bar(df_dash["Age"], df_dash["Balance"], color='#38bdf8')
            ax2.set_title("Age vs Balance", color='white')
            ax2.set_xlabel("Age", color='white')
            ax2.set_ylabel("Balance", color='white')
            ax2.tick_params(colors='white')
            st.pyplot(fig2)

    else:
        st.info("Run analysis first")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:

    st.markdown("<div class='tab-title'>🤖 Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Smart Chatbot (answers are curated carefully)</div>", unsafe_allow_html=True)

    user_input = st.text_input("Ask your question", key="chat_input", help="Type keywords like ‘loan’, ‘deposit’, ‘interest’")

    if st.button("Send", key="chat_btn"):
        if user_input:
            msg = user_input.lower()

            if "loan" in msg:
                reply = "We offer personal and home loans with flexible terms."
            elif "deposit" in msg:
                reply = "Fixed deposits are safe investments with higher interest for longer terms."
            elif "interest" in msg:
                reply = "Interest rates are around 5% to 7%, and depend on product type."
            else:
                reply = "Please contact bank support at support@bank.com for details."

            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("Bot", reply))

    if len(st.session_state.chat) == 0:
        st.info("Start a conversation to see intelligent response suggestions")

    for sender, msg in st.session_state.chat:
        if sender == 'Bot':
            st.markdown(f"<div style='background: rgba(15, 23, 42, 0.8); padding:10px; border-radius:10px; color:#f0f9ff; margin-bottom:8px;'><b>{sender}:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: rgba(16, 185, 129, 0.15); padding:10px; border-radius:10px; color:#d9f99d; margin-bottom:8px;'><b>{sender}:</b> {msg}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)