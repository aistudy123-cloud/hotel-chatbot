import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os, base64, time, random

st.set_page_config(page_title="KI-Chatbot", page_icon="💬")

# ---- Logo laden & als Base64 einbetten (vermeidet Pfad-/Serving-Probleme) ----
LOGO_PATH = "/mnt/data/AI-Chatbot.png"
def to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""
LOGO_B64 = to_b64(LOGO_PATH)

# ---- Kopfbereich mit Titel + Reset-Button rechts oben ----
header_col_l, header_col_c, header_col_r = st.columns([1, 3, 1])
with header_col_c:
    st.markdown(
        "<div style='text-align:center; margin-bottom:0.5rem;'>"
        "<h1 style='margin-bottom:0.2rem;'>🏨 Hotel Bellevue Grand</h1>"
        "<p style='margin-top:0; color:#666;'>Schnelle Hilfe beim Check-in, Zimmer & mehr mit unserem KI-Chatbot</p>"
        "</div>",
        unsafe_allow_html=True
    )
with header_col_r:
    if st.button("🧹 Neue Unterhaltung starten", key="btn_reset_top"):
        st.session_state.history = []
        st.rerun()

# ---- Fixiertes Seiten-Panel rechts (HTML ohne Einrückungen, damit kein Codeblock entsteht) ----
img_tag = f"<img src='data:image/png;base64,{LOGO_B64}' alt='Chatbot Logo'>" if LOGO_B64 else ""
st.markdown(
"""<style>
.fixed-sidebox{
position:fixed;
top:160px;
right:24px;
width:230px;
background:#ffffff;
border:1px solid #e8e8e8;
border-radius:16px;
box-shadow:0 8px 20px rgba(0,0,0,0.08);
text-align:center;
padding:16px 14px 18px;
z-index:1000;
}
.fixed-sidebox h3{
margin:10px 0 0;
color:#222;
font-size:24px;
line-height:1.25;
}
.fixed-sidebox img{
display:block;
margin:2px auto 6px;
max-width:90%;
height:auto;
border-radius:12px;
}
@media (max-width:900px){
.fixed-sidebox{display:none;}
}
</style>
<div class="fixed-sidebox">
""" + img_tag + """
<h3>KI-Chatbot</h3>
</div>
""",
    unsafe_allow_html=True
)

# ---- Daten laden & Helfer ----
@st.cache_resource
def load_kb(csv_path="answers.csv"):
    df = pd.read_csv(csv_path).fillna("")
    if not {"id","question","answer"}.issubset(df.columns):
        raise ValueError("CSV braucht Spalten: id, question, answer")
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words=None, lowercase=True)
    X = vec.fit_transform(df["question"].tolist())
    return df, vec, X

def find_best_answer(user_text, df, vec, X, threshold=0.25, topk=3):
    q = vec.transform([user_text])
    sims = cosine_similarity(q, X).flatten()
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    if best_sim < threshold:
        return None, best_sim, []
    top_idx = sims.argsort()[::-1][:topk]
    top = [(df.iloc[i]["id"], df.iloc[i]["question"], float(sims[i])) for i in top_idx]
    return df.iloc[best_idx], best_sim, top

def log_event(user_text, picked_id, sim, logfile="logs.csv"):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_text": user_text,
        "picked_id": picked_id,
        "similarity": sim,
    }
    exists = os.path.exists(logfile)
    pd.DataFrame([row]).to_csv(logfile, mode="a", index=False, header=not exists)

# ---- Hauptlogik ----
df, vec, X = load_kb("answers.csv")
if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    with st.chat_message("assistant"):
        st.write("Willkommen im Hotel! Wie kann ich helfen?")

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

user_msg = st.chat_input("Frag mich etwas …")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)

    best, sim, top = find_best_answer(user_msg, df, vec, X, threshold=0.20, topk=3)
    if best is None:
        bot_text = "Dazu kann ich dir leider nicht weiterhelfen."
        picked_id = ""
    else:
        bot_text = best["answer"]
        picked_id = best["id"]

    with st.chat_message("assistant"):
        dots = st.empty()
        for i in range(3):
            dots.markdown(f"_schreibt{'.' * ((i % 3) + 1)}_")
            time.sleep(0.35)
        dots.empty()

        output = st.empty()
        displayed = ""
        for ch in bot_text:
            displayed += ch
            output.markdown(displayed)
            time.sleep(random.uniform(0.01, 0.03))

    st.session_state.history.append(("assistant", bot_text))
    log_event(user_msg, picked_id, sim if best is not None else 0.0)
