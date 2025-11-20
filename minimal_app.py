import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os, base64, time, random

st.set_page_config(page_title="KI-Chatbot", page_icon="💬", layout="wide")

# ---- Logo laden & als Base64 einbetten (vermeidet Pfad-/Serving-Probleme) ----
LOGO_PATH = "AI-Chatbot.png"
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
with header_col_l:
    if st.button("🧹 Unterhaltung neu starten", key="btn_reset_top"):
        st.session_state.history = []
        st.rerun()

# ---- Fixiertes Seiten-Panel rechts (HTML ohne Einrückungen, damit kein Codeblock entsteht) ----
img_tag = f"<img src='data:image/png;base64,{LOGO_B64}' alt='Chatbot Logo'>" if LOGO_B64 else ""
st.markdown("""
<style>
/* ========== Design Preset: Hotel Bellevue Grand ========== */

/* — Farben & Typo — */
:root{
  --brand:#0E2A47;     /* Dunkelblau */
  --brand-2:#3B6EA8;   /* Akzent */
  --bg:#F6F7F9;        /* App-Hintergrund */
  --card:#FFFFFF;      /* Karten / Bubbles */
  --muted:#6B7280;     /* Sekundärtext */
  --border:#E6E8EC;    /* Ränder */
  --shadow:0 8px 24px rgba(15,23,42,0.08);
}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] *{
  font-family:'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

/* — App-Layout: dezenter Hintergrund & Content-Breite — */
[data-testid="stAppViewContainer"]{
  background:var(--bg);
}
main [data-testid="block-container"]{
  max-width: 980px;
  padding-top: 0.5rem;
  padding-bottom: 2rem;
  /* Platz rechts, damit nichts unter das fixe Sidepanel läuft */
  padding-right: 290px;
}

/* — Überschrift-Zeile (dezent) — */
h1{
  color: var(--brand);
  letter-spacing: .2px;
}
p, li { color:#111; }

/* — Buttons allgemein — */
.stButton button{
  border-radius: 12px;
  background: var(--brand);
  color:#fff;
  border:1px solid var(--brand);
  padding:.55rem .9rem;
  box-shadow: var(--shadow);
  font-weight:600;
}
.stButton button:hover{ filter: brightness(1.05); }
.stButton button:active{ transform: translateY(1px); }

/* — Eingabefeld (Chat) — */
[data-testid="stChatInput"] textarea{
  border-radius:14px !important;
  border:1px solid var(--border) !important;
  background:#fff !important;
  box-shadow: var(--shadow);
}

/* — Chatblasen — */
[data-testid="stChatMessage"]{
  background:transparent;
  padding:0;
  margin: 0 0 .4rem 0;
}
[data-testid="stChatMessage"] > div{ /* Bubble-Container */
  background: var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding: .75rem .9rem;
  box-shadow: var(--shadow);
}

/* User rechts ausrichten, Assistant links */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div{
  background: #FAFCFF;
  border-color:#DDE7F5;
  margin-left: 80px;   /* Platz für Avatar links */
  margin-right: 0;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div{
  background: #FFFFFF;
  margin-right: 80px;  /* Platz für Avatar rechts */
}

/* Avatare dezenter */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{
  filter: saturate(.9);
}

/* Markdown im Chat kompakter & ohne große Lücken */
[data-testid="stChatMessage"] p{
  margin: .2rem 0 .1rem 0;
  line-height: 1.45;
}
[data-testid="stChatMessage"] ul, 
[data-testid="stChatMessage"] ol{
  margin: .2rem 0 .2rem 1.1rem;
}

/* „schreibt…“ Platzhalter etwas feiner */
[data-testid="stMarkdownContainer"] em{
  color: var(--muted);
}

/* — Scrollbar dezent — */
::-webkit-scrollbar{ width:10px; }
::-webkit-scrollbar-thumb{
  background:#C9D4E3; border-radius:8px; border:2px solid transparent; background-clip: padding-box;
}
::-webkit-scrollbar-track{ background:transparent; }

/* — Fixiertes Seiten-Panel rechts — */
.fixed-sidebox{
  position:fixed;
  top:140px;                 /* etwas höher für mehr Balance */
  right:24px;
  width:230px;
  background:#ffffff;
  border:1px solid var(--border);
  border-radius:16px;
  box-shadow: var(--shadow);
  text-align:center;
  padding:16px 14px 18px;
  z-index:1000;
}
.fixed-sidebox h3{
  margin:10px 0 0;
  color:var(--brand);
  font-weight:700;
  font-size:22px;
  line-height:1.2;
}
.fixed-sidebox img{
  display:block;
  margin:2px auto 6px;
  max-width:65%;             /* <- kleineres Bild */
  height:auto;
  border-radius:12px;
}
@media (max-width: 1100px){
  main [data-testid="block-container"]{ padding-right: 0; }
}
@media (max-width:900px){
  .fixed-sidebox{ display:none; }
}

/* — Karten/Container, falls irgendwo st.write(...) Karten rendert — */
.block-container .stMarkdown, .block-container .stText{
  color:#111;
}

/* — Feine Kanten für Tabellen (falls du welche zeigst) — */
table{
  border-collapse:separate !important;
  border-spacing:0 !important;
  overflow:hidden;
  border:1px solid var(--border);
  border-radius:12px;
}
thead tr th{ background:#F3F6FA !important; color:#0F172A; }
tbody tr + tr td{ border-top:1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

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
