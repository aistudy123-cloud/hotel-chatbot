import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import time, random

st.set_page_config(page_title="KI-Chatbot", page_icon="💬")

# ====== THEME / STYLES (inkl. Fix gegen weiße Balken) ======
st.markdown("""
<style>
/* --- Globale Container-Abstände minimieren --- */
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
div[data-testid="stVerticalBlock"]         { gap: 0 !important; }
div[data-testid="stVerticalBlock"] > div   { margin: 0 !important; padding: 0 !important; }
div[data-testid="stHorizontalBlock"]       { gap: 0 !important; }
hr, div[data-testid="stDivider"]           { display: none !important; }

/* --- App-Hintergrund & zentrierte Chatkarte --- */
[data-testid="stAppViewContainer"] { background: #f4f6fb; }
.chat-shell {
  max-width: 840px; margin: 1.2rem auto;
  border-radius: 16px; box-shadow: 0 6px 18px rgba(0,0,0,.08);
  overflow: hidden; background: #fff; border: 1px solid #e9eef5;
}

/* --- Header-Leiste (Hotel-Stil) --- */
.chat-header {
  display: flex; align-items: center; gap: 12px;
  background: #003a6d; color: #fff; padding: 12px 16px;
}
.chat-header .title   { font-weight: 600; letter-spacing: .2px; }
.chat-header .subtitle{ opacity: .85; font-size: .9rem; }

/* --- Chat-Content --- */
.chat-body {
  padding: 12px 16px;
  background: linear-gradient(180deg,#ffffff 0%,#ffffff 65%, #fafbfe 100%);
}

/* --- Reihen & Bubbles --- */
.chat-row { display: flex; margin: 4px 0 !important; }  /* <- kleine, feste Abstände */
.chat-row.bot  { justify-content: flex-start; }
.chat-row.user { justify-content: flex-end; }

.bubble {
  max-width: 78%; padding: 10px 14px; border-radius: 16px;
  word-wrap: break-word; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  position: relative;
}
.bubble.bot {
  background: #fff7e6; border: 1px solid #f3e2be; border-bottom-left-radius: 6px;
}
.bubble.user {
  background: #e8f0fe; border: 1px solid #cfdaf7; border-bottom-right-radius: 6px;
}

/* --- Avatare --- */
.avatar {
  width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
  border-radius: 50%; box-shadow: 0 1px 3px rgba(0,0,0,.15); font-size: 18px;
}
.avatar.bot  { background: #ffecb3; color: #5a4800; margin-right: 8px; }
.avatar.user { background: #dbe6ff; color: #0b3b8c; margin-left: 8px; }

/* --- Footer/Chat-Input optisch --- */
.chat-footer { padding: 6px 16px 4px 16px; border-top: 1px solid #eef2f7; background: #fff; }
div[data-testid="stChatInputContainer"] textarea {
  border: 1px solid #cdd7e3 !important; border-radius: 12px !important;
  transition: box-shadow .15s ease, border-color .15s ease;
}
div[data-testid="stChatInputContainer"] textarea:focus {
  border-color: #4179c8 !important; box-shadow: 0 0 0 3px rgba(65,121,200,.18) !important;
}
</style>
""", unsafe_allow_html=True)

# ====== (Optional) oberes Intro beibehalten ======
st.markdown(
    """
    <div style="text-align:center; margin-bottom:0.5rem;">
      <h1 style="margin-bottom:0.2rem;">🏨 Hotel Bellevue Grand</h1>
      <p style="margin-top:0; color:#666;">Schnelle Hilfe beim Check-in, Zimmer & mehr mit unserem KI-Chatbot</p>
    </div>
    """,
    unsafe_allow_html=True
)

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

# ====== SHELL ======
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

# Header-Leiste
st.markdown("""
<div class="chat-header">
  <div class="avatar bot">🏨</div>
  <div>
    <div class="title">Concierge · Bellevue Grand</div>
    <div class="subtitle">Online · antwortet in wenigen Sekunden</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Chat-Body
st.markdown('<div class="chat-body">', unsafe_allow_html=True)

df, vec, X = load_kb("answers.csv")
if "history" not in st.session_state:
    st.session_state.history = []

# Reset (bleibt oben in der Karte – ohne weißen Balken)
if st.button("🧹 Neue Unterhaltung starten"):
    st.session_state.history = []
    st.rerun()

# Helper: Bubble-Renderer
def render_bubble(role, text):
    safe_text = text  # ggf. markdown.escape nutzen, wenn du HTML im Text verhindern willst
    if role == "user":
        st.markdown(f"""
        <div class="chat-row user">
          <div class="bubble user">{safe_text}</div>
          <div class="avatar user">🧑</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row bot">
          <div class="avatar bot">🏨</div>
          <div class="bubble bot">{safe_text}</div>
        </div>
        """, unsafe_allow_html=True)

# Welcome
if not st.session_state.history:
    render_bubble("assistant", "Willkommen im Hotel! Wie kann ich helfen?")

# Historie
for role, text in st.session_state.history:
    render_bubble(role, text)

# Body zu – Footer optisch vor dem Input
st.markdown('</div>', unsafe_allow_html=True)  # /chat-body
st.markdown('<div class="chat-footer"></div>', unsafe_allow_html=True)

# Chat-Input
user_msg = st.chat_input("Frag mich etwas …")
if user_msg:
    # User anzeigen + speichern
    st.session_state.history.append(("user", user_msg))
    render_bubble("user", user_msg)

    # Antwort bestimmen
    best, sim, top = find_best_answer(user_msg, df, vec, X, threshold=0.25, topk=3)
    if best is None:
        bot_text = "Da bin ich unsicher. Meinst du Check-in, Parkplatz oder WLAN?"
        picked_id = ""
    else:
        bot_text = best["answer"]; picked_id = best["id"]

    # Tipp-Animation (Bubble-Placeholder)
    ph = st.empty()
    ph.markdown("""
    <div class="chat-row bot">
      <div class="avatar bot">🏨</div>
      <div class="bubble bot"><em>schreibt…</em></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(min(1.2, max(0.4, len(bot_text) * 0.01)))
    ph.empty()

    # Typing-Effekt
    container = st.empty()
    displayed = ""
    for ch in bot_text:
        displayed += ch
        container.markdown(f"""
        <div class="chat-row bot">
          <div class="avatar bot">🏨</div>
          <div class="bubble bot">{displayed}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(random.uniform(0.005, 0.02))

    # Speichern & Loggen
    st.session_state.history.append(("assistant", bot_text))
    log_event(user_msg, picked_id, sim if best is not None else 0.0)

st.markdown('</div>', unsafe_allow_html=True)  # /chat-shell
