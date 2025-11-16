import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import time, random

# ─────────────────────────────────────────────────────────────────────────────
# Grundeinstellungen
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hotel Bellevue Grand – Digitaler Concierge", page_icon="💬")

# Globales CSS – Weißer Hintergrund, klare Typo, edle Chatkarten
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] * {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
}

/* Weißer Hintergrund */
[data-testid="stAppViewContainer"] {
  background: #FFFFFF;
}

/* Bühne */
section.main > div {
  max-width: 880px !important;
  margin: 0 auto !important;
  padding-top: 1rem !important;
}

/* Überschriften */
h1, h2, h3 { letter-spacing: 0.2px; margin-top: 0.2rem; }

/* Chatkarten */
div[data-testid="stChatMessage"] {
  background: #FFFFFF;
  border: 1px solid rgba(15, 23, 42, 0.06);
  border-radius: 16px;
  padding: 0.75rem 0.9rem;
  box-shadow: 0 4px 16px rgba(2, 6, 23, 0.05);
  margin-bottom: 0.6rem;
}
div[data-testid="stChatMessage"] p { margin: 0.1rem 0; line-height: 1.5; }

/* User vs. Assistant subtil unterscheiden */
div[data-testid="stChatMessage"]:has(img[alt="🧑"]) { background: #F8FAFC; }
div[data-testid="stChatMessage"]:has(img[alt="🏨"]) { background: #FFFFFF; }

/* Chat-Input */
div[data-testid="stChatInput"] textarea {
  border-radius: 12px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: #FFFFFF !important;
}

/* Buttons */
button[kind="primary"] {
  border-radius: 999px !important;
  box-shadow: 0 4px 12px rgba(30,136,229,.15) !important;
}
button:hover { filter: brightness(0.98); }

/* Trennlinie */
hr { border: none; border-top: 1px solid rgba(15,23,42,0.08); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Header / Branding
st.markdown("""
<div style="
    background:#ffffff;
    border:1px solid rgba(15,23,42,0.08);
    border-radius:18px;
    padding:24px;
    margin-bottom:18px;
    box-shadow:0 8px 30px rgba(2,6,23,0.05);
    text-align:center;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png"
       width="70" style="border-radius:50%;margin-bottom:10px;" alt="Hotel Logo">
  <h1 style="margin-bottom:0;">🏨 Hotel Bellevue Grand</h1>
  <p style="margin-top:6px;margin-bottom:8px;color:#475569;font-size:16px;">
    Ihr persönlicher digitaler Concierge – 24 Stunden für Sie da
  </p>
  <div style="margin-top:10px;">
    <span style="display:inline-block;padding:6px 14px;border-radius:999px;
      background:#E8F5E9;color:#2E7D32;font-size:13px;">
      Rezeption heute geöffnet – 07:00–23:00 Uhr
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# Bild zentriert (optional)
center_col = st.columns([2, 2, 2])[1]
with center_col:
    st.image("AI-Chatbot.jpg", width=200)

# ─────────────────────────────────────────────────────────────────────────────
# Daten & Logik
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_kb(csv_path="answers.csv"):
    df = pd.read_csv(csv_path).fillna("")
    if not {"id", "question", "answer"}.issubset(df.columns):
        raise ValueError("CSV braucht Spalten: id, question, answer")
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=None, lowercase=True)
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

# Tippanimation mit Längenbremse
def type_out(text: str):
    n = len(text)
    if n <= 120:
        base_min, base_max, chunk = 0.012, 0.025, 1
    elif n <= 400:
        base_min, base_max, chunk = 0.008, 0.016, 2
    elif n <= 800:
        base_min, base_max, chunk = 0.004, 0.010, 4
    else:
        base_min, base_max, chunk = 0.001, 0.003, 8

    dots = st.empty()
    for i in range(3):
        dots.markdown(f"<span style='opacity:.75;'><em>schreibt{'.' * (i % 3 + 1)}</em></span>", unsafe_allow_html=True)
        time.sleep(0.3)
    dots.empty()

    output = st.empty()
    shown = ""
    i = 0
    while i < n:
        shown += text[i:i+chunk]
        output.markdown(shown)
        time.sleep(random.uniform(base_min, base_max))
        i += chunk

# ─────────────────────────────────────────────────────────────────────────────
# Chatlogik
# ─────────────────────────────────────────────────────────────────────────────
df, vec, X = load_kb("answers.csv")
if "history" not in st.session_state:
    st.session_state.history = []

# Reset-Button
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if st.button("🧹 Neue Unterhaltung starten", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Begrüßung
if not st.session_state.history:
    with st.chat_message("assistant", avatar="🏨"):
        st.write("👋 Willkommen im **Hotel Bellevue Grand**! Wie kann ich Ihnen heute helfen?")
        st.write("Ich unterstütze Sie gern zu Check-in, Frühstück, Parken, Spa, WLAN, Zimmerservice und mehr.")

# Verlauf anzeigen
for role, text in st.session_state.history:
    avatar = "🧑" if role == "user" else "🏨"
    with st.chat_message(role, avatar=avatar):
        st.write(text)

# Eingabe + Antwort
user_msg = st.chat_input("Stellen Sie Ihre Frage …")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user", avatar="🧑"):
        st.write(user_msg)

    best, sim, top = find_best_answer(user_msg, df, vec, X, threshold=0.25, topk=3)
    if best is None:
        bot_text = "Dazu habe ich in meinem freigegebenen Katalog leider keine passende Antwort."
        picked_id = ""
    else:
        bot_text = best["answer"]
        picked_id = best["id"]

    with st.chat_message("assistant", avatar="🏨"):
        type_out(bot_text)

    st.session_state.history.append(("assistant", bot_text))
    log_event(user_msg, picked_id, sim if best is not None else 0.0)
