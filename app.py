import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import time, random  # <— nach oben geholt

st.set_page_config(page_title="AI-Chatbot", page_icon="💬")

# ——— Globales CSS: Hintergründe, Breite, Kartenoptik, Fonts ———
st.markdown("""
<style>
/* Optional: Google Font (für feineres Schriftbild) */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] * { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

/* Gesamt-Hintergrund mit sanftem Verlauf */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 500px at 20% -10%, #E8F1FF 0%, rgba(232,241,255,0) 60%),
              radial-gradient(900px 400px at 90% 0%, #F7E9FF 0%, rgba(247,233,255,0) 55%),
              #F7F9FC;
}

/* Hauptbreite und zentrierte Bühne */
section.main > div {
  max-width: 880px !important;    /* Bühne für Content */
  margin: 0 auto !important;
  padding-top: 1rem !important;
}

/* Header-Block (dein Branding oben) etwas kompakter */
h1, h2, h3 { letter-spacing: 0.2px; margin-top: 0.2rem; }

/* Chatkarten (Streamlit chat messages) aufwerten */
div[data-testid="stChatMessage"] {
  background: var(--background-color, #FFFFFF);
  border: 1px solid rgba(15, 23, 42, 0.06);
  border-radius: 16px;
  padding: 0.75rem 0.9rem;
  box-shadow: 0 8px 24px rgba(2, 6, 23, 0.04);
  margin-bottom: 0.6rem;
}

/* User vs. Assistant subtil unterscheiden */
div[data-testid="stChatMessage"] p { margin: 0.1rem 0; line-height: 1.5; }
div[data-testid="stChatMessage"]:has(img[alt="🧑"]) { background: #F3F6FB; }        /* User */
div[data-testid="stChatMessage"]:has(img[alt="🏨"]) { background: #FFFFFF; }        /* Hotel */

/* Chat-Input-Leiste etwas luftiger */
div[data-testid="stChatInput"] textarea {
  border-radius: 12px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
}

/* Buttons runder & dezente Hover-States */
button[kind="primary"] {
  border-radius: 999px !important;
  box-shadow: 0 4px 14px rgba(30,136,229,.18) !important;
}
button:hover { filter: brightness(0.98); }

/* Trennlinie dezenter */
hr { border: none; border-top: 1px solid rgba(15,23,42,0.08); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="background:#ffffff;border:1px solid rgba(15,23,42,0.06);border-radius:16px;
            padding:18px 22px;box-shadow:0 8px 24px rgba(2,6,23,0.04);margin-bottom:12px;text-align:center;">
  <h1 style="margin:0 0 6px 0;">🏨 Hotel-Service Chat</h1>
  <p style="margin:0;color:#475569;">Schnelle Hilfe beim Check-in, Zimmer &amp; mehr mit unserem KI-Chatbot</p>
  <div style="margin-top:10px;">
    <span style="display:inline-block;padding:6px 12px;border-radius:999px;background:#E8F5E9;color:#2E7D32;font-size:12px;">
      Rezeption heute: 07:00–23:00
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


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

# Titel mittig (per HTML)
st.markdown('<h1 style="text-align:center;">AI-Chatbot</h1>', unsafe_allow_html=True)

# Bild zentriert
center_col = st.columns([2, 2, 2])[1]
with center_col:
    st.image("AI-Chatbot.jpg", width=200)

df, vec, X = load_kb("answers.csv")
if "history" not in st.session_state:
    st.session_state.history = []

# Button zum Löschen des bisherigen Chatverlaufs
if st.button("🧹 Neue Unterhaltung starten"):
    st.session_state.history = []
    st.rerun()

# Willkommensnachricht nur wenn keine Historie
if not st.session_state.history:
    with st.chat_message("assistant"):
        st.write("Willkommen im Hotel! Wie kann ich helfen?")

# Bisherige Nachrichten anzeigen
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# Eingabefeld unten
user_msg = st.chat_input("Frag mich etwas …")
if user_msg:
    # Nutzer-Nachricht anzeigen + speichern
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)

    # Antwort bestimmen
    best, sim, top = find_best_answer(user_msg, df, vec, X, threshold=0.25, topk=3)
    if best is None:
        bot_text = "Dazu kann ich dir leider nicht weiterhelfen."
        picked_id = ""
    else:
        bot_text = best["answer"]
        picked_id = best["id"]

    # Realistische Tipp-Animation + Ausgabe (JETZT liegt bot_text vor!)
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

    # Bot-Antwort im Verlauf speichern (damit sie nach Rerun bleibt)
    st.session_state.history.append(("assistant", bot_text))

    # Logging
    log_event(user_msg, picked_id, sim if best is not None else 0.0)
