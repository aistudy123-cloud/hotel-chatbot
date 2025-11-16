import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

st.set_page_config(page_title="AI-Chatbot", page_icon="💬")



st.markdown(
    """
    <div style="text-align:center; margin-bottom:0.5rem;">
      <h1 style="margin-bottom:0.2rem;">🏨 Hotel-Service Chat</h1>
      <p style="margin-top:0; color:#666;">Schnelle Hilfe beim Check-in, Zimmer & mehr mit unserem KI-Chatbot</p>
      <span style="display:inline-block; padding:4px 10px; border-radius:999px; background:#e8f5e9; color:#2e7d32; font-size:12px;">
        Rezeption heute: 07:00–23:00
      </span>
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



#st.title("KI-Chatbot ")

#st.image("AI-Chatbot.jpg", width=150)

# Titel mittig (per HTML) …
st.markdown('<h1 style="text-align:center;">AI-Chatbot</h1>', unsafe_allow_html=True)

# … und Bild sauber zentrieren mit Spalten
center_col = st.columns([2, 2, 2])[1]
with center_col:
    st.image("AI-Chatbot.jpg", width=200)   




#with st.expander("Hinweis zum Datenschutz"):
 #   st.write(
  #      "Eingaben werden nur für Studienzwecke gespeichert (ohne Klarnamen). "
 #       "Du kannst jederzeit aufhören. Keine Diagnose oder Beratung."
 #   )

df, vec, X = load_kb("answers.csv")
if "history" not in st.session_state:
    st.session_state.history = []

# Button zum Löschen des bisherigen Chatverlaufs
if st.button("🧹 Neue Unterhaltung starten"):
    st.session_state.history = []
    st.rerun()

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
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)

    best, sim, top = find_best_answer(user_msg, df, vec, X, threshold=0.25, topk=3)
    if best is None:
        bot_text = "Dazu kann ich dir leider nicht weiterhelfen."
        picked_id = ""
    else:
        bot_text = best["answer"]
        picked_id = best["id"]


import time, random

with st.chat_message("assistant", avatar="🏨"):
    dots = st.empty()
    for i in range(3):
        dots.markdown(f"_schreibt{'.' * ((i % 3) + 1)}_")
        time.sleep(0.35)
    dots.empty()

    output = st.empty()
    displayed = ""
    typing_speed = random.uniform(0.01, 0.03)  # leicht variierende Tippgeschwindigkeit
    for ch in bot_text:
        displayed += ch
        output.markdown(displayed)
        time.sleep(typing_speed)

    


    with st.chat_message("assistant"):
        st.write(bot_text)
    st.session_state.history.append(("assistant", bot_text))
     

    log_event(user_msg, picked_id, sim if best is not None else 0.0)

