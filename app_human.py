import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os, base64, time, random

st.set_page_config(page_title="Mitarbeiter-Chat", page_icon="💬")

# ---- Logo laden & als Base64 einbetten (vermeidet Pfad-/Serving-Probleme) ----
def to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

# Rechtsseitiges Chatbot-Logo
LOGO_PATH = "Mitarbeiter.jpg"
LOGO_B64 = to_b64(LOGO_PATH)

# Header-Bild oben mittig
HEADER_LOGO_PATH = "bed.jpg"
HEADER_LOGO_B64 = to_b64(HEADER_LOGO_PATH)

# ---- Kopfbereich mit Headerbild (Banner-Stil), Titel & Reset-Button ----
HEADER_IMG_PATH = "bed.jpg"
HEADER_IMG_B64 = to_b64(HEADER_IMG_PATH)

if HEADER_IMG_B64:
    st.markdown(
        f"""
         <div style='position:relative; text-align:center; margin-top:-20px; margin-bottom:1rem;'>
            <img src='data:image/jpeg;base64,{HEADER_IMG_B64}'
                 alt='Hotel Header'
                 style='width:100%; max-height:200px; object-fit:cover; border-radius:0 0 20px 20px;
                        box-shadow:0 4px 14px rgba(0,0,0,0.15);'>
            <div style='position:absolute; bottom:25px; left:0; width:100%; text-align:center; color:white;
                        text-shadow:0 2px 6px rgba(0,0,0,0.5);'>
                <h1 style='font-size:2.2rem; margin-bottom:0.2rem;'>🏨 Hotel Bellevue Grand</h1>
                <p style='font-size:1.05rem; margin-top:0;'>Schnelle Hilfe beim Check-in, Zimmer & mehr mit unserem Mitarbeiter-Chat</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='text-align:center; margin-bottom:0.5rem;'>"
        "<h1 style='margin-bottom:0.2rem;'>🏨 Hotel Bellevue Grand</h1>"
        "<p style='margin-top:0; color:#666;'>Schnelle Hilfe beim Check-in, Zimmer & mehr mit unserem Mitarbeiter-Chat<</p>"
        "</div>",
        unsafe_allow_html=True
    )

# Reset-Button oben rechts
#reset_col = st.columns([5, 1])[1]
#with reset_col:
#    if st.button('🧹 Unterhaltung neu starten', key='btn_reset_top'):
#        st.session_state.history = []
#        st.rerun()


# ---- Fixiertes Seiten-Panel rechts ----
img_tag = f"<img src='data:image/png;base64,{LOGO_B64}' alt='Chatbot Logo'>" if LOGO_B64 else ""
st.markdown("""
<style>
/* ========== Design Preset: Hotel Bellevue Grand ========== */

/* — Farben & Typo — */
:root{
  --brand:#8fd1f2;     /* Primärfarbe */
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

/* — App-Layout — */
[data-testid="stAppViewContainer"]{ background:var(--bg); }
main [data-testid="block-container"]{
  max-width: 980px;
  padding-top: 0rem;
  padding-bottom: 2rem;
  padding-right: 290px; /* Platz rechts für Sidepanel */
}

/* Buttons */
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



/* Scrollbar */
::-webkit-scrollbar{ width:10px; }
::-webkit-scrollbar-thumb{
  background:#C9D4E3; border-radius:8px; border:2px solid transparent; background-clip: padding-box;
}
::-webkit-scrollbar-track{ background:transparent; }

/* Fixiertes Seitenpanel */
.fixed-sidebox{
  position:fixed;
  top:120px;  /* tiefer wegen Header-Bild */
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
  max-width:65%;
  height:auto;
  border-radius:12px;
}
@media (max-width: 1100px){
  main [data-testid="block-container"]{ padding-right: 0; }
}
@media (max-width:900px){
  .fixed-sidebox{ display:none; }
}
</style>

<div class="fixed-sidebox">
""" + img_tag + """
<p style='margin:0; font-weight:600; font-size:14px;'>Sarah</p>
<h3>Mitarbeiter-Chat</h3>
</div>
""", unsafe_allow_html=True)

# ---- Daten laden & Helfer ----
@st.cache_resource
def load_kb(csv_path="answers.csv"):
    df = pd.read_csv(csv_path).fillna("")
    if not {"id","question","answer"}.issubset(df.columns):
        raise ValueError("CSV braucht Spalten: id, question, answer")
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words=None, lowercase=True)
    X = vec.fit_transform(df["question"].tolist())
    return df, vec, X

def find_best_answer(user_text, df, vec, X, threshold=0.20, topk=3):
    q = vec.transform([user_text])
    sims = cosine_similarity(q, X).flatten()
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    if best_sim < threshold:
        return None, best_sim, []
    top_idx = sims.argsort()[::-1][:topk]
    top = [(df.iloc[i]["id"], df.iloc[i]["question"], float(sims[i])) for i in top_idx]
    return df.iloc[best_idx], best_sim, top

import gspread
from google.oauth2.service_account import Credentials

# Scopes: Sheets lesen/schreiben + Drive lesen (für open_by_…)
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

@st.cache_resource(show_spinner=False)
def _get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=_SCOPES
    )
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def _open_worksheet():
    gc = _get_gsheet_client()
    ss_conf = st.secrets["sheets"]
    # Öffnen per Name ODER per ID (falls in secrets gesetzt)
    if "spreadsheet_id" in ss_conf and ss_conf["spreadsheet_id"]:
        sh = gc.open_by_key(ss_conf["spreadsheet_id"])
    else:
        sh = gc.open(ss_conf["spreadsheet_name"])
    try:
        ws = sh.worksheet(ss_conf.get("worksheet_name", "Logs"))
    except gspread.WorksheetNotFound:
        # Falls Blatt nicht existiert: anlegen und Header setzen
        ws = sh.add_worksheet(title=ss_conf.get("worksheet_name", "Logs"), rows="1000", cols="10")
        ws.update("A1:E1", [["timestamp", "user_text", "picked_id", "similarity", "session_id"]])
    return ws

def log_event_to_gsheet(timestamp_iso: str, user_text: str, picked_id: str, similarity: float, session_id: str | None = None):
    """Schreibt ein Log-Event in Google Sheets."""
    ws = _open_worksheet()
    row = [timestamp_iso, user_text, picked_id, similarity, session_id or ""]
    ws.append_row(row, value_input_option="USER_ENTERED")



def log_event(user_text, picked_id, sim, logfile="logs.csv"):
    ts = datetime.utcnow().isoformat()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_text": user_text,
        "picked_id": picked_id,
        "similarity": sim,
    }
    exists = os.path.exists(logfile)
    pd.DataFrame([row]).to_csv(logfile, mode="a", index=False, header=not exists)

    session_id = st.session_state.get("session_id", "")
    try:
        log_event_to_gsheet(ts, user_text, picked_id, sim, session_id=session_id)
    except Exception as e:
        # Nicht hart failen, nur Info loggen:
        st.warning(f"Log in Google Sheets fehlgeschlagen: {e}")

# ---- Hauptlogik ----
df, vec, X = load_kb("answers.csv")
if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    with st.chat_message("assistant", avatar="👩‍💼"):
        st.write("Willkommen im Hotel! Mein Name ist Sarah. Wie kann ich helfen?")

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

user_msg = st.chat_input("Frag mich etwas …")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user", avatar="🧒"):
        st.write(user_msg)

    best, sim, top = find_best_answer(user_msg, df, vec, X, threshold=0.20, topk=3)
    if best is None:
        bot_text = "Das weiß ich leider nicht."
        picked_id = ""
    else:
        bot_text = best["answer"]
        picked_id = best["id"]

    with st.chat_message("assistant", avatar="👩‍💼"):
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

import gspread
from google.oauth2.service_account import Credentials

# Scopes: Sheets lesen/schreiben + Drive lesen (für open_by_…)
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

@st.cache_resource(show_spinner=False)
def _get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=_SCOPES
    )
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def _open_worksheet():
    gc = _get_gsheet_client()
    ss_conf = st.secrets["sheets"]
    # Öffnen per Name ODER per ID (falls in secrets gesetzt)
    if "spreadsheet_id" in ss_conf and ss_conf["spreadsheet_id"]:
        sh = gc.open_by_key(ss_conf["spreadsheet_id"])
    else:
        sh = gc.open(ss_conf["spreadsheet_name"])
    try:
        ws = sh.worksheet(ss_conf.get("worksheet_name", "Logs"))
    except gspread.WorksheetNotFound:
        # Falls Blatt nicht existiert: anlegen und Header setzen
        ws = sh.add_worksheet(title=ss_conf.get("worksheet_name", "HumanLogs"), rows="1000", cols="10")
        ws.update("A1:D1", [["timestamp", "user_text", "picked_id", "similarity"]])
    return ws

def log_event_to_gsheet(timestamp_iso: str, user_text: str, picked_id: str, similarity: float, session_id: str | None = None):
    ws = _open_worksheet()
    # optional: Session-ID als 5. Spalte
    row = [timestamp_iso, user_text, picked_id, similarity]
    if session_id is not None:
        # Stelle sicher, dass deine Headline 5 Spalten hat, oder erweitere automatisch:
        # ws.update("A1:E1", [["timestamp", "user_text", "picked_id", "similarity", "session_id"]])
        row.append(session_id)
    ws.append_row(row, value_input_option="USER_ENTERED")




