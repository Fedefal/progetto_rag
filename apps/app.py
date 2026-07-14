import time
import streamlit as st

from rag_engine import create_rag_engine


# ---------- 1. CONFIGURAZIONE PAGINA ----------

st.set_page_config(
    page_title="RAG Aziendale (Privacy-First)",
    page_icon="🔒",
    layout="centered",
)

st.title("🤖 Assistant Aziendale (Mistral Locale)")
st.markdown(
    "Fai una domanda. Il sistema cercherà le risposte usando il **Semantic Chunking** "
    "e un database vettoriale locale (**Chroma**)."
)
st.divider()


# ---------- 2. INIZIALIZZAZIONE MOTORE RAG (UNA VOLTA SOLA) ----------

@st.cache_resource(show_spinner=True)
def get_rag_engine():
    # Usa la factory definita in rag_engine.py
    return create_rag_engine()


motore_rag = get_rag_engine()


def interroga_documenti(domanda_utente: str) -> str:
    """Funzione ponte: retrieval + generazione risposta."""
    documenti_trovati = motore_rag.retrieve(domanda_utente)
    risposta_finale = motore_rag.generate_answer(domanda_utente, documenti_trovati)
    return risposta_finale


# ---------- 3. GESTIONE DELLA CRONOLOGIA (STATEFUL UI) ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi passati a schermo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- 4. INPUT UTENTE E GENERAZIONE RISPOSTA ----------

prompt = st.chat_input("Cerca nei documenti aziendali...")

if prompt:
    # Mostra domanda utente
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Risposta dell'assistente
    with st.chat_message("assistant"):
        with st.spinner("Ricerca vettoriale ed elaborazione LLM in corso..."):
            risposta = interroga_documenti(prompt)
            st.markdown(risposta)

    # Salva nella cronologia
    st.session_state.messages.append({"role": "assistant", "content": risposta})