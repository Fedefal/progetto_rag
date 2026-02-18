import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CONFIGURAZIONE
# Definiamo come spezzettare il testo.
# chunk_size=500: Ogni pezzo sarà di circa 500 caratteri.
# chunk_overlap=50: I pezzi si sovrappongono un po' per non perdere il filo del discorso tra uno e l'altro.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = []

#2. CARICAMENTO PDF (Tecnico)
pdf_path = "data/manuale.pdf" 
if os.path.exists(pdf_path):
    print(f"Caricamento PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    # Chunking del PDF
    pdf_chunks = text_splitter.split_documents(pdf_docs)
    documents.extend(pdf_chunks)
    print(f" -> PDF diviso in {len(pdf_chunks)} chunks.")

#3. CARICAMENTO WEB (Sporco - simulato da txt)
web_path = "data/articolo_web.txt"
if os.path.exists(web_path):
    print(f"Caricamento Web Txt: {web_path}...")
    loader = TextLoader(web_path, encoding='utf-8')
    web_docs = loader.load()
    # Chunking del Web
    web_chunks = text_splitter.split_documents(web_docs)
    documents.extend(web_chunks)
    print(f" -> Web diviso in {len(web_chunks)} chunks.")

# 4. CARICAMENTO CSV (FAQ - Strutturato)
# Per le FAQ, il chunking è diverso: di solito 1 riga = 1 chunk
csv_path = "data/faq.csv"
if os.path.exists(csv_path):
    print(f"Caricamento CSV: {csv_path}...")
    loader = CSVLoader(csv_path)
    csv_docs = loader.load()
    # Qui NON usiamo il text_splitter classico, perché ogni riga è già un'unità di senso
    documents.extend(csv_docs)
    print(f" -> CSV caricato come {len(csv_docs)} chunks.")

# 5. VERIFICA
print("-" * 30)
print(f"TOTALE CHUNKS GENERATI: {len(documents)}")
if len(documents) > 0:
    print("Esempio di un chunk (contenuto):")
    print(documents[0].page_content)
    print("Metadati:", documents[0].metadata)