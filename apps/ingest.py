import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_and_chunk_documents():
    print("Avvio del Semantic Chunking...")
    
    # Inizializziamo l'IA che farà da "bisturi" semantico per tagliare le frasi
    embedder_for_chunking = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_splitter = SemanticChunker(embedder_for_chunking, breakpoint_threshold_type="percentile")
    
    docs = []
    
    # 1. PDF Tecnico
    if os.path.exists("data/manuale.pdf"):
        print(" Analisi semantica del PDF in corso...")
        docs.extend(semantic_splitter.split_documents(PyPDFLoader("data/manuale.pdf").load()))
    else:
        print("Attenzione: data/manuale.pdf non trovato!")
        
    # 2. Pagina Web (TXT)
    if os.path.exists("data/articolo_web.txt"):
        print("Analisi semantica del file TXT in corso...")
        docs.extend(semantic_splitter.split_documents(TextLoader("data/articolo_web.txt", encoding='utf-8').load()))
    else:
        print("⚠️ Attenzione: data/articolo_web.txt non trovato!")
        
    # 3. FAQ (CSV) -> Qui NON usiamo il Semantic Chunker, lasciamo le righe intatte!
    if os.path.exists("data/faq.csv"):
        print("📊 Caricamento FAQ da CSV (chunking naturale per riga)...")
        docs.extend(CSVLoader("data/faq.csv").load())
    else:
        print("Attenzione: data/faq.csv non trovato!")
        
    print(f"Fatto! Generati {len(docs)} chunks intelligenti.")
    return docs

# Questo blocco serve solo se vuoi testare il file singolarmente
if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    
    # Stampa il primo chunk per vedere come ha tagliato bene la frase
    if chunks:
        print("\n--- Esempio del primo chunk semantico ---")
        print(f"Contenuto: {chunks[0].page_content}")
        print(f"Metadati (Fonte): {chunks[0].metadata.get('source', 'Sconosciuta')}")
        print("-----------------------------------------")