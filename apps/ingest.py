import os
import json

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag_engine import RAGModularPipeline


PDF_PATH = "data/manuale.pdf"
JSON_OUTPUT = "data/manuale_strutturato.json"


def fase_1_e_2_estrazione_e_json(pdf_path: str, json_path: str):
    """
    Legge il PDF, estrae le pagine e genera un file JSON strutturato
    (1 blocco per pagina).
    """
    print("\n🔍 FASE 1 & 2: Parsing PDF e Creazione JSON strutturato...")

    if not os.path.exists(pdf_path):
        print(f"❌ Errore: file {pdf_path} non trovato!")
        return None

    loader = PyMuPDFLoader(pdf_path)
    pagine_grezze = loader.load()  # 1 Document per pagina con metadata 'page'

    dati_strutturati = []

    for i, pagina in enumerate(pagine_grezze):
        testo = pagina.page_content.strip()
        if not testo:
            continue

        blocco_json = {
            "id_blocco": f"blocco_pag_{i + 1}",
            "fonte": os.path.basename(pdf_path),
            "pagina": i + 1,  # 1-based per l'utente
            "page_loader": pagina.metadata.get("page", i),  # 0-based del loader
            "categoria": "manuale_tecnico",
            "contenuto_testuale": testo,
        }
        dati_strutturati.append(blocco_json)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dati_strutturati, f, indent=4, ensure_ascii=False)

    print(f"✅ File JSON generato: {json_path}")
    print(f"📊 Totale blocchi strutturati: {len(dati_strutturati)}")

    return dati_strutturati


def fase_3_indicizzazione_con_payload(dati_json):
    """
    Trasforma il JSON in Document con metadati, fa il chunking
    e restituisce i chunks pronti per l'indicizzazione.
    """
    print("\n💾 FASE 3: Chunking e Indicizzazione con metadati...")

    docs_da_indicizzare = []

    # Ogni blocco JSON diventa un Document con metadati completi
    for blocco in dati_json:
        doc = Document(
            page_content=blocco["contenuto_testuale"],
            metadata={
                "fonte": blocco["fonte"],
                "pagina": blocco["pagina"],
                "page_loader": blocco.get("page_loader"),
                "categoria": blocco["categoria"],
                "id_blocco": blocco["id_blocco"],
            },
        )
        docs_da_indicizzare.append(doc)

    # Split "ragionato": chunk più piccoli con overlap moderato
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "; ", ": ", " "],
        chunk_size=500,
        chunk_overlap=80,
    )

    chunks_finali = text_splitter.split_documents(docs_da_indicizzare)
    print(f"✂️ Taglio completato: ottenuti {len(chunks_finali)} chunks con payload.")

    # Aggiungiamo un ID di chunk per debug
    for idx, chunk in enumerate(chunks_finali):
        chunk.metadata["chunk_id"] = idx

    return chunks_finali


if __name__ == "__main__":
    print("🚀 AVVIO PIPELINE AVANZATA RAG...")

    # 1. Estrazione e salvataggio JSON
    dati_json = fase_1_e_2_estrazione_e_json(PDF_PATH, JSON_OUTPUT)

    if dati_json:
        # 2. Creazione chunks con metadati
        chunks_pronti = fase_3_indicizzazione_con_payload(dati_json)

        # 3. Creazione indice vettoriale
        print("\n⚙️ Avvio del Motore RAG per il salvataggio su Chroma...")
        motore_rag = RAGModularPipeline(
            db_type="chroma",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            llm_model="mistral:instruct",
        )

        motore_rag.create_index(chunks_pronti)
        print("\n🎉 PIPELINE COMPLETATA! Database aggiornato con struttura JSON e payload.")
    else:
        print("⛔ Pipeline interrotta: nessun dato JSON generato.")