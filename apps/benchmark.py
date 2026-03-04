import json
import time
import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from rag_engine import RAGModularPipeline  

# --- GESTIONE ERRORI PER LO SPLITTER ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. FUNZIONE PER CARICARE I DOCUMENTI (Come in ingest.py)
def load_all_documents():
    print(" Caricamento dei documenti per la gara...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    
    # PDF
    if os.path.exists("data/manuale.pdf"):
        docs.extend(text_splitter.split_documents(PyPDFLoader("data/manuale.pdf").load()))
    # TXT
    if os.path.exists("data/articolo_web.txt"):
        docs.extend(text_splitter.split_documents(TextLoader("data/articolo_web.txt", encoding='utf-8').load()))
    # CSV
    if os.path.exists("data/faq.csv"):
        docs.extend(CSVLoader("data/faq.csv").load())
        
    print(f"Trovati {len(docs)} chunks totali.")
    return docs

# 2. FUNZIONE PER CARICARE IL GOLDEN DATASET (Le 20 domande)
def load_golden_dataset():
    with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

# --- INIZIO DELLA GARA ---
def run_benchmark():
    documents = load_all_documents()
    golden_dataset = load_golden_dataset()
    
    
    configurations = [
        {
            "name": "FAISS_Base",
            "db_type": "faiss",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": None
        },
        {
            "name": "Chroma_Base",
            "db_type": "chroma",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": None
        },
        
        {
            "name": "Qdrant_Base",
            "db_type": "qdrant",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": None
        },
        # -------------------------------------
        {
            "name": "FAISS_con_Reranker",
            "db_type": "faiss",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },

        {
            "name": "FAISS_Multilingual",
            "db_type": "faiss",
            "embedder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "reranker": None
        },
        {
            "name": "FAISS_BGE_M3",
            "db_type": "faiss",
            "embedder": "BAAI/bge-m3",
            "reranker": None
        }
    ]

    results = []

    print("\n INIZIO DEL BENCHMARK \n")

    for config in configurations:
        print(f"🚀 Testando Configurazione: {config['name']}")
        
        # Inizializziamo il motore
        rag = RAGModularPipeline(
            db_type=config["db_type"],
            embedding_model_name=config["embedder"],
            reranker_model_name=config["reranker"]
        )
        
        # Test 1: Velocità di indicizzazione
        index_time = rag.create_index(documents)
        
        # Variabili per il Test 2 e 3
        correct_retrievals = 0
        total_retrieval_time = 0
        
        print("   🔍 Risoluzione delle 20 domande in corso...")
        
        for item in golden_dataset:
            query = item["question"]
            expected_source = item["source"]
            
            # Cronometriamo quanto ci mette a rispondere
            start_q = time.time()
            if config["reranker"]:
                retrieved_docs = rag.retrieve_with_rerank(query, k=3)
            else:
                retrieved_docs = rag.retrieve(query, k=3)
            end_q = time.time()
            
            total_retrieval_time += (end_q - start_q)
            
            # Verifichiamo la precisione: il documento giusto è tra i primi 3 trovati?
            is_correct = False
            for doc in retrieved_docs:
                # Controlliamo se la 'source' (es. manuale.pdf) è nei metadati del documento trovato
                if expected_source in doc.metadata.get("source", ""):
                    is_correct = True
                    break
            
            if is_correct:
                correct_retrievals += 1

        # Calcoliamo le medie
        avg_retrieval_time = total_retrieval_time / len(golden_dataset)
        accuracy_percent = (correct_retrievals / len(golden_dataset)) * 100
        
        print(f"    Risultato {config['name']}: Accuratezza {accuracy_percent}% | Tempo medio: {avg_retrieval_time:.3f} sec\n")
        
        # Salviamo i risultati di questo giro
        results.append({
            "Configurazione": config["name"],
            "Database": config["db_type"],
            "Reranker": "Sì" if config["reranker"] else "No",
            "Tempo Indicizzazione (sec)": round(index_time, 2),
            "Tempo Risposta Medio (sec)": round(avg_retrieval_time, 3),
            "Accuratezza (%)": accuracy_percent
        })

    # --- FINE GARA: SALVATAGGIO RISULTATI ---
    df_results = pd.DataFrame(results)
    df_results.to_csv("risultati_benchmark.csv", index=False)
    
    print("BENCHMARK COMPLETATO")
    print("\nEcco la classifica finale:")
    print(df_results.to_string(index=False))
    print("\nI risultati sono stati salvati in 'risultati_benchmark.csv'. Puoi aprirlo con Excel!")

if __name__ == "__main__":
    run_benchmark()