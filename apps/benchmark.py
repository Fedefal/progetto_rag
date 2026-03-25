import json
import time
import pandas as pd

# --- IL COLLEGAMENTO AI NOSTRI FILE ---
from ingest import load_and_chunk_documents
from rag_engine import RAGModularPipeline

# 1. Carichiamo le 20 domande
def load_golden_dataset():
    with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

# --- INIZIO DELLA GARA ---
def run_benchmark():
    # 2. CHIAMIAMO L'INGESTION AVANZATA (I famosi 148 chunk semantici!)
    documents = load_and_chunk_documents()
    golden_dataset = load_golden_dataset()
    
    # Le 6 Scuderie
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

    print("\n INIZIO DEL BENCHMARK SEMANTICO \n")

    for config in configurations:
        print(f" Testando Configurazione: {config['name']}")
        
        rag = RAGModularPipeline(
            db_type=config["db_type"],
            embedding_model_name=config["embedder"],
            reranker_model_name=config["reranker"]
        )
        
        index_time = rag.create_index(documents)
        
        correct_retrievals = 0
        total_retrieval_time = 0
        
        print("    Risoluzione delle 20 domande in corso...")
        
        for item in golden_dataset:
            query = item["question"]
            expected_source = item["source"]
            
            start_q = time.time()
            if config["reranker"]:
                retrieved_docs = rag.retrieve_with_rerank(query, k=3)
            else:
                retrieved_docs = rag.retrieve(query, k=3)
            end_q = time.time()
            
            total_retrieval_time += (end_q - start_q)
            
            is_correct = False
            for doc in retrieved_docs:
                if expected_source in doc.metadata.get("source", ""):
                    is_correct = True
                    break
            
            if is_correct:
                correct_retrievals += 1

        avg_retrieval_time = total_retrieval_time / len(golden_dataset)
        accuracy_percent = (correct_retrievals / len(golden_dataset)) * 100
        
        print(f"    Risultato {config['name']}: Accuratezza {accuracy_percent}% | Tempo medio: {avg_retrieval_time:.3f} sec\n")
        
        results.append({
            "Configurazione": config["name"],
            "Database": config["db_type"],
            "Reranker": "Sì" if config["reranker"] else "No",
            "Tempo Indicizzazione (sec)": round(index_time, 2),
            "Tempo Risposta Medio (sec)": round(avg_retrieval_time, 3),
            "Accuratezza (%)": accuracy_percent
        })

    df_results = pd.DataFrame(results)
    # Salvo con un nome nuovo così non sovrascrivo quello vecchio
    df_results.to_csv("risultati_benchmark_semantico.csv", index=False)
    
    print(" BENCHMARK COMPLETATO ")
    print("\nEcco la classifica finale con il Chunking Avanzato:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()