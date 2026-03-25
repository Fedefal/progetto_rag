from ingest import load_and_chunk_documents
from rag_engine import RAGModularPipeline

print(" Preparazione del sistema RAG in corso... (Attendere)")

# Carichiamo i documenti
docs = load_and_chunk_documents()

rag = RAGModularPipeline(
    db_type="faiss", 
    embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Creiamo l'indice
rag.create_index(docs)

print("\n Sistema pronto! Il Chatbot è in ascolto.")
print("Scrivi 'esci' per chiudere il programma.")
print("-" * 50)

while True:
    query = input("\n Tu: ")
    
    if query.lower() in ['esci', 'exit', 'quit']:
        print(" Arrivederci!")
        break
        
    print(" Ricerca dei documenti in corso...")
    
    # Peschiamo ben 10 documenti (k=10) per dare più contesto
    found_docs = rag.retrieve(query, k=10)
    
    print(" Mistral sta leggendo e scrivendo la risposta...")
    answer = rag.generate_answer(query, found_docs)
    
    print(f"\n Mistral: {answer}")
    
    # Vediamo i primi 100 caratteri di ogni pezzo trovato
    print("\n Le 10 Fonti lette da Mistral:")
    for i, doc in enumerate(found_docs):
        testo_tagliato = doc.page_content.replace('\n', ' ')[:100]
        print(f"  {i+1}) {doc.metadata.get('source', 'Sconosciuta')} -> '{testo_tagliato}...'")
    print("-" * 50)