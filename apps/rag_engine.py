import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import shutil
import os
from langchain_ollama import OllamaLLM

# Definiamo una classe che gestisce TUTTO il processo RAG
class RAGModularPipeline:
    def __init__(self, db_type, embedding_model_name, reranker_model_name=None):
        """
        Inizializza la pipeline con i componenti scelti.
        :param db_type: 'chroma', 'faiss', o 'qdrant'
        :param embedding_model_name: nome del modello (es. 'all-MiniLM-L6-v2')
        :param reranker_model_name: nome del modello reranker (opzionale)
        """
        self.db_type = db_type
        self.embedding_model_name = embedding_model_name
        self.vector_db = None
        
        print(f" Inizializzazione RAG: DB={db_type}, Embedder={embedding_model_name}")
        
        # 1. Carichiamo l'Embedder (Il Traduttore)
        # Usiamo la CPU per compatibilità, se hai GPU cambierà da solo
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 2. Carichiamo il Reranker (Il Professore), se richiesto
        if reranker_model_name:
            self.reranker = CrossEncoder(reranker_model_name)
        else:
            self.reranker = None

    def create_index(self, documents):
        """
        Prende i documenti (chunks), calcola i vettori e li salva nel DB scelto.
        """
        start_time = time.time()
        print(f" Indicizzazione di {len(documents)} chunks in corso...")

        if self.db_type == 'chroma':
            # Chroma salva su disco in una cartella
            persist_dir = f"./chroma_db_{self.embedding_model_name.replace('/', '_')}"
            # Pulizia preventiva se esiste già
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            
        elif self.db_type == 'faiss':
            # FAISS lavora in RAM (velocissimo ma volatile)
            self.vector_db = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            
        elif self.db_type == 'qdrant':
            # Qdrant versione in-memory per test
            self.vector_db = QdrantVectorStore.from_documents(
                documents,
                self.embeddings,
                location=":memory:",  # Solo in RAM per questo test
                collection_name="my_documents"
            )
            
        end_time = time.time()
        indexing_time = end_time - start_time
        print(f" Indicizzazione completata in {indexing_time:.2f} secondi.")
        return indexing_time

    def retrieve(self, query, k=5):
        """
        Cerca i 5 documenti più simili alla domanda.
        """
        # Il Vector DB fa la ricerca semantica grezza
        docs = self.vector_db.similarity_search(query, k=k)
        return docs

    def retrieve_with_rerank(self, query, k=5):
        """
        Cerca 10 documenti, poi usa il Reranker per scegliere i 5 migliori.
        """
        # 1. Recuperiamo PIÙ documenti del necessario (es. 10)
        initial_k = 10
        initial_docs = self.vector_db.similarity_search(query, k=initial_k)
        
        if not self.reranker:
            return initial_docs[:k] # Se non c'è reranker, ridiamo i primi k

        # 2. Prepariamo le coppie [Query, Documento] per il Reranker
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # 3. Il Reranker assegna un punteggio a ogni coppia
        scores = self.reranker.predict(pairs)
        
        # 4. Ordiniamo i documenti in base al punteggio (dal più alto al più basso)
        # Zip unisce docs e scores, sorted li ordina
        sorted_docs_with_scores = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # 5. Prendiamo solo i top k
        reranked_docs = [doc for doc, score in sorted_docs_with_scores[:k]]
        
        return reranked_docs
    
    def generate_answer(self, query, retrieved_docs):
        # 1. Inizializziamo Mistral tramite Ollama
        llm = OllamaLLM(model="mistral")
        
        # 2. Uniamo i testi dei documenti trovati in un unico grande "contesto"
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 3. Scriviamo le istruzioni ferree (Il Prompt)
        prompt = f"""Sei un assistente virtuale aziendale gentile e professionale.
Usa SOLO ed ESCLUSIVAMENTE le informazioni contenute nel seguente Contesto per rispondere alla Domanda dell'utente. 
Se la risposta non è nel Contesto, dì semplicemente "Non ho questa informazione nel mio database", non inventare nulla.
Rispondi sempre in italiano.

Contesto:
{context}

Domanda: {query}

Risposta:"""

        # 4. Chiediamo a Mistral di generare la risposta
        return llm.invoke(prompt)