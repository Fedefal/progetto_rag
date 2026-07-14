import time
import shutil
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM

# Directory base del file (apps/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class RAGModularPipeline:
    def __init__(
        self,
        db_type: str,
        embedding_model_name: str,
        reranker_model_name: str | None = None,
        llm_model: str = "mistral:latest",
    ):
        """
        :param db_type: 'chroma', 'faiss', o 'qdrant'
        :param embedding_model_name: es. 'sentence-transformers/all-MiniLM-L6-v2'
        :param reranker_model_name: es. 'cross-encoder/ms-marco-MiniLM-L-6-v2' (opzionale)
        :param llm_model: modello Ollama, es. 'mistral:latest'
        """
        self.db_type = db_type
        self.embedding_model_name = embedding_model_name
        self.vector_db = None

        print(
            f"Inizializzazione RAG: DB={db_type}, "
            f"Embedder={embedding_model_name}, LLM={llm_model}"
        )

        # Embedder
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Reranker
        if reranker_model_name:
            self.reranker = CrossEncoder(reranker_model_name)
        else:
            self.reranker = None

        # LLM (inizializzato una volta sola)
        self.llm = OllamaLLM(
            model=llm_model,
            temperature=0.2,
            num_predict=384,
            num_ctx=4096,
            keep_alive="10m",  # tiene il modello in RAM per un po'
        )

    def _get_persist_dir(self) -> str:
        """Costruisce il path assoluto alla cartella Chroma per questo embedder."""
        return os.path.join(
            BASE_DIR,
            f"chroma_db_{self.embedding_model_name.replace('/', '_')}",
        )

    def create_index(self, documents):
        """
        Prende i chunks (Document) e crea l'indice nel DB scelto.
        """
        start_time = time.time()
        print(f"Indicizzazione di {len(documents)} chunks in corso...")

        if self.db_type == "chroma":
            persist_dir = self._get_persist_dir()
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)

            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_dir,
            )

        elif self.db_type == "faiss":
            self.vector_db = FAISS.from_documents(documents, self.embeddings)

        elif self.db_type == "qdrant":
            self.vector_db = QdrantVectorStore.from_documents(
                documents,
                self.embeddings,
                location=":memory:",
                collection_name="my_documents",
            )
        else:
            raise ValueError("db_type deve essere 'chroma', 'faiss' o 'qdrant'.")

        indexing_time = time.time() - start_time
        print(f"Indicizzazione completata in {indexing_time:.2f} secondi.")
        return indexing_time

    def load_index(self):
        """
        Carica in memoria il database vettoriale Chroma già salvato su disco.
        """
        print("Caricamento del database vettoriale dal disco...")

        if self.db_type == "chroma":
            persist_dir = self._get_persist_dir()

            if not os.path.exists(persist_dir):
                raise FileNotFoundError(
                    f"Directory {persist_dir} non trovata. "
                    "Hai già eseguito lo script di ingest?"
                )

            self.vector_db = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
            print(f"✅ DB Chroma caricato con successo da: {persist_dir}")
        else:
            raise ValueError("load_index è implementato solo per 'chroma'.")

    def retrieve(self, query: str, k: int = 4, fetch_k: int = 12):
        """
        Retrieval con MMR per avere chunk meno ridondanti.
        """
        if not self.vector_db:
            raise ValueError(
                "Vector DB non inizializzato. Chiama create_index() o load_index() prima."
            )

        retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )
        docs = retriever.invoke(query)
        return docs

    def retrieve_with_rerank(self, query: str, k: int = 5, initial_k: int = 12):
        """
        Recupera initial_k documenti, poi usa il reranker CrossEncoder
        (se presente) per tenerne solo k.
        """
        if not self.vector_db:
            raise ValueError("Vector DB non inizializzato.")

        initial_docs = self.vector_db.similarity_search(query, k=initial_k)

        if not self.reranker:
            return initial_docs[:k]

        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        sorted_docs_with_scores = sorted(
            zip(initial_docs, scores), key=lambda x: x[1], reverse=True
        )
        reranked_docs = [doc for doc, _ in sorted_docs_with_scores[:k]]
        return reranked_docs

    def generate_answer(self, query: str, retrieved_docs):
        """
        Genera la risposta usando SOLO i documenti passati.
        """
        frammenti = []
        for i, doc in enumerate(retrieved_docs, start=1):
            pagina = doc.metadata.get("pagina", "Sconosciuta")
            fonte = doc.metadata.get("fonte", "sconosciuta")
            chunk_id = doc.metadata.get("chunk_id", "n.d.")

            frammenti.append(
                f"[DOC {i} | CHUNK {chunk_id} | FONTE {fonte} | PAGINA {pagina}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(frammenti)

        prompt = f"""Sei un assistente virtuale aziendale.

Rispondi sempre in italiano corretto, chiaro e professionale.

REGOLE OBBLIGATORIE:
- Usa SOLO le informazioni presenti nel CONTEXTO.
- NON inventare informazioni o procedure che non siano presenti.
- Se una informazione non è presente, rispondi esattamente:
  "Non ho questa informazione nel mio database".
- Ogni informazione importante deve indicare la pagina nel formato [Pagina X].
- Usa SOLO numeri di pagina che compaiono nel CONTEXTO.
- Se prendi informazioni da più pagine, cita ogni parte con la pagina corretta.
- Mantieni la risposta il più sintetica possibile ma completa.

CONTEXTO:
{context}

DOMANDA:
{query}

RISPOSTA (in italiano):
"""

        return self.llm.invoke(prompt)


# --- FACTORY PER L'APP / STREAMLIT ---


def create_rag_engine() -> RAGModularPipeline:
    """
    Factory da usare nell'app (NON nell'ingest):
    crea il motore, carica l'indice e lo restituisce pronto.
    """
    motore = RAGModularPipeline(
        db_type="chroma",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        # reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model="mistral:latest",
    )
    motore.load_index()
    return motore