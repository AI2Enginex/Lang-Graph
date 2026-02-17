from langchain_community.vectorstores import FAISS
from LLMUtils.LLMConfigs import  EmbeddingModel

# ========================== VECTOR STORE ============================

class Vectors:
    """
    Handles generating vector embeddings and storing them in FAISS.
    """
    embeddings = None

    @classmethod
    def initialize(cls, config=None):
        """
        Initializes the embedding model from the Gemini configuration.
        """
        try:
            cls.embeddings = EmbeddingModel(config=config).embeddings
            if cls.embeddings:
                print(f"Embedding model loaded: {config.embedding_model_name}")
            else:
                print("Embedding model failed to load.")
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")
            cls.embeddings = None

    @classmethod
    def generate_vectors_from_documents(cls, chunks=None):
        """
        Generates FAISS vector store from document chunks.
        """
        try:
            if cls.embeddings is None:
                print("Embedding model not initialized.")
                return None
            if not chunks:
                print("No chunks provided for vector generation.")
                return None
            return FAISS.from_documents(chunks, embedding=cls.embeddings, normalize_L2=True)
        except Exception as e:
            print(f"Error generating vectors: {e}")
            return None