import os
from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import cleantext

# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key =  "AIzaSyC4m4mJdC4Meic3W6501FjdFX-giYFgE28"
os.environ['GOOGLE_API_KEY'] = api_key
# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')

# ========================== GEMINI CONFIGURATION ============================

class GeminiConfig:
    """
    Stores all parameters for Gemini models (Chat & Embeddings).
    API keys are passed separately to avoid hardcoding.
    """
    def __init__(
        self,
        chat_model_name: str = "gemini-2.5-flash",
        embedding_model_name: str = "models/gemini-embedding-001",
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 32,
        candidate_count: int = 1,
        max_output_tokens: int = 3000,
        generation_max_tokens: int = 8192
    ):
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.candidate_count = candidate_count
        self.max_output_tokens = max_output_tokens
        self.generation_max_tokens = generation_max_tokens

# ========================== QA STATE ============================

class QAState(BaseModel):
    """
    Stores state for QA system.
    """
    question: str
    retrieved_chunks: List[str]
    answer: str
    prompt_type: str

# ========================== GEMINI MODELS ============================

class GeminiModel:
    """
    Generic Gemini model wrapper using Google Generative AI SDK.
    """
    def __init__(self, config: GeminiConfig):
        try:
            self.config = config
            self.model = genai.GenerativeModel(self.config.chat_model_name)
            self.generation_config = genai.GenerationConfig(
                temperature=0,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                candidate_count=self.config.candidate_count,
                max_output_tokens=self.config.generation_max_tokens
            )
        except Exception as e:
            print(f"Error initializing GeminiModel: {e}")
            self.model = None
            self.generation_config = None

class GeminiChatModel(GeminiModel):
    """
    Wrapper to start a Gemini chat session.
    """
    def __init__(self, config: GeminiConfig):
        try:
            super().__init__(config)
            if self.model:
                self.chat = self.model.start_chat()
            else:
                self.chat = None
        except Exception as e:
            print(f"Error initializing GeminiChatModel: {e}")
            self.chat = None

class ChatGoogleGENAI:
    """
    Wrapper for ChatGoogleGenerativeAI with proper configuration.
    """
    def __init__(self, config: GeminiConfig, api_key: str):
        try:
            self.config = config
            self.llm = ChatGoogleGenerativeAI(
                temperature=self.config.temperature,
                model=self.config.chat_model_name,
                google_api_key=api_key,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                candidate_count=self.config.candidate_count,
                max_output_tokens=self.config.max_output_tokens
            )
        except Exception as e:
            print(f"Error initializing ChatGoogleGENAI: {e}")
            self.llm = None

class EmbeddingModel:
    """
    Wrapper for Gemini embedding model.
    """
    def __init__(self, config: GeminiConfig, api_key: str):
        try:
            self.config = config
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model_name,
                google_api_key=api_key
            )
        except Exception as e:
            print(f"Error initializing EmbeddingModel: {e}")
            self.embeddings = None

# ========================== FILE UTILITIES ============================

class ReadFile:
    """
    Utility to read text from PDFs or plain text files.
    """
    @classmethod
    def read_file_text(cls, folder_name=None):
        try:
            text = ""
            with open(folder_name, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    @classmethod
    def read_file_and_store_elements(cls, filename):
        try:
            text = ""
            with open(filename, "r") as file:
                for line in file:
                    text += line.strip()
            return text
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

# ========================== TEXT CHUNKING ============================

class TextChunks:
    """
    Handles splitting text into chunks.
    """
    text_splitter = None

    @classmethod
    def initialize(cls, separator=None, chunksize=None, overlap=None):
        try:
            cls.text_splitter = RecursiveCharacterTextSplitter(
                separators=separator,
                chunk_size=chunksize,
                chunk_overlap=overlap
            )
        except Exception as e:
            print(f"Failed to initialize splitter: {e}")

    @classmethod
    def get_text_chunks_doc(cls, text=None):
        try:
            if cls.text_splitter is None:
                print("Text splitter not initialized!")
                return None
            return cls.text_splitter.create_documents([text])
        except Exception as e:
            print(f"Error creating document chunks: {e}")
            return None

# ========================== VECTOR STORE ============================

class Vectors:
    """
    Handles generating vector embeddings and storing them in FAISS.
    """
    embeddings = None

    @classmethod
    def initialize(cls, config: GeminiConfig, api_key: str):
        try:
            cls.embeddings = EmbeddingModel(config=config, api_key=api_key).embeddings
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")

    @classmethod
    def generate_vectors_from_documents(cls, chunks=None):
        try:
            if cls.embeddings is None:
                print("Embedding model not initialized!")
                return None
            return FAISS.from_documents(chunks, embedding=cls.embeddings)
        except Exception as e:
            print(f"Error generating vectors: {e}")
            return None

# ========================== PROMPT TEMPLATES ============================

class PromptTemplates:
    @classmethod
    def key_word_extraction(cls):
        prompt = """
        You are an intelligent assistant.
        Below is the information retrieved from the document:
        {context}
        Answer strictly based on above.
        Question: {question}
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

    @classmethod
    def chain_of_thoughts(cls):
        prompt = """
        You are a thoughtful assistant.
        Here is the document content:
        {context}
        Question: {question}
        Think step by step based ONLY on the provided content.
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

    @classmethod
    def verification_prompt(cls):
        prompt = """
        You are a careful assistant.
        Here is the document content:
        {context}
        Question: {question}
        Verify if the answer is supported by the content.
        """
        return PromptTemplate(template=prompt.strip(), input_variables=["context", "question"])

class PromptManager:
    def __init__(self):
        self.prompt_dict = {
            "key word extraction": PromptTemplates.key_word_extraction,
            "chain of thoughts": PromptTemplates.chain_of_thoughts,
            "verification prompt": PromptTemplates.verification_prompt
        }

    def get_prompt(self, name):
        try:
            func = self.prompt_dict.get(name)
            if not func:
                raise ValueError(f"Prompt '{name}' not found!")
            return func()
        except Exception as e:
            print(f"Error retrieving prompt: {e}")
            return None



# ========================== PREPARE TEXT ============================

class PrepareText:
    """
    Prepares and processes text from PDF files.
    Handles reading, cleaning, chunking, and vectorization.
    """
    def __init__(self, file_path: str, config: GeminiConfig, api_key: str):
        try:
            self.file_path = file_path
            self.config = config
            self.api_key = api_key
            self.raw_text = ReadFile.read_file_text(file_path)
        except Exception as e:
            print(f"Error initializing PrepareText: {e}")
            self.raw_text = ""

    def clean_data(self):
        try:
            return cleantext.clean(
                self.raw_text,
                lowercase=True,
                punct=True,
                extra_spaces=True
            )
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return self.raw_text

    def get_chunks(self, separator=None, chunksize=None, overlap=None):
        try:
            TextChunks.initialize(separator=separator, chunksize=chunksize, overlap=overlap)
            return TextChunks.get_text_chunks_doc(text=self.clean_data())
        except Exception as e:
            print(f"Error creating chunks: {e}")
            return []

    def create_text_vectors(self, separator=None, chunksize=None, overlap=None):
        try:
            Vectors.initialize(config=self.config, api_key=self.api_key)
            return Vectors.generate_vectors_from_documents(
                chunks=self.get_chunks(separator, chunksize, overlap)
            )
        except Exception as e:
            print(f"Error creating vectors: {e}")
            return None

# ========================== QASYSTEM ============================

class QASystem(PrepareText, ChatGoogleGENAI):
    """
    Combines text preparation and ChatGoogleGENAI to create a QA system.
    """
    def __init__(self, file_path: str, config: GeminiConfig, api_key: str,
                 separator=None, chunk_size=None, overlap=None):
        try:
            PrepareText.__init__(self, file_path=file_path, config=config, api_key=api_key)
            ChatGoogleGENAI.__init__(self, config=config, api_key=api_key)
            self.vector_store = self.create_text_vectors(separator, chunk_size, overlap)
        except Exception as e:
            print(f"Error initializing QASystem: {e}")
            self.vector_store = None

    def retrieve_chunks(self, state: QAState):
        """
        Retrieve top-k document chunks relevant to the question.
        """
        try:
            if not self.vector_store:
                print("Vector store not initialized!")
                return state
            docs = self.vector_store.similarity_search(state.question, k=4)
            retrieved_chunks = [doc.page_content for doc in docs]
            state.retrieved_chunks = retrieved_chunks
            return state
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return state

    def answer_questions(self, state: QAState):
        """
        Answer the question based on retrieved chunks using selected prompt template.
        """
        try:
            prompt_manager = PromptManager()
            prompt_template = prompt_manager.get_prompt(state.prompt_type)
            if not prompt_template:
                print("Prompt template not found!")
                return state

            context = "\n\n".join(state.retrieved_chunks)
            prompt = prompt_template.format(context=context, question=state.question)
            response = self.llm.invoke(prompt)

            state.answer = response.content if response else ""
            return state
        except Exception as e:
            print(f"Error answering question: {e}")
            return state

# ========================== GRAPH EXECUTION ============================

class QASystemGraphExecution(QASystem):
    """
    Builds a LangGraph execution graph to automate QA workflow.
    """
    def __init__(self, file_path: str, config: GeminiConfig, api_key: str,
                 separator=None, chunk_size=None, overlap=None):
        try:
            super().__init__(file_path=file_path, config=config, api_key=api_key,
                             separator=separator, chunk_size=chunk_size, overlap=overlap)
        except Exception as e:
            print(f"Error initializing GraphExecution: {e}")

    def build_graph(self):
        try:
            graph = StateGraph(QAState)
            graph.add_node("retrieve", self.retrieve_chunks)
            graph.add_node("QA", self.answer_questions)
            graph.add_edge("retrieve", "QA")
            graph.set_entry_point("retrieve")
            return graph
        except Exception as e:
            print(f"Error building graph: {e}")
            return None

    def answer(self, question: str, prompt_type: str):
        """
        Executes the graph to answer the question using LangGraph flow.
        """
        try:
            graph_executor = self.build_graph()
            if not graph_executor:
                print("Graph not built correctly!")
                return None

            executor = graph_executor.compile()
            initial_state = {"question": question, "retrieved_chunks": [], "answer": "", "prompt_type": prompt_type}
            result = executor.invoke(initial_state)
            return result.get("answer", "")
        except Exception as e:
            print(f"Error executing graph: {e}")
            return None

# ========================== MAIN EXECUTION ============================

if __name__ == "__main__":
    try:
        
        config = GeminiConfig(
            chat_model_name="gemini-2.5-flash",
            embedding_model_name="models/gemini-embedding-001",
            temperature=0.7,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=3000,
            generation_max_tokens=8192
        )

        file_path = "E:/Lang-Graph/data.pdf"

        question = input("Ask your question here: ")
        prompt_type = input("Choose prompt type (key word extraction / chain of thoughts / verification prompt): ")

        qa_system = QASystemGraphExecution(
            file_path=file_path,
            config=config,
            api_key=api_key,
            separator="\n\n",
            chunk_size=4000,
            overlap=300
        )

        answer = qa_system.answer(question=question, prompt_type=prompt_type)
        print("\nQuestion:", question)
        print("Answer:", answer)
    except Exception as e:
        print(f"Error in main execution: {e}")
