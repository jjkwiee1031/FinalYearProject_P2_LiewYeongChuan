import os

# Suppress TensorFlow and other backend warnings before they are loaded
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import csv
import time

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Hybrid Retrieval Imports
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ArxivRAGSystem:
    def __init__(
        self, 
        dir_path="C:/Users/jjkwiee/Documents/myPython/Master/FYP/CORE/db/", 
        db_list_file="db_list.csv", 
        enhancer_model_name="llama3.2:1b",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.dir_path = dir_path
        self.db_list = db_list_file
        self.retriever = None

        if not os.path.exists(self.db_list):
            with open(self.db_list, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["embeddings", "db"])  
                
        self.enhancement_template = """
Convert the user's question into a short search query (maximum 5 words) for an arXiv database.
Do NOT include bullet points, explanations, or labels. ONLY output the search keywords.

User Question: {question}
Query:"""
        self.enhancer_model = OllamaLLM(model=enhancer_model_name)
        self.enhancement_prompt = ChatPromptTemplate.from_template(self.enhancement_template)
        self.enhancement_chain = self.enhancement_prompt | self.enhancer_model

        self.template = """
You are an chatbot assistant specialized in analyzing research papers from the arXiv dataset.
Your task is to generate a direct and short answer to the user's question by synthesizing
the key insights from the retrieved documents provided below.

Guidelines:
1. Base your answer only on the retrieved documents; do not hallucinate or add unsupported claims.
2. Summarize the relevant findings, arguments, and conclusions from the documents.
3. If documents contain conflicting views, present both perspectives clearly.
4. You are not necessary to use all the references given.
5. Write in an academic and objective tone, keep it concise and direct (in 2-3 sentences).
6. If the documents do not contain the information needed to answer the question, respond with "No Answer" only.

Retrieved Documents:
{references}

User Question:
{question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

        print("Loading Cross-Encoder model...")
        self.cross_encoder_model = HuggingFaceCrossEncoder(model_name=cross_encoder_model_name)

    def list_databases(self):
        """Returns the list of available databases and their embedding models."""
        if os.path.exists(self.db_list):
            df = pd.read_csv(self.db_list, dtype={"embeddings": str, "db": str})
            return df.to_dict(orient="records")
        return []

    def switch_db(self, db_name, embeddings_model_name, k_value=10, n_value=5):
        """Loads a specific database and configures the hybrid retriever pipeline."""
        safe_model_name = embeddings_model_name.replace(":", "-").replace("/", "-")
        db_location = os.path.join(self.dir_path, safe_model_name, db_name)
        embeddings = OllamaEmbeddings(model=embeddings_model_name)

        vector_store = Chroma(
            collection_name="arxiv",
            persist_directory=db_location,
            embedding_function=embeddings
        )       

        dense_retriever = vector_store.as_retriever(search_kwargs={"k": int(k_value)})
        
        # Build BM25 index from all documents in the Chroma collection
        db_data = vector_store.get(include=["documents", "metadatas"])
        texts = db_data["documents"]
        metadatas = db_data["metadatas"]
        docs_for_bm25 = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]
        
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = int(k_value)
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever], 
            weights=[0.5, 0.5]
        )
        
        compressor = CrossEncoderReranker(model=self.cross_encoder_model, top_n=int(n_value))
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        
        print(f"Loaded Hybrid Database '{db_name}'.")
        print(f"Vector store size (docs for BM25): {len(texts)}")
        print(f"Hybrid configs: Dense K={k_value}, BM25 K={k_value}, Final Top N={n_value}")

    def create_new_db(self, csv_file_path, db_name, embeddings_model_name):
        """Processes a CSV into Vectors and saves the database (Replaces /new route)."""
        safe_model_name = embeddings_model_name.replace(":", "-").replace("/", "-")
        db_location = os.path.join(self.dir_path, safe_model_name, db_name)
        embeddings = OllamaEmbeddings(model=embeddings_model_name)
        
        print(f"Reading CSV '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path)

        vector_store = Chroma(
            collection_name="arxiv",
            persist_directory=db_location,
            embedding_function=embeddings
        )

        documents = []
        ids = []

        for i, row in df.iterrows():
            clean_summary = str(row['summary']).replace('\n', ' ')
            document = Document(
                page_content=f"Title: {row['title']}.\nPublished_date: {row['published_date']}.\nSummary: {clean_summary}",
                metadata={"id": str(row['id'])}
            )
            documents.append(document)
            ids.append(str(i))
        
        print("Splitting documents to fit context length...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = text_splitter.split_documents(documents)
        
        split_ids = [f"{doc.metadata['id']}_{i}" for i, doc in enumerate(split_docs)]
        
        print(f"Generating embeddings and adding to Chroma in batches (total {len(split_docs)} chunks)...")
        batch_size = 20
        for i in range(0, len(split_docs), batch_size):
            batch_docs = split_docs[i:i+batch_size]
            batch_ids = split_ids[i:i+batch_size]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                    print(f"  Processed batch {i//batch_size + 1}/{(len(split_docs) + batch_size - 1)//batch_size}")
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"  Error on batch {i//batch_size + 1}, attempt {attempt + 1}: {e}")
                    if "context length" in error_msg or "400" in error_msg:
                        import sys
                        print(f"\n❌ FATAL: Context limit exceeded.")
                        lengths = [len(doc.page_content) for doc in batch_docs]
                        print(f"  Maximum chunk size in this batch was: {max(lengths)} characters.")
                        print(f"  Current TextSplitter chunk_size is set to: {text_splitter._chunk_size}")
                        print("  Ending the run to allow parameter adjustments.")
                        sys.exit(1)
                        
                    if attempt < max_retries - 1:
                        print("  Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        raise e  # re-raise if it still fails
        
        with open(self.db_list, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([embeddings_model_name, db_name])
            
        print(f"Database successfully saved to '{db_location}'.")

    def switch_generator(self, model_name):
        """Switches the generation model for answering questions."""
        print(f"Switching generator model to '{model_name}'...")
        self.model = OllamaLLM(model=model_name)
        self.chain = self.prompt | self.model
        print(f"Generator model successfully changed to '{model_name}'.")

    def ask(self, question, return_docs=False, return_doc_texts=False):
        """Processes a user question, retrieves context, and generates an answer."""
        if not self.retriever:
            raise ValueError("Retriever is not initialized. Call switch_db() first before querying.")

        print(f"\nReceived question: {question}")

        try:
            start_enhance = time.time()
            enhanced_query = self.enhancement_chain.invoke({"question": question}).strip()
            enhance_time = time.time() - start_enhance
            print(f"Enhanced query: {enhanced_query}")
            print(f"Enhancement time: {enhance_time:.2f} s")

            start_retrieve = time.time()
            reference_docs = self.retriever.invoke(enhanced_query)
            retrieve_time = time.time() - start_retrieve

            raw_ids = [doc.metadata.get("id", getattr(doc, "id", "Unknown")) for doc in reference_docs]
            retrieved_ids = []
            for d in raw_ids:
                if d not in retrieved_ids:
                    retrieved_ids.append(d)
            print(f"Retrieved Document IDs: {retrieved_ids}")

            start_gen = time.time()
            result = self.chain.invoke({"references": reference_docs, "question": question})
            gen_time = time.time() - start_gen
            
            print(f"Retriever time: {retrieve_time:.2f} s")
            print(f"Generation time: {gen_time:.2f} s")

            if return_doc_texts:
                return result, retrieved_ids, [doc.page_content for doc in reference_docs]
            if return_docs:
                return result, retrieved_ids
            return result

        except Exception as e:
            print(f"An error occurred while processing the question: {e}")
            if return_doc_texts:
                return None, [], []
            if return_docs:
                return None, []
            return None

def test():
    rag = ArxivRAGSystem()

    rag.switch_db(
        db_name="df_2024_Artificial Intelligence", 
        embeddings_model_name="all-minilm", 
        k_value=10
    )
    rag.switch_generator("llama3.2:1b")
    rag.ask("What is artificial intelligence?")

if __name__ == "__main__":
    test()

