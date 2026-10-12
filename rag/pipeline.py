import os
from io import BytesIO

from dotenv import load_dotenv
from pypdf import PdfReader

from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import os

from llama_index.postprocessor.cohere_rerank import CohereRerank

def configure_models():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. "
            "Please check your .env file."
        )

    # Generate embeddings locally.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Generate answers using OpenRouter's cloud model.
    Settings.llm = OpenRouter(
    model="openrouter/free",
    api_key=api_key,
    max_tokens=512,
    context_window=4096,
)

def load_pdf_documents(
    pdf_bytes: bytes,
    file_name: str,
) -> list[Document]:
    reader = PdfReader(BytesIO(pdf_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": file_name,
                    "page_number": page_number,
                },
            )
        )

    if not documents:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "It may be scanned or image-based."
        )

    return documents


def create_query_engine(pdf_bytes: bytes, file_name: str):
    # Configure embeddings and the cloud LLM.
    configure_models()

    # Extract PDF pages into LlamaIndex Documents.
    documents = load_pdf_documents(
        pdf_bytes=pdf_bytes,
        file_name=file_name,
    )

    # Build an in-memory vector index.
    index = VectorStoreIndex.from_documents(documents)

    # Guide the LLM toward short, grounded answers.
    qa_prompt = PromptTemplate(
        """
        Answer the question using only the provided context.

        Give a direct answer in 2-4 sentences.
        Do not output your thinking process or step-by-step analysis.
        Do not invent facts, qualifications, or experience.
        If the context does not provide enough evidence, say so.

        Context:
        {context_str}

        Question:
        {query_str}

        Answer:
        """
    )

    # Retrieve up to three relevant nodes for each question.
    # 1. Create the semantic vector retriever.
    vector_retriever = index.as_retriever(
        similarity_top_k=5
    )

    # 2. Create the BM25 keyword retriever.
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=5,
    )

    # 3. Combine both retrievers using Reciprocal Rank Fusion.
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        similarity_top_k=8,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )

    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        raise ValueError(
            "COHERE_API_KEY not found. Check your .env file."
        )

    cohere_reranker = CohereRerank(
        api_key=cohere_api_key,
        model="rerank-v3.5",
        top_n=3,
    )
    # 4. Connect hybrid retrieval to answer generation.
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=Settings.llm,
        text_qa_template=qa_prompt,
         node_postprocessors=[cohere_reranker],
    )

    return query_engine