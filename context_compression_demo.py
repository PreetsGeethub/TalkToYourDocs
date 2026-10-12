
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Configure local embeddings.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create a document containing relevant and irrelevant sentences.
document = Document(
    text="""
    KK Store is a full-stack ecommerce application.
    Its backend runs on AWS EC2 using Docker.
    GitHub Actions builds a Docker image and publishes it to GHCR.
    The workflow then deploys the updated application to AWS EC2.
    Nginx acts as a reverse proxy for the backend.
    The frontend uses Next.js and TypeScript.
    PostgreSQL stores products, customers, and orders.
    Redis stores conversation history for the AI shopping assistant.
    The production API was load-tested using k6.
    The frontend uses responsive layouts and reusable components.
    """
)


# 3. Build the index and retrieve one relevant Node.
index = VectorStoreIndex.from_documents([document])
retriever = index.as_retriever(similarity_top_k=1)

question = "Where were the Docker images published?"
retrieved_nodes = retriever.retrieve(question)


# 4. Save the original text BEFORE compression.
original_texts = [
    result.node.get_content()
    for result in retrieved_nodes
]


# 5. Configure compression to keep selected sentences only.
compressor = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,
    context_before=0,
    context_after=0,
)

compressed_nodes = compressor.postprocess_nodes(
    retrieved_nodes,
    query_bundle=QueryBundle(query_str=question),
)


# 6. Capture compressed text separately.
compressed_texts = [
    result.node.get_content()
    for result in compressed_nodes
]


def show_text(label, texts):
    print(f"\n--- {label} ---")

    for text in texts:
        print(text)


show_text("ORIGINAL RETRIEVED CONTEXT", original_texts)
show_text("COMPRESSED CONTEXT", compressed_texts)


# 7. Compare character counts.
original_chars = sum(len(text) for text in original_texts)
compressed_chars = sum(len(text) for text in compressed_texts)

print("\n--- COMPRESSION SUMMARY ---")
print(f"Original characters:   {original_chars}")
print(f"Compressed characters: {compressed_chars}")

if original_chars:
    reduction = (1 - compressed_chars / original_chars) * 100
    print(f"Text reduction:        {reduction:.1f}%")
