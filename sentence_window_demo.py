
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Configure local embeddings.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create a sample document.
document = Document(
    text="""
    KK Store is a full-stack ecommerce application.
    Its backend runs on AWS EC2 using Docker.
    Nginx acts as a reverse proxy for the backend.
    GitHub Actions checks out the source code and runs backend tests.
    The workflow builds a Docker image and publishes it to GHCR.
    It then deploys the updated application to AWS EC2.
    PostgreSQL stores products, customers, and orders.
    Redis stores conversation history for the AI shopping assistant.
    The frontend uses Next.js and TypeScript.
    The production API was load-tested using k6.
    """
)


# 3. Create one Node per sentence and store its surrounding window.
parser = SentenceWindowNodeParser.from_defaults(
    window_size=2,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

nodes = parser.get_nodes_from_documents([document])

print(f"Sentence Nodes created: {len(nodes)}")


# 4. Index the sentence Nodes.
index = VectorStoreIndex(nodes)


# 5. Retrieve the most relevant individual sentences.
retriever = index.as_retriever(
    similarity_top_k=3
)


# 6. Prepare the postprocessor that restores surrounding context.
window_postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)


# 7. Run a query.
question = "Where were the Docker images published?"

retrieved_nodes = retriever.retrieve(question)


# 8. Display the original retrieved sentences.
print("\n--- ORIGINAL RETRIEVED SENTENCES ---")

for rank, result in enumerate(retrieved_nodes, start=1):
    print(f"\nResult {rank}:")
    print(result.node.get_content())


# 9. Replace each matched sentence with its stored window.
window_nodes = window_postprocessor.postprocess_nodes(
    retrieved_nodes
)


# 10. Display the expanded context.
print("\n--- SENTENCE-WINDOW CONTEXT ---")

for rank, result in enumerate(window_nodes, start=1):
    print(f"\nResult {rank}:")
    print(result.node.get_content())
