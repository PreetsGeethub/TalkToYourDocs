
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Configure local embeddings.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create a sample document.
document = Document(
    text="""
    KK Store is a full-stack ecommerce application.

    The backend runs on AWS EC2. Docker packages the backend
    application into a container. Nginx acts as a reverse proxy,
    and HTTPS secures communication with the application.

    The GitHub Actions workflow checks out the source code,
    installs dependencies, runs backend tests, and generates
    the Prisma Client. It then builds a Docker image, publishes
    that image to GitHub Container Registry (GHCR), and deploys
    the updated application to AWS EC2.

    The application uses PostgreSQL to store products, customers,
    and orders. Prisma provides database access to the Node.js
    backend. Redis is used for conversation history in the AI
    shopping assistant.

    The frontend uses Next.js and TypeScript. It communicates
    with the backend API to retrieve products and submit orders.
    The backend validates requests and checks inventory before
    processing an order.

    The production API was load-tested using k6. The test covered
    product-list and product-detail endpoints and measured response
    times under concurrent requests.

    During deployment, GitHub Actions builds and publishes the
    Docker image before updating the backend running on EC2.
    Nginx routes incoming requests to the application, while
    HTTPS provides encrypted communication.
    """,
    metadata={"source": "sample_kk_store"},
)


# 3. Create a hierarchy of parent and child Nodes.
parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[128, 32]
)

all_nodes = parser.get_nodes_from_documents([document])
leaf_nodes = get_leaf_nodes(all_nodes)

print(f"All hierarchical Nodes: {len(all_nodes)}")
print(f"Leaf Nodes to search: {len(leaf_nodes)}")


# 4. Store every Node, including the parent Nodes.
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(all_nodes)


# 5. Create a vector index from only the child Nodes.
index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)


# 6. Create the ordinary vector retriever.
top_k = min(10, len(leaf_nodes))

vector_retriever = index.as_retriever(
    similarity_top_k=top_k
)


# 7. Wrap it with the parent-aware retriever.
auto_merging_retriever = AutoMergingRetriever(
    vector_retriever,
    storage_context,
    simple_ratio_thresh=0.4,
    verbose=True,
)


# 8. Compare retrieved children with auto-merged results.
question = (
    "Explain how GitHub Actions publishes Docker images "
    "and deploys the backend."
)

print("\nQUESTION:")
print(question)


def display_results(label, results):
    print(f"\n--- {label} ---")

    for rank, result in enumerate(results, start=1):
        text = result.node.get_content()
        print(f"\nResult {rank}")
        print(f"Node ID: {result.node.node_id}")
        print(text)


ordinary_results = vector_retriever.retrieve(question)

merged_results = auto_merging_retriever.retrieve(question)

display_results("ORDINARY VECTOR RETRIEVAL", ordinary_results)
display_results("AUTO-MERGING RETRIEVAL", merged_results)
