
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from rag.pipeline import load_pdf_documents


TEST_CASES = [
    {
        "question": "Where were Docker images published?",
        "expected_term": "GHCR",
    },
    {
        "question": "Which AWS service was used to deploy the backend?",
        "expected_term": "EC2",
    },
    {
        "question": "What technology was used as a reverse proxy?",
        "expected_term": "Nginx",
    },
    {
        "question": "Which tool was used to load-test the API?",
        "expected_term": "k6",
    },
]


def evidence_rank(results, expected_term):
    """Return the 1-based rank of the first result containing the evidence."""
    expected_term = expected_term.casefold()

    for rank, result in enumerate(results, start=1):
        text = result.node.get_content().casefold()

        if expected_term in text:
            return rank

    return None


def show_results(results, expected_term):
    """Print retrieved passages, highlighting the expected evidence in context."""
    for rank, result in enumerate(results, start=1):
        metadata = result.node.metadata
        page = metadata.get("page_number", "?")
        text = " ".join(result.node.get_content().split())

        term_position = text.casefold().find(expected_term.casefold())

        if term_position >= 0:
            start = max(0, term_position - 80)
            end = min(
                len(text),
                term_position + len(expected_term) + 100,
            )
            excerpt = text[start:end]

            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt += "..."
        else:
            excerpt = text[:180]
            if len(text) > 180:
                excerpt += "..."

        print(f"  {rank}. Page {page}: {excerpt}")


def main():
    raw_path = input(
        "Paste the full path to your resume PDF: "
    ).strip().strip('"')

    pdf_path = Path(raw_path)

    if not pdf_path.is_file():
        print("PDF not found. Check the path and try again.")
        return

    # Configure local embeddings. No cloud LLM is needed for this evaluation.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = load_pdf_documents(
        pdf_bytes=pdf_path.read_bytes(),
        file_name=pdf_path.name,
    )

    print("Building the index...")

    # Use smaller, overlapping chunks for a more discriminating retrieval test.
    splitter = SentenceSplitter(
        chunk_size=128,
        chunk_overlap=20,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
    )

    available_nodes = len(index.docstore.docs)
    print(f"Indexed Nodes: {available_nodes}")

    if available_nodes == 0:
        print("No indexed Nodes were found.")
        return

    top_k = min(3, available_nodes)

    # Retriever A: semantic vector search.
    vector_retriever = index.as_retriever(
        similarity_top_k=top_k,
    )

    # Retriever B: BM25 keyword search.
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=top_k,
    )

    # Fuse the two ranked lists with Reciprocal Rank Fusion.
    # MockLLM avoids LlamaIndex's default OpenAI LLM initialization.
    # num_queries=1 means no alternate-query generation is needed.
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        llm=MockLLM(),
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )

    vector_hits = 0
    hybrid_hits = 0

    vector_top1_hits = 0
    hybrid_top1_hits = 0

    vector_mrr = 0.0
    hybrid_mrr = 0.0

    for test in TEST_CASES:
        question = test["question"]
        expected_term = test["expected_term"]

        vector_results = vector_retriever.retrieve(question)
        hybrid_results = hybrid_retriever.retrieve(question)

        vector_rank = evidence_rank(
            vector_results,
            expected_term,
        )
        hybrid_rank = evidence_rank(
            hybrid_results,
            expected_term,
        )

        # Hit@3: evidence appears somewhere in the returned top results.
        vector_hits += int(vector_rank is not None)
        hybrid_hits += int(hybrid_rank is not None)

        # Hit@1: evidence appears in the first result.
        vector_top1_hits += int(vector_rank == 1)
        hybrid_top1_hits += int(hybrid_rank == 1)

        # MRR: reward evidence appearing higher in the ranking.
        if vector_rank is not None:
            vector_mrr += 1 / vector_rank

        if hybrid_rank is not None:
            hybrid_mrr += 1 / hybrid_rank

        print(f"\nQuestion: {question}")
        print(f"Expected evidence: {expected_term}")

        print(
            "\nVector evidence rank: "
            f"{vector_rank if vector_rank is not None else 'MISS'}"
        )
        show_results(vector_results, expected_term)

        print(
            "\nHybrid evidence rank: "
            f"{hybrid_rank if hybrid_rank is not None else 'MISS'}"
        )
        show_results(hybrid_results, expected_term)

    total = len(TEST_CASES)

    print("\n========== SUMMARY ==========")

    print(
        f"Vector Hit@3: {vector_hits}/{total} "
        f"({vector_hits / total:.0%})"
    )
    print(
        f"Hybrid Hit@3: {hybrid_hits}/{total} "
        f"({hybrid_hits / total:.0%})"
    )

    print(
        f"Vector Hit@1: {vector_top1_hits}/{total} "
        f"({vector_top1_hits / total:.0%})"
    )
    print(
        f"Hybrid Hit@1: {hybrid_top1_hits}/{total} "
        f"({hybrid_top1_hits / total:.0%})"
    )

    print(f"Vector MRR: {vector_mrr / total:.3f}")
    print(f"Hybrid MRR: {hybrid_mrr / total:.3f}")


if __name__ == "__main__":
    main()
