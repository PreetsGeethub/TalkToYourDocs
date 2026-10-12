
import streamlit as st

from rag.pipeline import create_query_engine


st.set_page_config(
    page_title="Talk to Your Docs",
    page_icon="📄",
)

st.title("📄 Talk to Your Docs")
st.write("Upload a PDF and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)

if uploaded_file is not None:
    file_signature = (
        uploaded_file.name,
        uploaded_file.size,
    )

    # Clear the old index when a different PDF is selected.
    if (
        st.session_state.get("indexed_file_signature")
        != file_signature
    ):
        st.session_state.pop("query_engine", None)
        st.session_state.pop("indexed_file_signature", None)

    if st.button("Index PDF", type="primary"):
        try:
            with st.spinner("Reading and indexing your PDF..."):
                query_engine = create_query_engine(
                    pdf_bytes=uploaded_file.getvalue(),
                    file_name=uploaded_file.name,
                )

            st.session_state.query_engine = query_engine
            st.session_state.indexed_file_signature = file_signature

            st.success("PDF indexed successfully!")

        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")


if "query_engine" in st.session_state:
    st.divider()
    st.subheader("Ask your document")

    question = st.text_input(
        "What would you like to know?"
    )

    if st.button("Ask") and question.strip():
        try:
            with st.spinner("Finding an answer..."):
                # Call the query engine only once.
                response = st.session_state.query_engine.query(
                    question
                )

            answer = str(response).strip()
            source_nodes = getattr(
                response,
                "source_nodes",
                [],
            ) or []

            # Temporary diagnostics.
            st.write("Debug — answer characters:", len(answer))
            st.write("Debug — source passages:", len(source_nodes))

            if source_nodes:
                with st.expander("Inspect retrieved passages"):
                    for i, source in enumerate(
                        source_nodes,
                        start=1,
                    ):
                        st.write(f"Passage {i}")
                        st.write(source.node.get_content())

            if answer:
                st.markdown("### Answer")
                st.write(answer)
            else:
                st.warning(
                    "The query engine returned an empty answer. "
                    "Check the terminal and the debug information."
                )

        except Exception as exc:
            st.error(f"Could not answer your question: {exc}")
