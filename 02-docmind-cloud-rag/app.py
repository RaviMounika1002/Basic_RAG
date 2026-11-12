
import hashlib

import streamlit as st

from rag import DocMindRAG


st.set_page_config(
    page_title="DocMind | Cloud RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocMind")
st.caption("Ask questions about your PDFs using Cloud RAG")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@st.cache_resource
def create_engine():
    return DocMindRAG(
        openai_key=st.secrets["OPENAI_API_KEY"],
        supabase_url=st.secrets["SUPABASE_URL"],
        supabase_key=st.secrets["SUPABASE_SERVICE_ROLE_KEY"],
    )


try:
    engine = create_engine()
except Exception:
    st.error(
        "Cloud configuration is missing or invalid. "
        "Check your Streamlit secrets."
    )
    st.stop()


if "document" not in st.session_state:
    st.session_state.document = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Document workspace")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()

        if len(file_bytes) > MAX_FILE_SIZE:
            st.error("Maximum file size is 10 MB.")

        else:
            current_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            if current_hash != st.session_state.file_hash:
                if st.button(
                    "Process PDF",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "Extracting, embedding, and storing..."
                    ):
                        try:
                            result = engine.ingest_pdf(
                                uploaded_file.name,
                                file_bytes,
                            )

                            st.session_state.document = result
                            st.session_state.file_hash = current_hash
                            st.session_state.messages = []

                            st.success("PDF processed successfully!")

                        except Exception as error:
                            st.error(
                                f"Processing failed: {error}"
                            )
            else:
                st.success("This PDF is already processed.")

    if st.session_state.document:
        document = st.session_state.document

        st.divider()
        st.subheader("Active document")
        st.write(document["filename"])

        col1, col2 = st.columns(2)

        col1.metric("Pages", document["pages"])
        col2.metric("Chunks", document["chunks"])

        if st.button("Delete document data"):
            try:
                engine.db.delete_document(
                    document["document_id"]
                )
                st.session_state.document = None
                st.session_state.file_hash = None
                st.session_state.messages = []
                st.rerun()
            except Exception:
                st.error("Could not delete document data.")


if not st.session_state.document:
    st.info(
        "Upload a PDF in the sidebar and click Process PDF "
        "to start asking questions."
    )
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Page {source['page_number']}** "
                        f"— Similarity: {source['similarity']:.3f}"
                    )
                    st.write(source["content"])
                    st.divider()


question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                result = engine.ask(
                    document_id=st.session_state.document[
                        "document_id"
                    ],
                    question=question,
                )

                st.markdown(result["answer"])

                with st.expander("View retrieved sources"):
                    for source in result["sources"]:
                        st.markdown(
                            f"**Page {source['page_number']}** "
                            f"— Similarity: "
                            f"{source['similarity']:.3f}"
                        )
                        st.write(source["content"])
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception:
                st.error(
                    "Unable to generate an answer. "
                    "Check the cloud service configuration "
                    "and logs."
                )
