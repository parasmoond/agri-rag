import streamlit as st
from pathlib import Path
import subprocess
import sys

from src.rag_pipeline import RAGPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriCrop AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    "<style>"
    ".block-container {"
    "max-width: 1200px;"
    "padding-top: 2rem;"
    "padding-bottom: 4rem;"
    "}"
    ".main-title {"
    "font-size: 3.2rem;"
    "font-weight: 800;"
    "letter-spacing: -1px;"
    "margin-bottom: 0.2rem;"
    "}"
    ".main-subtitle {"
    "font-size: 1.1rem;"
    "opacity: 0.65;"
    "margin-bottom: 1rem;"
    "}"
    ".status-box {"
    "padding: 0.5rem 1rem;"
    "border-radius: 20px;"
    "display: inline-block;"
    "background: rgba(46, 204, 113, 0.15);"
    "font-size: 0.85rem;"
    "font-weight: 600;"
    "}"
    ".footer-text {"
    "text-align: center;"
    "opacity: 0.45;"
    "font-size: 0.8rem;"
    "margin-top: 50px;"
    "}"
    "</style>",
    unsafe_allow_html=True
)


# ============================================================
# LOAD RAG PIPELINE
# ============================================================

@st.cache_resource
def load_rag():
    return RAGPipeline(
        model_name="qwen3:1.7b"
    )


# ============================================================
# UPDATE KNOWLEDGE BASE
# ============================================================

def update_knowledge_base():
    ingest_script = PROJECT_ROOT / "scripts" / "ingest.py"
    build_script = PROJECT_ROOT / "scripts" / "build_index.py"

    ingest_result = subprocess.run(
        [
            sys.executable,
            str(ingest_script)
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if ingest_result.returncode != 0:
        return False, (
            "Ingestion failed:\n\n"
            + ingest_result.stdout
            + "\n"
            + ingest_result.stderr
        )

    build_result = subprocess.run(
        [
            sys.executable,
            str(build_script)
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if build_result.returncode != 0:
        return False, (
            "FAISS index rebuild failed:\n\n"
            + build_result.stdout
            + "\n"
            + build_result.stderr
        )

    load_rag.clear()

    return True, (
        ingest_result.stdout
        + "\n"
        + build_result.stdout
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "show_pdf_uploader" not in st.session_state:
    st.session_state.show_pdf_uploader = False

if "selected_pdf_name" not in st.session_state:
    st.session_state.selected_pdf_name = None

if "selected_pdf_bytes" not in st.session_state:
    st.session_state.selected_pdf_bytes = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ RAG Settings")
    st.caption("Configure the retrieval pipeline.")
    st.divider()

    retrieval_k = st.slider(
        "FAISS candidates",
        min_value=5,
        max_value=20,
        value=10,
        help=(
            "Number of chunks retrieved from FAISS "
            "before reranking."
        )
    )

    final_k = st.slider(
        "Final context chunks",
        min_value=1,
        max_value=5,
        value=3,
        help=(
            "Number of chunks passed to Qwen "
            "after reranking."
        )
    )

    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help=(
            "Chunks below this similarity score "
            "are rejected."
        )
    )

    st.divider()

    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.subheader("📚 Knowledge Base")

    # Show PDFs already present in data/raw
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if pdf_files:
        st.caption(f"📄 {len(pdf_files)} PDF(s) in knowledge base")

        for pdf_file in pdf_files:
            st.write(f"• {pdf_file.name}")
    else:
        st.caption("No PDFs added yet.")

    st.write("")

    # --------------------------------------------------------
    # ADD PDF BUTTON
    # --------------------------------------------------------

    if not st.session_state.show_pdf_uploader:

        if st.button(
            "➕ Add PDF",
            use_container_width=True,
            key="open_pdf_uploader"
        ):
            st.session_state.show_pdf_uploader = True
            st.session_state.selected_pdf_name = None
            st.session_state.selected_pdf_bytes = None
            st.rerun()

    # --------------------------------------------------------
    # PDF UPLOAD MODE
    # --------------------------------------------------------

    if st.session_state.show_pdf_uploader:

        st.info(
            "Select a new agricultural PDF, then click "
            "**Process & Add PDF**."
        )

        uploaded_file = st.file_uploader(
            "Upload an agricultural PDF",
            type=["pdf"],
            key="pdf_uploader",
            help="Select a PDF containing agricultural information."
        )

        if uploaded_file is not None:

            st.session_state.selected_pdf_name = uploaded_file.name
            st.session_state.selected_pdf_bytes = uploaded_file.getvalue()

            st.success(
                f"📄 Selected: **{uploaded_file.name}**"
            )

        if (
            st.session_state.selected_pdf_name
            and st.session_state.selected_pdf_bytes
        ):

            if st.button(
                "✅ Process & Add PDF",
                use_container_width=True,
                key="process_pdf_button"
            ):

                filename = st.session_state.selected_pdf_name
                file_bytes = st.session_state.selected_pdf_bytes

                destination = RAW_DATA_DIR / filename

                with open(destination, "wb") as f:
                    f.write(file_bytes)

                with st.spinner(
                    "🔄 Processing PDF and rebuilding FAISS index..."
                ):
                    success, output = update_knowledge_base()

                if success:

                    st.session_state.last_uploaded_file = filename
                    st.session_state.selected_pdf_name = None
                    st.session_state.selected_pdf_bytes = None
                    st.session_state.show_pdf_uploader = False
                    st.session_state.messages = []

                    st.success(
                        f"🎉 {filename} was added successfully!"
                    )

                    with st.expander(
                        "📋 View processing details"
                    ):
                        st.code(
                            output,
                            language="text"
                        )

                    st.rerun()

                else:

                    st.error(
                        "❌ Failed to update the knowledge base."
                    )

                    with st.expander(
                        "View error details"
                    ):
                        st.code(
                            output,
                            language="text"
                        )

            if st.button(
                "✖ Cancel",
                use_container_width=True,
                key="cancel_pdf_upload"
            ):
                st.session_state.show_pdf_uploader = False
                st.session_state.selected_pdf_name = None
                st.session_state.selected_pdf_bytes = None
                st.rerun()

    if st.session_state.last_uploaded_file:
        st.caption(
            "✅ Last added: "
            + st.session_state.last_uploaded_file
        )

    st.divider()

    # ========================================================
    # MODELS
    # ========================================================

    st.subheader("🧠 Models")

    st.code(
        "Embedding\n"
        "all-MiniLM-L6-v2\n\n"
        "Reranker\n"
        "ms-marco-MiniLM-L-6-v2\n\n"
        "LLM\n"
        "qwen3:1.7b\n\n"
        "Vector Store\n"
        "FAISS",
        language="text"
    )

    st.divider()

    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader("🔄 Pipeline")

    st.markdown(
        """
**Question**

↓

**Embedding**

↓

**FAISS Top-K**

↓

**Cross-Encoder**

↓

**Top-K Context**

↓

**Qwen**

↓

**Grounded Answer**
"""
    )

    st.divider()

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌾 AgriCrop AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Agricultural Knowledge Assistant powered by '
    'Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status-box">'
    '🟢 Local AI • Grounded in your documents'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        "## 🌱 Ask something from your agricultural knowledge base"
    )

    st.caption(
        "The system retrieves relevant passages, "
        "reranks them, and generates a grounded answer."
    )

    st.write("")

    st.subheader("💡 Try an example")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🦠 Rice blast symptoms",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What are the symptoms of rice blast disease?"
            )
            st.rerun()

    with col2:
        if st.button(
            "🌱 Rice blast management",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "How can rice blast disease be managed?"
            )
            st.rerun()

    with col3:
        if st.button(
            "🔬 Cause of rice blast",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What causes rice blast disease?"
            )
            st.rerun()

    st.divider()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤"
        ):
            st.write(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🌾"
        ):

            st.markdown(
                message["content"]
            )

            sources = message.get(
                "sources",
                []
            )

            if sources:

                st.markdown(
                    "### 📚 Sources"
                )

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    with st.expander(
                        f"📄 Source {i} — "
                        f"{source['source']} — "
                        f"Page {source['page']}"
                    ):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "FAISS Similarity",
                                f"{source['score']:.4f}"
                            )

                        with col2:
                            st.metric(
                                "Reranker Score",
                                f"{source['rerank_score']:.4f}"
                            )

                        st.markdown(
                            "**Retrieved passage**"
                        )

                        st.write(
                            source["text"]
                        )


# ============================================================
# QUESTION INPUT
# ============================================================

pending_question = st.session_state.pending_question
st.session_state.pending_question = None

question = st.chat_input(
    "Ask an agricultural question..."
)

if pending_question:
    question = pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "assistant",
            avatar="🌾"
        ):

            with st.spinner(
                "🔎 Searching agricultural knowledge..."
            ):

                rag = load_rag()

                result = rag.answer(
                    question=question,
                    retrieval_k=retrieval_k,
                    final_k=final_k,
                    similarity_threshold=similarity_threshold
                )

            answer = result["answer"]
            sources = result["sources"]
            answer_lower = answer.lower()

            insufficient = (
                not sources
                or "don't have enough information"
                in answer_lower
                or "do not have enough information"
                in answer_lower
            )

            if insufficient:

                st.warning(
                    "⚠️ The available agricultural "
                    "documents do not contain enough "
                    "information to answer this question."
                )

            st.markdown(answer)

            if sources:

                st.markdown("### 📚 Sources")

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    with st.expander(
                        f"📄 Source {i} — "
                        f"{source['source']} — "
                        f"Page {source['page']}"
                    ):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "FAISS Similarity",
                                f"{source['score']:.4f}"
                            )

                        with col2:
                            st.metric(
                                "Reranker Score",
                                f"{source['rerank_score']:.4f}"
                            )

                        st.markdown(
                            "**Retrieved passage**"
                        )

                        st.write(
                            source["text"]
                        )

            else:

                st.info(
                    "No relevant document passages "
                    "were retrieved."
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌾 AgriCrop AI • "
    "FAISS • Cross-Encoder • Ollama • "
    "Retrieval-Augmented Generation"
)
